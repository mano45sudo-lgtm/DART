"""
Colab-friendly episode-level REINFORCE logging (env rollouts, not static data).
Used by the multi-model Colab notebook for JSON + figure export.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from env.digital_twin_env import DigitalTwinDiabetesEnv  # noqa: E402
from evaluation.baseline_random_agent import RandomAgent  # noqa: E402
from training.llm_reinforce import reinforce_loss_on_episode, rollout_episode  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def collect_random_baseline(
    *,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
    model_key: str = "random",
) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, float]]]:
    """
    Return (rows as dicts, per_episode_return, one_glucose_traj for first ep).
    """
    agent = RandomAgent(seed=seed)
    rows: List[Dict[str, Any]] = []
    running = 0.0
    first_glucose: List[float] = []
    for i in range(1, n_episodes + 1):
        env = DigitalTwinDiabetesEnv(seed=seed + i, max_steps=max_steps)
        obs, _ = env.reset(seed=seed + i)
        ret = 0.0
        nact = 0
        glist: List[float] = [float(obs["fasting_glucose"])]
        done = False
        while not done:
            a = agent.act(obs)
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            nact += 1
            glist.append(float(obs["fasting_glucose"]))
            done = bool(term or trunc)
        if i == 1:
            first_glucose = glist
        running += ret
        rows.append(
            {
                "episode": i,
                "reward": ret,
                "avg_reward": running / i,
                "action_count": nact,
                "model": model_key,
            }
        )
    return rows, [float(r["reward"]) for r in rows], first_glucose


def collect_trained_episode_glucose(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    *,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    env_seed: int = 9_999,
) -> List[float]:
    """Single rollout for behavior plot."""
    from training.llm_reinforce import obs_to_prompt, sample_action_via_generate

    env = DigitalTwinDiabetesEnv(seed=env_seed, max_steps=max_steps)
    obs, _ = env.reset(seed=env_seed)
    g: List[float] = [float(obs["fasting_glucose"])]
    done = False
    model.eval()
    with torch.no_grad():
        while not done:
            prompt = obs_to_prompt(obs)
            _ids, _q, _t, action, _ = sample_action_via_generate(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            obs, _r, term, trunc, _ = env.step(action)
            g.append(float(obs["fasting_glucose"]))
            done = bool(term or trunc)
    return g


def train_reinforce_with_episode_log(
    *,
    model_id: str,
    short_label: str,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    updates: int,
    episodes_per_update: int,
    max_steps: int,
    train_seed_base: int,
    lr: float = 2e-5,
    max_new_tokens: int = 48,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> Tuple[List[Dict[str, Any]], torch.nn.Module, Any, torch.device]:
    """REINFORCE on live env; one JSON-like row per training episode. Returns (rows, model, tokenizer, device)."""
    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit model requires a CUDA runtime")

    dev_s = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        device = torch.device(next(m.parameters()).device)
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        m.to(device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rows: List[Dict[str, Any]] = []
    total_reward_accum = 0.0
    ep_count = 0
    for u in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        ep_losses: List[torch.Tensor] = []
        for e in range(episodes_per_update):
            ep_count += 1
            env_seed = train_seed_base + u * 1000 + e * 17
            steps, ep_ret = rollout_episode(
                m,
                tok,
                env_seed=env_seed,
                max_steps=max_steps,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            total_reward_accum += ep_ret
            loss_ep = reinforce_loss_on_episode(steps, gamma=1.0)
            if loss_ep is not None:
                ep_losses.append(loss_ep)
            rows.append(
                {
                    "episode": ep_count,
                    "reward": float(ep_ret),
                    "avg_reward": float(total_reward_accum / ep_count),
                    "action_count": len(steps),
                    "model": short_label,
                }
            )
        if ep_losses:
            loss = torch.stack(ep_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return rows, m, tok, device


def council_self_repair_episode_log(
    *,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Rule-based council + SelfImprovementController. Returns (rows, episode_indices_with_repair_signal).
    Repair marked when use_fallback is True or exploration jumped vs previous episode end.
    """
    from council import build_default_council
    from self_improvement import SelfImprovementController
    from training.council_rollout import run_council_episode

    council = build_default_council(seed=seed)
    ctrl = SelfImprovementController()
    rows: List[Dict[str, Any]] = []
    repair_episodes: List[int] = []
    total_r = 0.0
    prev_ex = 0.0
    for i in range(1, n_episodes + 1):
        ep_seed = seed + i * 31
        total, det = run_council_episode(ep_seed, max_steps, council, ctrl, log=True)
        n_steps = len(det.get("steps", [])) or 1
        gstd = float(det.get("glucose_fasting_stdev", 0.0) or 0.0)
        ctrl.update(float(total), last_episode_fasting_std=gstd)
        ctrl.adjust_council(council)
        ctrl.adjust_exploration()
        total_r += float(total)
        sn = ctrl.snapshot()
        ex = float(sn.get("exploration", 0.0) or 0.0)
        if sn.get("use_fallback", 0) > 0.5 or ex > prev_ex + 0.05:
            repair_episodes.append(i)
        prev_ex = ex
        rows.append(
            {
                "episode": i,
                "reward": float(total),
                "avg_reward": float(total_r / i),
                "action_count": n_steps,
                "model": "council_self_repair",
            }
        )
    return rows, repair_episodes
