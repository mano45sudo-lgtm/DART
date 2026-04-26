"""
Colab-friendly episode-level REINFORCE logging (env rollouts, not static data).
Used by the multi-model Colab notebook for JSON + figure export.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _empty_clinical_trace() -> Dict[str, Any]:
    return {
        "week": [],
        "fasting_glucose": [],
        "hba1c": [],
        "egfr": [],
        "bmi": [],
        "systolic_bp": [],
        "step_reward": [],
        "cumulative_return": [],
        "action_type": [],
        "reward_components": [],
        "weekly_cost_usd": [],
        "n_side_effects": [],
    }


def _append_initial_observation(trace: Dict[str, Any], obs: Dict[str, Any]) -> None:
    trace["week"].append(int(obs["week"]))
    trace["fasting_glucose"].append(float(obs["fasting_glucose"]))
    trace["hba1c"].append(float(obs["hba1c"]))
    trace["egfr"].append(float(obs["egfr"]))
    trace["bmi"].append(float(obs["bmi"]))
    trace["systolic_bp"].append(float(obs["systolic_bp"]))
    trace["weekly_cost_usd"].append(0.0)
    trace["n_side_effects"].append(0.0)


def _append_post_step(
    trace: Dict[str, Any], reward: float, info: Dict[str, Any], next_obs: Dict[str, Any], cum: float
) -> float:
    trace["fasting_glucose"].append(float(next_obs["fasting_glucose"]))
    trace["hba1c"].append(float(next_obs["hba1c"]))
    trace["egfr"].append(float(next_obs["egfr"]))
    trace["bmi"].append(float(next_obs["bmi"]))
    trace["systolic_bp"].append(float(next_obs["systolic_bp"]))
    trace["week"].append(int(next_obs["week"]))
    r = float(reward)
    new_cum = float(cum + r)
    trace["step_reward"].append(r)
    trace["cumulative_return"].append(new_cum)
    rc = info.get("reward", {}) or {}
    if isinstance(rc, dict):
        trace["reward_components"].append({k: float(v) for k, v in rc.items() if isinstance(v, (int, float))})
    else:
        trace["reward_components"].append({})
    a = info.get("action", {}) or {}
    trace["action_type"].append(str(a.get("type", "noop")))
    trace["weekly_cost_usd"].append(float(info.get("weekly_cost_usd", 0.0) or 0.0))
    se = list(info.get("side_effects") or [])
    trace["n_side_effects"].append(float(len(se)))
    return new_cum


def rollout_clinical_trace(
    policy: Callable[[Dict[str, Any]], Any],
    *,
    env_seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    """
    One full episode: partial observations, decomposed rubric, costs, and action types.
    `policy(obs)` must return a JSON action dict the env accepts.
    """
    env = DigitalTwinDiabetesEnv(seed=env_seed, max_steps=max_steps)
    obs, _ = env.reset(seed=env_seed)
    trace = _empty_clinical_trace()
    _append_initial_observation(trace, obs)
    cum = 0.0
    done = False
    while not done:
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        cum = _append_post_step(trace, reward, info, obs, cum)
        done = bool(term or trunc)
    return trace


def rollout_clinical_trace_random(*, env_seed: int, max_steps: int, random_seed: int = 0) -> Dict[str, Any]:
    agent = RandomAgent(seed=random_seed)
    return rollout_clinical_trace(lambda o: agent.act(o), env_seed=env_seed, max_steps=max_steps)


def rollout_clinical_trace_trained(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    *,
    env_seed: int,
    max_steps: int,
    max_new_tokens: int = 48,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> Dict[str, Any]:
    from training.llm_reinforce import obs_to_prompt, sample_action_via_generate

    model.eval()

    def policy(obs: Dict[str, Any]) -> Any:
        with torch.no_grad():
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
        return action

    return rollout_clinical_trace(policy, env_seed=env_seed, max_steps=max_steps)


def collect_council_clinical_trace(
    *,
    env_seed: int,
    max_steps: int,
    council_seed: int = 0,
) -> Dict[str, Any]:
    from council import build_default_council
    from self_improvement import SelfImprovementController

    council = build_default_council(seed=council_seed)
    ctrl = SelfImprovementController()
    env = DigitalTwinDiabetesEnv(seed=env_seed, max_steps=max_steps)
    obs, _ = env.reset(seed=env_seed)
    trace = _empty_clinical_trace()
    _append_initial_observation(trace, obs)
    cum = 0.0
    done = False
    while not done:
        ex = council.decide(
            obs,
            env,
            use_fallback=bool(ctrl.use_fallback),
            exploration=float(ctrl.exploration),
        )
        action = ex["final_action"]
        obs, reward, term, trunc, info = env.step(action)
        cum = _append_post_step(trace, reward, info, obs, cum)
        done = bool(term or trunc)
    return trace


def collect_episode_endpoints(
    policy: Callable[[Dict[str, Any]], Any],
    *,
    n_episodes: int,
    max_steps: int,
    seed: int,
    model_key: str,
) -> List[Dict[str, Any]]:
    """Run N episodes; record terminal metrics for distribution / box plots."""
    out: List[Dict[str, Any]] = []
    for i in range(1, n_episodes + 1):
        es = int(seed) + int(i) * 97
        env = DigitalTwinDiabetesEnv(seed=es, max_steps=max_steps)
        obs, _ = env.reset(seed=es)
        ret = 0.0
        n = 0
        done = False
        while not done:
            a = policy(obs)
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            n += 1
            if term or trunc:
                break
        out.append(
            {
                "model": model_key,
                "episode": i,
                "return": float(ret),
                "final_hba1c": float(obs["hba1c"]),
                "final_fpg": float(obs["fasting_glucose"]),
                "final_egfr": float(obs["egfr"]),
                "n_steps": n,
            }
        )
    return out


def collect_endpoints_trained(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    *,
    n_episodes: int,
    max_steps: int,
    seed: int,
    model_key: str,
    max_new_tokens: int = 48,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> List[Dict[str, Any]]:
    from training.llm_reinforce import obs_to_prompt, sample_action_via_generate

    model.eval()
    res: List[Dict[str, Any]] = []
    for i in range(1, n_episodes + 1):
        es = int(seed) + int(i) * 97
        env = DigitalTwinDiabetesEnv(seed=es, max_steps=max_steps)
        obs, _ = env.reset(seed=es)
        ret = 0.0
        n = 0
        done = False
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
                obs, r, term, trunc, _ = env.step(action)
                ret += float(r)
                n += 1
                done = bool(term or trunc)
        res.append(
            {
                "model": model_key,
                "episode": i,
                "return": float(ret),
                "final_hba1c": float(obs["hba1c"]),
                "final_fpg": float(obs["fasting_glucose"]),
                "final_egfr": float(obs["egfr"]),
                "n_steps": n,
            }
        )
    return res


def collect_endpoints_random_baseline(
    *,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
    model_key: str = "random",
) -> List[Dict[str, Any]]:
    ag = RandomAgent(seed=seed)
    return collect_episode_endpoints(
        lambda o: ag.act(o),
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        model_key=model_key,
    )


def action_type_histogram(trace: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not trace or not trace.get("action_type"):
        return {}
    c = Counter(str(t) for t in trace["action_type"])
    return {k: int(c[k]) for k in sorted(c.keys())}


def sum_reward_components(trace: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not trace or not trace.get("reward_components"):
        return {}
    keys: set[str] = set()
    for d in trace["reward_components"]:
        if isinstance(d, dict):
            keys |= set(d.keys())
    acc: Dict[str, float] = {k: 0.0 for k in keys}
    for d in trace["reward_components"]:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + float(v)
    return acc


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
