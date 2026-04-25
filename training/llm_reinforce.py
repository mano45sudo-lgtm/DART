from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from env.action_parser import safe_action
from env.digital_twin_env import DigitalTwinDiabetesEnv


def obs_to_prompt(obs: Dict[str, Any]) -> str:
    return (
        "You are a diabetes treatment policy.\n"
        "Return exactly ONE minified JSON object and nothing else (no prose, no markdown).\n"
        'Schema: {"type":"noop|start|add|stop|switch|dose_adjust",'
        '"drug":"metformin|glp1|sglt2|dpp4|sulfonylurea|insulin",'
        '"dose":0.0-1.0,"lifestyle":0.0-1.0,"from_drug":"...","to_drug":"..."}\n'
        "Rules: switch requires from_drug+to_drug+dose; noop only needs type.\n"
        f"week: {obs['week']}\n"
        f"hba1c: {float(obs['hba1c']):.2f}\n"
        f"fasting_glucose: {float(obs['fasting_glucose']):.0f}\n"
        f"bmi: {float(obs['bmi']):.1f}\n"
        f"egfr: {float(obs['egfr']):.0f}\n"
        f"ckd: {obs['ckd']} cvd: {obs['cvd']}\n"
        'Examples: {"type":"start","drug":"metformin","dose":1.0,"lifestyle":0.7} '
        '{"type":"add","drug":"glp1","dose":0.8} {"type":"noop"}\n'
        "Output JSON now:"
    )


@dataclass
class StepRecord:
    logp_sum: torch.Tensor
    reward: float
    parse_ok: float


def response_logprob_sum(model: torch.nn.Module, full_ids: torch.Tensor, query_len: int) -> torch.Tensor:
    """Sum of log p(token | context) over generated response tokens (fixed sequence)."""
    L = int(full_ids.shape[1])
    R = L - int(query_len)
    if R <= 0:
        return full_ids.new_zeros(())
    mask = torch.ones((1, L), dtype=torch.long, device=full_ids.device)
    out = model(input_ids=full_ids, attention_mask=mask)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    lp = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    start = int(query_len) - 1
    return lp[:, start : start + R].sum()


@torch.no_grad()
def sample_action_via_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, int, str, Dict[str, Any], Dict[str, Any]]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    q_len = int(input_ids.shape[1])
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    gen = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 1e-4),
        top_p=float(top_p),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    resp_ids = gen[0, q_len:]
    text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    action, pinfo = safe_action(text)
    return gen, q_len, text, action, pinfo


def rollout_episode(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    env_seed: int,
    max_steps: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[StepRecord], float]:
    env = DigitalTwinDiabetesEnv(seed=env_seed, max_steps=max_steps)
    obs, _ = env.reset(seed=env_seed)
    records: List[StepRecord] = []
    total_return = 0.0
    done = False
    while not done:
        prompt = obs_to_prompt(obs)
        with torch.no_grad():
            full_ids, q_len, _txt, action, pinfo = sample_action_via_generate(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        logp_sum = response_logprob_sum(model, full_ids, q_len)
        obs, r, term, trunc, _info = env.step(action)
        total_return += float(r)
        ok = 1.0 if pinfo.get("ok") else 0.0
        records.append(StepRecord(logp_sum=logp_sum, reward=float(r), parse_ok=ok))
        done = bool(term or trunc)
    return records, total_return


def reinforce_loss_on_episode(steps: List[StepRecord], *, gamma: float = 1.0) -> Optional[torch.Tensor]:
    if not steps:
        return None
    rewards = [s.reward for s in steps]
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
        returns.append(g)
    returns.reverse()
    m = sum(returns) / len(returns)
    adv = [returns[i] - m for i in range(len(steps))]
    terms = [-steps[i].logp_sum * adv[i] for i in range(len(steps))]
    return torch.stack(terms).mean()


@torch.no_grad()
def eval_mean_return(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    seeds: List[int],
    max_steps: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, float, float]:
    returns: List[float] = []
    parse_rates: List[float] = []
    model.eval()
    for sd in seeds:
        env = DigitalTwinDiabetesEnv(seed=sd, max_steps=max_steps)
        obs, _ = env.reset(seed=sd)
        g = 0.0
        oks: List[float] = []
        done = False
        while not done:
            prompt = obs_to_prompt(obs)
            full_ids, _q, _t, action, pinfo = sample_action_via_generate(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            obs, r, term, trunc, _ = env.step(action)
            g += float(r)
            oks.append(1.0 if pinfo.get("ok") else 0.0)
            done = bool(term or trunc)
        returns.append(g)
        parse_rates.append(sum(oks) / max(len(oks), 1))
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(parse_rates))
