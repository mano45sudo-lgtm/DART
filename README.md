---
title: DART — Digital Twin Medicine
emoji: 🧬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# 🧬 DART — Digital Twin Medicine

<div align="center">

   

**An RL agent that learns personalized T2DM treatment policies inside a stochastic digital twin — prescribing to *individuals*, not populations.**

</div>

---

## What DART does??

DART trains a **reinforcement learning agent** to manage T2DM treatment decisions for a single simulated patient, one week at a time.

Every week, the agent receives a partial clinical observation — current HbA1c, fasting plasma glucose, BMI, kidney function, blood pressure, cardiovascular history. It then issues a treatment action: start a drug, stop one, adjust a dose, add to the regimen, or do nothing. The environment responds: the patient's physiology evolves, new labs come in, costs accumulate, and risk compounds or recedes.

This loop runs for up to **52 simulated weeks** per episode. At the end, the agent is scored not just on glucose control, but on hypoglycaemia avoided, costs kept reasonable, and physiological stability maintained.

Then `reset()` is called. A new patient is drawn from the simulator. The agent does it again.

Over thousands of episodes, the agent learns a policy that generalizes — not to a single patient's quirks, but to the *structure* of personalized decision-making. When do you escalate? When do you hold? When does adding a second drug make sense, and when does it introduce interaction risk that outweighs the glycaemic benefit?

**This is what population guidelines cannot capture. DART learns it from experience.**

| | |
| --- | --- |
| **Submission Colab** | [`training/DART_Colab_submission.ipynb`](https://colab.research.google.com/github/mano45sudo-lgtm/DART/blob/main/training/DART_Colab_submission.ipynb) |
| **Streamlit Space** | [huggingface.co/spaces/mano678/DART_1](https://huggingface.co/spaces/mano678/DART_1) |
| **Figures on Hub** | [`docs/figures` in Space](https://huggingface.co/spaces/mano678/DART_1/tree/main/docs/figures) (sync: `python scripts/upload_figures_to_hf_space.py`) |

---

## The environment: a stochastic digital twin

The core of DART is `DigitalTwinDiabetesEnv` — a Gym-style environment that simulates a T2DM patient's physiology week by week.

```
reset(seed=42)  →  draws a new virtual patient (age, BMI, baseline labs, comorbidities)
step(action)    →  applies treatment, evolves physiology, returns (obs, reward, done, info)

```

**Observations are partial by design.** The agent cannot see everything — just what a clinician would see at a weekly check-in: `week`, `hba1c`, `fasting_glucose`, `bmi`, `systolic_bp`, `egfr`, `ckd_flag`, `cvd_flag`. Latent physiological state, drug pharmacokinetics, patient adherence — these are hidden. The agent must reason under uncertainty, exactly as a real clinician does.

**Actions are structured JSON**, validated against the patient's current state before execution:


| Action type   | What it does                          |
| ------------- | ------------------------------------- |
| `noop`        | Hold current regimen                  |
| `start`       | Initiate a new drug                   |
| `add`         | Add to existing regimen               |
| `stop`        | Discontinue a drug                    |
| `switch`      | Replace one drug with another         |
| `dose_adjust` | Increase or decrease an existing dose |


A safety layer (`fall_detection.py`) intercepts actions that would create dangerous drug interactions, contraindicated combinations, or doses outside safe ranges. The agent learns, over time, not to propose them.

---

## The reward: what we actually care about

Most RL environments have a reward that is easy to describe and hard to interpret. DART inverts this. The reward is a **named, decomposed sum** of nine clinical terms. Every term has a meaning you can explain to a doctor.


| Component              | What it measures                                          | Why it matters                                                   |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| `glucose_improvement`  | FPG and HbA1c moving toward target                        | The primary goal of T2DM treatment                               |
| `hypoglycemia_penalty` | Low glucose events and hypo-type side effects             | Hypoglycaemia is acutely dangerous and common from overtreatment |
| `instability`          | Glucose swing magnitude week-over-week                    | Volatile control is as harmful as chronic elevation              |
| `treatment_cost`       | Weekly USD cost of the regimen                            | Cost is a real constraint on real patients                       |
| `inactivity`           | Stalling on `noop` or repeating actions                   | The agent must engage, not coast                                 |
| `time_decay`           | Mild pressure across the episode                          | Deferred action has compounding costs                            |
| `cumulative_hyper`     | Prolonged exposure to elevated glucose                    | Slow damage accumulates when HbA1c stays high                    |
| `tool_intervention`    | Safety overrides triggered by the safety layer            | Penalises actions that required external correction              |
| `terminal`             | Large bonus for remission, large penalty for hard failure | Episode-end outcomes matter                                      |


This structure serves a purpose beyond interpretability. When you look at the judge figures and see that `distilgpt2` improved on `random` mostly through `glucose_improvement` and `hypoglycemia_penalty` but showed minimal change in `treatment_cost` — that is a finding. That tells you what the model learned and what it did not. A black-box scalar reward cannot give you that.

---

## The clinical toolbox

The agent's world model is enriched by eight clinical modules that feed into both the environment and the evaluation stack:


| Module                          | Role                                               |
| ------------------------------- | -------------------------------------------------- |
| `tools/ehr.py`                  | Structured patient history and current medications |
| `tools/genomics.py`             | Variant-level signals that affect drug metabolism  |
| `tools/interactions.py`         | Drug–drug interaction checker                      |
| `tools/progression_forecast.py` | Short-horizon trajectory projection                |
| `tools/trial_sim.py`            | In-silico trialing of candidate next actions       |
| `tools/biomarkers.py`           | Biomarker-level signal interpretation              |
| `tools/resistance.py`           | Heuristics for drug resistance patterns            |
| `tools/risk.py`                 | Cardiovascular and renal risk scoring              |


These are not decorative. They shape the observation space, constrain the action validator, and are callable during rollouts. The agent operates inside a system that knows what a cardiologist would flag.

---

## Training: REINFORCE on live rollouts

DART uses on-policy REINFORCE. There is no replay buffer, no offline dataset. Every gradient update comes from fresh episodes rolled out in the digital twin.

```
for each update:
    sample K episodes in DigitalTwinDiabetesEnv
    compute REINFORCE loss on the language model's generated actions
    optimizer step

```

The language model generates the JSON action each week. The environment scores it. The loss propagates back through the model. Over updates, the model learns which action sequences produce high-reward trajectories.

**Three training modes:**

```bash
# Fast wiring check (CPU, tiny-gpt2, ~2 minutes)
python scripts/train_reinforce_twin.py --quick

# Judge preset (distilgpt2, full schedule)
python scripts/train_reinforce_twin.py --judge-preset

# Full run (4-bit 8B LLM, requires GPU)
python scripts/train_reinforce_twin.py \
  --judge-schedule --model meta-llama/Meta-Llama-3-8B \
  --load-in-4bit --out-json logs/my_run.json

```

**The council layer** (`council.py`, `training/council_rollout.py`) fuses multiple rule-style agents with the LM policy, handles exploration schedules, and marks self-repair episodes — cases where the agent corrected itself after a safety intervention. These are visible in the `self_repair_episodes.png` plot.

---

## The submission Colab: one notebook, full evidence

`training/DART_Colab_submission.ipynb` is the single source of truth for the submission. Run it top to bottom and it produces:

1. **Training curves** — smoothed episode return for random baseline, `distilgpt2`, optional 8B, and council
2. **Final comparison bars** — tail mean ± std per model
3. **Clinical traces** — full week-by-week vitals, actions, costs, and rubric components for matched episodes (same patient seed, different policies)
4. **Judge outcome boxplots** — final HbA1c, FPG, and episode return across N independent seeds
5. **Action mix plots** — what types of interventions each policy preferred
6. **Rubric breakdown** — which reward components drove the improvement
7. A single JSON: `logs/colab_experiment.json` containing all of the above in structured form

Everything a judge needs to verify, inspect, and reproduce is in that one file and those seven figures.

---

## Results

Figures below match the **Colab** pipeline (`plot_colab_publication.py` + `plot_colab_judge_insights.py`). They live in **`docs/figures/`** on GitHub `main`; URLs use **`raw.githubusercontent.com`** so they render here and on Hugging Face. After your own Colab run, copy `logs/colab_experiment.json`, re-run the plotters with `--also-svg`, commit, `git push origin main`, then optionally `python scripts/upload_figures_to_hf_space.py` to refresh the [Space `docs/figures`](https://huggingface.co/spaces/mano678/DART_1/tree/main/docs/figures).

### Training curves and tail comparison

![Smoothed training curves — random, distilgpt2, optional 8B, council](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/training_curve.png)

*Smoothed per-episode return across policies. The trained agent improves on the random baseline; the council variant shows the effect of structured exploration and self-repair.*

![Tail mean ± std by model](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/final_comparison_bars.png)

*Mean return over the last *K* episodes per model with uncertainty bars.*

![Example fasting glucose trajectories (one episode each)](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/behavior_glucose.png)

*One random vs one trained small-LM trajectory (same horizon as Colab `max_steps`).*

### Council + self-repair

![Council episode return with repair markers](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/self_repair_episodes.png)

*Council return over episodes; dashed vertical lines mark self-repair / exploration signals from `colab_experiment.json`.*

### Judge dashboard — same-patient traces (matched seed)

![FPG, HbA1c, eGFR, weekly cost — example episodes per policy](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_clinical_state.png)

*Clinical dynamics on the **same virtual patient** (shared `judge_trace_env_seed`): FPG, HbA1c, eGFR, and weekly cost across simulated weeks.*

![Per-step and cumulative return from the rubric](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_step_and_cumulative_return.png)

*Dense weekly reward and cumulative return for the traced episodes.*

![Action type counts (parsed JSON `type`)](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_action_mix.png)

*Which intervention types each policy used in the logged example episode.*

### Stochasticity — *N* independent eval seeds

![Final HbA1c, FPG, and episode return — boxplots](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_outcome_distributions.png)

*Boxplots over *N* evaluation seeds. The trained policy consistently achieves lower final HbA1c and higher episode return than the random baseline — and the variance tells us something about which patients it struggles with.*

### Rubric breakdown (what actually moved)

![Summed reward components over the traced episode](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_rubric_episode_totals.png)

*Stacked reward components per policy. This is where you see what the model actually learned: the trained agent's gains come primarily from glucose improvement and hypoglycaemia avoidance, not from cost or stability terms — a specific, actionable finding for the next training run.*

### Council — example glucose path (non-LLM baseline)

![Council example FPG path](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_council_glucose_example.png)

*One council rollout on the glucose axis (rule fusion; complements LM traces above).*

---

## Reproduce it yourself

```bash
git clone https://github.com/mano45sudo-lgtm/DART
cd DART
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_hackathon.txt

# Sanity check
python scripts/run_sanity.py

# Full evaluation
python scripts/run_evaluation.py

# Regenerate README figures without retraining
python scripts/generate_readme_demo_figures.py

```

The Streamlit UI runs with `streamlit run app.py`. The OpenEnv FastAPI server is in `dtm_openenv/` — health check at `/health`, API docs at `/docs`.

---

## Repository layout

```
env/                  DigitalTwinDiabetesEnv, PatientTwin, action validator, fall detection
tools/                EHR, genomics, interactions, risk, trialing, biomarkers
reward/               RewardRubric — decomposed reward
training/             DART_Colab_submission.ipynb, REINFORCE helpers, council rollout
council.py            Multi-agent fusion layer
self_improvement.py   Exploration and self-repair controller
evaluation/           Baselines and metrics
scripts/              Training CLI, plotting, sanity, evaluation runners
dtm_openenv/          FastAPI OpenEnv server
ui/                   Streamlit app
docs/figures/         Training figures (generated by Colab or scripts)
logs/                 colab_experiment.json and other run outputs

```

---

## Submission checklist

- [x] Gym-style environment with `reset`, `step`, `state` — `env/digital_twin_env.py`
- [x] OpenEnv HTTP server — `dtm_openenv/` (FastAPI, Docker-ready)
- [x] Training on live rollouts only — `train_reinforce_twin.py`, `colab_episode_rl.py`
- [x] Single Colab with training + figures + JSON evidence — `training/DART_Colab_submission.ipynb`
- [x] Quantitative comparison (bars, boxplots) + qualitative traces (clinical state, rubric breakdown)
- [x] Decomposed, interpretable reward — every term named and logged

---

## What comes next

The hardest problems are still ahead. A 20-week horizon is a proof of concept; real T2DM management runs for decades. The patient simulator is richer than most, but it is still a simulator. The gap between in-silico performance and clinical utility is the gap this project is ultimately trying to close.

The next steps are longer horizons with curriculum-structured patient difficulty, tighter comparisons between small and large models under identical compute budgets, and calibrated uncertainty estimates so the agent can flag when it is operating outside its training distribution.

But the core claim — that a sequential RL policy trained on a digital twin can learn something meaningful about individualized treatment — is what this submission exists to test. The figures say it can.

---

<div align="center">

*Built for OpenEnv Hackathon 2026 · Track #3.1 — World Modeling (Professional Tasks)*

*PRs and forks welcome.*

</div>