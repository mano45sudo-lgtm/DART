---
title: "DART — Digital Twin Medicine"
emoji: 🧬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.56.0"
app_file: app.py
pinned: false
tags:
  - openenv
  - World modeling
  - digital-twin
  - healthcare
  - personalized-medicine
short_description: "Personalized T2DM RL agent in a stochastic digital twin"
---

# 🧬 DART — Digital Twin Medicine

> *An RL agent that learns personalized T2DM treatment policies inside a stochastic digital twin — prescribing to* ***individuals***, *not populations.*

<div align="center">

<a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>   <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank"><img src="https://img.shields.io/badge/OpenEnv-compliant-green" alt="OpenEnv"/></a>   <a href="https://huggingface.co/spaces/mano678/DART_1" target="_blank"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow" alt="HuggingFace Space"/></a>   <a href="https://colab.research.google.com/github/mano45sudo-lgtm/DART/blob/main/training/DART_Colab_submission.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</div>

---

## 🔗 Submission Links

> All links open in a new tab — you won't lose your place in this README.


| Resource                                  | Link                                                                                                                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🤗 **UI — Streamlit Space**               | <a href="https://huggingface.co/spaces/mano678/DART_1" target="_blank">huggingface.co/spaces/mano678/DART_1</a>                                                        |
| 🌐 **OpenEnv Environment Server**         | <a href="https://mano678-dart-1.hf.space" target="_blank">mano678-dart-1.hf.space</a>                                                                                  |
| 📓 **Submission Colab (single notebook)** | <a href="https://colab.research.google.com/github/mano45sudo-lgtm/DART/blob/main/training/DART_Colab_submission.ipynb" target="_blank">DART_Colab_submission.ipynb</a> |
| 💻 **GitHub Repository**                  | <a href="https://github.com/mano45sudo-lgtm/DART" target="_blank">github.com/mano45sudo-lgtm/DART</a>                                                                  |
| 📝 **Mini-blog / Demo video**             | <a href="https://ADD_BLOG_OR_VIDEO_URL_HERE" target="_blank">ADD_BLOG_OR_VIDEO_URL_HERE</a>                                                                            |


---

## 📌 Table of Contents

1. [What is this?](#what-is-this)
2. [The Problem with Medicine Today](#the-problem-with-medicine-today)
3. [How It Works](#how-it-works)
4. [The Environment — Digital Twin](#the-environment--digital-twin)
5. [Action Space](#action-space)
6. [Observation Space](#observation-space)
7. [Reward Function](#reward-function)
8. [Clinical Tools in the Stack](#clinical-tools-in-the-stack)
9. [Training — REINFORCE on Live Rollouts](#training--reinforce-on-live-rollouts)
10. [The Council Layer](#the-council-layer)
11. [Submission Colab — Full Evidence](#submission-colab--full-evidence)
12. [Results & Figures](#results--figures)
13. [Baseline Scores](#baseline-scores)
14. [Installation Guide](#installation-guide)
15. [Running the Environment](#running-the-environment)
16. [API Endpoints](#api-endpoints)
17. [Project Structure](#project-structure)
18. [License & Citation](#license--citation)

---

## What is this?

Every hospital, clinic, and GP surgery managing Type 2 diabetes (T2DM) follows the same escalation ladder. First metformin. If that fails, add a second drug. If that fails, escalate further. The ladder is built for populations — it is the best average policy across millions of patients studied in clinical trials.

But you are never treating an average. You are treating a person whose response depends on their kidney function, their cardiovascular history, their weight trajectory, their genetics, their existing drug interactions, and dozens of signals a static guideline cannot absorb.

**DART trains a reinforcement learning agent to do what guidelines cannot: make sequential, personalized treatment decisions, week by week, as new clinical data arrives.**

The agent operates inside a stochastic digital twin of a T2DM patient. It receives partial observations — the same labs a clinician would see at a weekly check-in. It chooses a treatment action. The twin responds. The agent learns. Over thousands of simulated patients and episodes, it builds a policy that adapts to the individual, not the average.

**This is not a recommendation system. This is a sequential decision-making agent trained on experience.**

---

## The Problem with Medicine Today

Two patients. Both T2DM. Both HbA1c 8.4%. Same prescription, same dose, same follow-up schedule.

Three months later — one is controlled, one is not. The guidelines say: escalate. Add a drug. Same drug for both, again.

This is not a failure of care. **This is the care.** And it costs enormously:

- **Hypoglycaemia** sends hundreds of thousands of patients to emergency rooms every year — a direct consequence of overshooting with one-size-fits-all dosing.
- Patients **cycle through drugs** that were never going to work for their physiology, accumulating side effects and losing months of glycaemic control.
- **Chronic hyperglycaemia** that persists through guideline-escalation delays silently destroys kidneys and hearts.

The root cause: medicine has no mechanism to learn from individual trajectories and adapt in real time. It has guidelines. Guidelines are not the same thing.

**DART is an attempt to build that mechanism.**

---

## How It Works

```
reset(seed=N)  →  draws a new virtual patient from the stochastic simulator
      ↓
Agent receives partial clinical observation
(HbA1c, FPG, eGFR, BMI, BP, CKD flag, CVD flag, week index)
      ↓
Agent issues a structured JSON action
(start drug / adjust dose / stop / switch / add / noop)
      ↓
Safety layer validates action
(drug interactions, contraindications, dose limits)
      ↓
Patient twin evolves — physiology updates, labs change, costs accrue
      ↓
Reward computed across 9 named clinical terms
      ↓
Repeat for up to 52 simulated weeks
      ↓
Terminal reward: remission bonus or hard-failure penalty

```

Training runs **entirely on live rollouts** — no static dataset, no replay buffer. Every gradient update comes from fresh episodes sampled in the digital twin. The agent learns what works through experience, exactly as a clinician does — except it runs thousands of patients simultaneously.

---

## The Environment — Digital Twin

`DigitalTwinDiabetesEnv` (`env/digital_twin_env.py`) is a Gym-style environment that simulates a T2DM patient's physiology week by week.


| Property          | Value                                                          |
| ----------------- | -------------------------------------------------------------- |
| **Horizon**       | Up to **52** simulated weeks (default training: **20**)        |
| **Observation**   | Partial — 8 clinical signals visible per step                  |
| **Action**        | Structured JSON object, one per simulated week                 |
| **Stochasticity** | `reset(seed=N)` draws a new virtual patient each episode       |
| **Safety layer**  | `fall_detection.py` intercepts unsafe actions before execution |


**Observations are partial by design.** The agent cannot see latent physiological state, drug pharmacokinetics, or patient adherence. It sees only what a clinician sees at a weekly review. Reasoning under this uncertainty is the core challenge — and the point.

---

## Action Space

At each simulated week, the agent emits a single structured JSON action, validated against the patient's current clinical state before execution.


| Action        | What it does                          | When to use                                     |
| ------------- | ------------------------------------- | ----------------------------------------------- |
| `noop`        | Hold current regimen unchanged        | Monitoring period; no escalation warranted      |
| `start`       | Initiate a new drug from scratch      | First-line treatment or new drug class          |
| `add`         | Add a drug to an existing regimen     | Combination therapy                             |
| `stop`        | Discontinue a drug                    | Side effects, contraindication, ineffectiveness |
| `switch`      | Replace one drug with another         | Intolerance or treatment failure                |
| `dose_adjust` | Increase or decrease an existing dose | Titration toward target                         |


Invalid actions — drug interactions, out-of-range doses, contraindicated combinations — are caught by the safety validator. The agent receives a `tool_intervention` penalty for triggering the safety layer, creating a training signal to avoid unsafe prescribing.

---

## Observation Space

At each step, the agent receives the following partial clinical observation:


| Field             | Type  | Description                          |
| ----------------- | ----- | ------------------------------------ |
| `week`            | int   | Current simulated week (0–52)        |
| `hba1c`           | float | Glycated haemoglobin (%)             |
| `fasting_glucose` | float | Fasting plasma glucose (mmol/L)      |
| `bmi`             | float | Body mass index                      |
| `systolic_bp`     | float | Systolic blood pressure (mmHg)       |
| `egfr`            | float | Estimated glomerular filtration rate |
| `ckd`             | bool  | Chronic kidney disease flag          |
| `cvd`             | bool  | Cardiovascular disease flag          |


**Hidden from the agent (latent state):** drug pharmacokinetics, patient adherence rate, comorbidity severity progression, genetic variant signals. The agent must act on partial information — as every clinician does.

---

## Reward Function

The step reward is a **named, decomposed sum of nine clinical terms** — not a black-box scalar. Every term has a clinical meaning, appears in training logs, and is plotted individually in the judge figures. This transparency is intentional: when the agent improves, you can see *exactly what it learned*.


| Component              | What it measures                                          | Why it matters                                     |
| ---------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| `glucose_improvement`  | FPG and HbA1c movement toward clinical targets            | The primary goal of T2DM management                |
| `hypoglycemia_penalty` | Low glucose events and hypo-type adverse effects          | Overtreatment is as dangerous as undertreatment    |
| `instability`          | Week-over-week glucose swing magnitude                    | Volatile control compounds long-term organ damage  |
| `treatment_cost`       | Weekly USD cost of the prescribed regimen                 | Cost is a real constraint for real patients        |
| `inactivity`           | Stalling on `noop` or repeating identical actions         | Passive management is penalised                    |
| `time_decay`           | Mild penalty compounding over the episode                 | Deferred action has real clinical cost             |
| `cumulative_hyper`     | Prolonged exposure to elevated glucose                    | Sustained hyperglycaemia causes progressive damage |
| `tool_intervention`    | Safety layer overrides triggered by unsafe actions        | Penalises actions requiring external correction    |
| `terminal`             | Large bonus for remission; large penalty for hard failure | Episode-end outcomes matter most                   |


**Episode return** = sum of weekly step rewards. Learning curves, comparison bars, and boxplots in the submission Colab all reflect this total.

---

## Clinical Tools in the Stack

Eight specialised modules feed signal into the environment's observation construction, action validation, and evaluation stack:


| Module                          | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `tools/ehr.py`                  | Structured patient history and current medication list |
| `tools/genomics.py`             | Variant-level signals affecting drug metabolism        |
| `tools/interactions.py`         | Drug–drug interaction checker                          |
| `tools/progression_forecast.py` | Short-horizon physiological trajectory projection      |
| `tools/trial_sim.py`            | In-silico trialing of candidate next actions           |
| `tools/biomarkers.py`           | Biomarker-level signal interpretation                  |
| `tools/resistance.py`           | Heuristics for emerging drug resistance patterns       |
| `tools/risk.py`                 | Cardiovascular and renal risk scoring                  |


These are not decorative modules. They shape the observation space, constrain the action validator, and are callable during rollouts. The agent operates inside a system that reflects what a specialist multidisciplinary team would flag.

---

## Training — REINFORCE on Live Rollouts

DART uses on-policy REINFORCE. There is no replay buffer, no offline dataset, no pre-collected demonstrations. Every gradient update comes from fresh episodes rolled out in the digital twin.

```
for each update:
    sample K episodes in DigitalTwinDiabetesEnv
    language model generates JSON action at each week
    compute REINFORCE loss on generated action sequences
    optimizer step

```

The language model generates the treatment action at each week. The environment scores it. The loss propagates back. Over updates, the model learns which action sequences produce high-reward trajectories — trading off glucose improvement, cost, safety, and stability simultaneously.

**Three training modes:**

```bash
# Smoke test — CPU only, tiny-gpt2, ~2 minutes
python scripts/train_reinforce_twin.py --quick

# Judge preset — distilgpt2, full update schedule
python scripts/train_reinforce_twin.py --judge-preset

# Full run — 4-bit quantised 8B LLM, requires GPU
python scripts/train_reinforce_twin.py \
  --judge-schedule \
  --model meta-llama/Meta-Llama-3-8B \
  --load-in-4bit \
  --out-json logs/my_run.json

```

---

## The Council Layer

Beyond the base REINFORCE loop, DART includes a **Council** (`council.py`, `training/council_rollout.py`) — a non-LLM fusion layer that runs alongside the language model policy.

Three rule-style agents with distinct clinical dispositions vote on each action independently:


| Agent               | Disposition             | Preferred actions                                  |
| ------------------- | ----------------------- | -------------------------------------------------- |
| Clinical Heuristics | Conservative escalation | Follow guideline ladder                            |
| Cost Optimiser      | Budget-aware            | Prefer cheaper, established drugs                  |
| Safety Monitor      | Risk-averse             | Flag interactions, prefer `noop` under uncertainty |


Their weighted votes are aggregated and logged as a separate policy track. The council does not control the LM policy — it acts as an exploration controller and safety net, flagging episodes where the LM diverged sharply from clinical heuristics.

**Self-repair episodes** (`self_improvement.py`) are cases where the agent corrected its own trajectory after a safety intervention or council divergence. These are marked in the `self_repair_episodes.png` figure with dashed vertical lines.

---

## Submission Colab — Full Evidence

`training/DART_Colab_submission.ipynb` is the single source of truth for this submission. Run it top to bottom on any GPU runtime to produce all training evidence, figures, and a structured JSON output.


| Output                       | Content                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| Training curves              | Smoothed episode return for `random`, `distilgpt2`, optional 8B, council |
| Final comparison bars        | Tail mean ± std for the last *K* episodes per model                      |
| Clinical traces              | Full week-by-week vitals, actions, costs, rubric components              |
| Judge outcome boxplots       | Final HbA1c, FPG, episode return across N independent seeds              |
| Action mix plots             | Action type distribution per policy                                      |
| Rubric breakdown             | Per-component reward contribution per policy                             |
| `logs/colab_experiment.json` | All of the above in structured, inspectable JSON                         |


**Default** `CONFIG` **block (edit once at the top of the notebook):**


| Key                       | Default                      | Meaning                                |
| ------------------------- | ---------------------------- | -------------------------------------- |
| `max_steps`               | `20`                         | Simulated weeks per episode            |
| `random_episodes`         | `40`                         | Random baseline log length             |
| `small.updates`           | `24`                         | distilgpt2 REINFORCE update count      |
| `episodes_per_update`     | `2`                          | Episodes per gradient step             |
| `council_episodes`        | `16`                         | Council + self-repair episode count    |
| `judge_endpoint_episodes` | `20`                         | Independent eval rollouts for boxplots |
| `judge_trace_env_seed`    | `50200`                      | Same seed for random vs distil trace   |
| `bar_tail_episodes`       | `15`                         | Last *K* episodes for tail mean        |
| `out_json`                | `logs/colab_experiment.json` | Output path for all episode data       |


**Regenerate figures from an existing JSON (no retraining needed):**

```bash
cd DART
python scripts/plot_colab_publication.py \
  --in-json logs/colab_experiment.json \
  --out-dir docs/figures --also-svg

python scripts/plot_colab_judge_insights.py \
  --in-json logs/colab_experiment.json \
  --out-dir docs/figures --also-svg

```

**Quick demo — regenerate committed README figures without GPU:**

```bash
python scripts/generate_readme_demo_figures.py

```

---

## Results & Figures

### Training curve



Smoothed per-episode return across `random`, `distilgpt2`, optional `llama-8b-4bit`, and council. The trained agent improves consistently over the random baseline across the update schedule.

![Training curve — smoothed episode return by policy](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/training_curve.png)

---

### Behavior — glucose trajectories



Random vs trained fasting glucose trajectories over simulated weeks (same logs as the publication Colab / `scripts/generate_readme_demo_figures.py`).

![Behavior — glucose trajectories](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/behavior_glucose.png)

---

### Final comparison — tail mean ± std



Mean return over the last `bar_tail_episodes` episodes per model with standard deviation error bars. Values are computed directly from `colab_experiment.json`.

![Final comparison — tail mean ± std across policies](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/final_comparison_bars.png)

---

### Clinical state — same virtual patient, different policies



FPG, HbA1c, eGFR, and weekly cost across simulated weeks for matched random vs trained traces. The `judge_trace_env_seed` ensures an identical virtual patient across policies — isolating the effect of the policy itself.

![Clinical state — matched random vs trained traces](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_clinical_state.png)

---

### Reward rubric — what actually changed



Stacked component sums per policy. The trained agent's gains are concentrated in `glucose_improvement` and `hypoglycemia_penalty`. Cost and instability terms show less movement — a concrete, specific finding for the next training iteration.

![Reward rubric — stacked component totals by policy](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_rubric_episode_totals.png)

---

### Outcome distributions — N seeds, many virtual patients



Boxplots of final HbA1c, final FPG, and episode return across `judge_endpoint_episodes` independent seeds per model. The trained policy must generalise across patient variation — not just perform on one seed.

![Outcome distributions — final labs and return across seeds](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_outcome_distributions.png)

---

### Action mix



Action type counts per policy during the traced episode. The random policy distributes uniformly. The trained policy develops preferences — a qualitative signal of learned clinical strategy.

![Action mix — action type counts per policy](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_action_mix.png)

---

### Self-repair episodes



Council episode return with dashed vertical lines at `self_repair_episodes` — moments where the agent corrected its trajectory following a safety intervention or exploration signal.

![Self-repair episodes — council return with repair markers](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/self_repair_episodes.png)

---

### Judge — step reward and cumulative return



Per-step reward and cumulative return for the traced episode used in the judge panels.

![Judge — step reward and cumulative return](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_step_and_cumulative_return.png)

---

### Council — glucose example trace



Glucose dynamics for the council self-repair trace window (example episode).

![Council — glucose example trace](https://raw.githubusercontent.com/mano45sudo-lgtm/DART/main/docs/figures/judge_council_glucose_example.png)

---

## Baseline Scores

Baseline policy: `distilgpt2` trained for `small.updates=24` steps at `episodes_per_update=2` episodes per update.


| Metric             | Random baseline     | distilgpt2 trained  |
| ------------------ | ------------------- | ------------------- |
| Tail mean return   | *compute from JSON* | *compute from JSON* |
| Tail std return    | *compute from JSON* | *compute from JSON* |
| Final HbA1c median | *from boxplot*      | *from boxplot*      |
| Final FPG median   | *from boxplot*      | *from boxplot*      |
| Judge N (seeds)    | 20                  | 20                  |


**Compute directly from** `colab_experiment.json`**:**

```python
import json, statistics
from collections import defaultdict

d = json.load(open("logs/colab_experiment.json"))
K = d.get("config", {}).get("bar_tail_episodes", 15)
by = defaultdict(list)
for r in d.get("episodes", []):
    by[r["model"]].append(r["reward"])
for m, xs in sorted(by.items()):
    tail = [float(x) for x in xs[-K:]]
    print(m,
          "tail_mean =", round(statistics.mean(tail), 3),
          "std =", round(statistics.stdev(tail) if len(tail) > 1 else 0.0, 3))
print("judge N:", {k: len(v) for k, v in d.get("endpoints", {}).items()})

```

---

## Installation Guide

### Prerequisites

**1. Check Python version (must be 3.10 or higher):**

```bash
python3 --version

```

**2. Check Git is installed:**

```bash
git --version

```

**3. For Docker-based OpenEnv server (optional):**

Download <a href="https://docs.docker.com/get-docker/" target="_blank">Docker Desktop</a> and ensure the engine is running before using any Docker commands.

**4. HuggingFace account (for gated LLMs):**

Sign up at <a href="https://huggingface.co" target="_blank">huggingface.co</a>. Required only when running the 8B model. For smoke tests and distilgpt2, no account is needed.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/mano45sudo-lgtm/DART.git
cd DART

```

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip

```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_hackathon.txt

```

---

### Step 4 — (Optional) Authenticate with HuggingFace

Required only for gated LLMs such as Llama-3-8B:

```bash
huggingface-cli login
# paste your HF read token when prompted

```

> ⚠️ **Security:** Never commit tokens to GitHub. The `.gitignore` in this repo already blocks `.env` and common token files.

---

## Running the Environment

### Option A — Sanity check (fastest, no GPU)

Verifies environment wiring without running full training:

```bash
python scripts/run_sanity.py

```

Expected output:

```
[OK] DigitalTwinDiabetesEnv: reset, step, state verified
[OK] RewardRubric: 9 components present
[OK] Safety layer: fall_detection responding
[OK] All sanity checks passed

```

---

### Option B — Full evaluation across N seeds

```bash
python scripts/run_evaluation.py

```

Runs the random baseline and trained policy across N seeds and writes results to `logs/`.

---

### Option C — Training CLI

```bash
# Smoke test — CPU, tiny-gpt2, ~2 minutes
python scripts/train_reinforce_twin.py --quick

# Judge preset — distilgpt2, full update schedule
python scripts/train_reinforce_twin.py --judge-preset

# Custom — 4-bit 8B on GPU
python scripts/train_reinforce_twin.py \
  --judge-schedule \
  --model meta-llama/Meta-Llama-3-8B \
  --load-in-4bit \
  --out-json logs/my_run.json

```

---

### Option D — Streamlit UI

```bash
streamlit run app.py

```

Opens an interactive dashboard showing the digital twin stepping in real time with live rubric component plots and clinical trace visualisation.

---

### Option E — OpenEnv HTTP server (Docker)

**Step 1 — Build the image:**

```bash
docker build -t dart-openenv .

```

**Step 2 — Run the container:**

```bash
docker run -p 7860:7860 dart-openenv

```

**Step 3 — Verify the server is running:**

```bash
curl http://localhost:7860/health

```

Expected response:

```json
{"status": "ok", "env": "DigitalTwinDiabetesEnv"}

```

**Step 4 — Reset the environment with a patient seed:**

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'

```

**Step 5 — Take a treatment step:**

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "start", "drug": "metformin", "dose": 500}}'

```

**Step 6 — Stop the container:**

```bash
docker ps                        # find the container ID
docker stop <container_id>

```

---

### Option F — Live HuggingFace Space (no installation)

```bash
# Health check
curl https://mano678-dart-1.hf.space/health

# Reset with patient seed
curl -X POST https://mano678-dart-1.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'

# Take a treatment action
curl -X POST https://mano678-dart-1.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "dose_adjust", "drug": "metformin", "delta": 250}}'

```

---

## API Endpoints


| Method | Endpoint  | Body                | Description                                   |
| ------ | --------- | ------------------- | --------------------------------------------- |
| GET    | `/health` | —                   | Health check and environment metadata         |
| POST   | `/reset`  | `{"seed": int}`     | Start a new episode with a given patient seed |
| POST   | `/step`   | `{"action": {...}}` | Take one treatment action step                |
| GET    | `/state`  | —                   | Current episode metadata and step count       |
| GET    | `/docs`   | —                   | Auto-generated FastAPI documentation          |


**Valid action types:** `noop` · `start` · `add` · `stop` · `switch` · `dose_adjust`

---

## Project Structure

```
DART/
│
├── app.py                              ← Streamlit entry point (loads ui/app.py)
├── requirements.txt                    ← Core Python dependencies
├── requirements_hackathon.txt          ← Hackathon-specific extras
├── Dockerfile                          ← OpenEnv server container (port 7860)
├── openenv.yaml                        ← OpenEnv manifest
├── council.py                          ← Multi-agent council fusion layer
├── self_improvement.py                 ← Exploration and self-repair controller
├── README.md                           ← You are here
├── LICENSE
│
├── env/
│   ├── digital_twin_env.py             ← DigitalTwinDiabetesEnv — Gym-style core
│   ├── patient_twin.py                 ← Stochastic patient physiology simulator
│   ├── actions.py                      ← Action schema and safety validator
│   └── fall_detection.py              ← Safety layer — intercepts unsafe actions
│
├── reward/
│   └── rubric.py                       ← RewardRubric — 9 named, decomposed components
│
├── tools/
│   ├── ehr.py                          ← Structured patient history
│   ├── genomics.py                     ← Variant-aware drug metabolism signals
│   ├── interactions.py                 ← Drug–drug interaction checker
│   ├── progression_forecast.py         ← Short-horizon trajectory projection
│   ├── trial_sim.py                    ← In-silico drug trialing
│   ├── biomarkers.py                   ← Biomarker signal interpretation
│   ├── resistance.py                   ← Drug resistance heuristics
│   └── risk.py                         ← CV and renal risk scoring
│
├── training/
│   ├── DART_Colab_submission.ipynb     ← Submission notebook (training + all figures)
│   ├── colab_episode_rl.py             ← REINFORCE helpers and clinical trace logging
│   └── council_rollout.py              ← Council episode runner
│
├── scripts/
│   ├── train_reinforce_twin.py         ← Training CLI (--quick / --judge-preset / custom)
│   ├── run_sanity.py                   ← Environment wiring check
│   ├── run_evaluation.py               ← Full eval across N seeds
│   ├── run_env_demo.py                 ← Single episode walkthrough
│   ├── plot_colab_publication.py       ← Training curve, bars, glucose, council figures
│   ├── plot_colab_judge_insights.py    ← Judge dashboard — clinical state, rubric, boxplots
│   ├── generate_readme_demo_figures.py ← Regenerate committed README figures (no GPU)
│   └── upload_figures_to_hf_space.py  ← Push generated figures to HF Space repo
│
├── dtm_openenv/
│   └── server/
│       └── app.py                      ← FastAPI OpenEnv server (/reset /step /state /health)
│
├── evaluation/
│   ├── baselines.py                    ← Random and heuristic baseline agents
│   └── metrics.py                      ← Evaluation metrics and episode graders
│
├── ui/
│   └── app.py                          ← Streamlit dashboard and clinical trace viewer
│
├── docs/
│   └── figures/                        ← Generated PNGs committed after each Colab run
│
└── logs/
    └── colab_experiment.json           ← All episode data, traces, endpoints, and config

```

---

<div align="center">

*Built for OpenEnv Hackathon 2026 · Track #3.1 — World Modeling (Professional Tasks)*

*PRs and forks welcome.*

</div>