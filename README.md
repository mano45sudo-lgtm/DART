# 🧬 Digital Twin Medicine — RL Agent for Personalized Treatment

> *"The same insulin dose that saves one patient sends another into hypoglycaemia. The same metformin that controls one patient's glucose does nothing for the next. Medicine prescribes to populations — this project prescribes to individuals."*

**OpenEnv Hackathon 2026 · Track #3.1 — World Modeling (Professional Tasks)**

| | |
|---|---|
| **UI (Streamlit)** | `<ADD_SPACE_URL_HERE>` |
| **OpenEnv Env Server (Docker/FastAPI)** | `<ADD_OPENENV_SPACE_URL_HERE>` |
| **Training Notebook (Colab)** | `<ADD_COLAB_URL_HERE>` |
| **Mini-blog / Demo video** | `<ADD_BLOG_OR_VIDEO_URL_HERE>` |

---

## 1 · The Problem — One-Size-Fits-Many Medicine Fails

Type 2 Diabetes (T2DM) affects **500+ million people** worldwide. Clinical guidelines usually start with the same first-line path, then escalate on failure. That linear ladder ignores what makes each patient different:

- Genomic variants affecting drug response (e.g. transport/metabolism differences)
- Comorbidities (CKD, CVD) that change which drugs are safe
- Lifestyle trajectory (diet, adherence, activity)
- Drug interactions and affordability constraints

**Capability gap:** most systems do not make *sequential personalized decisions under uncertainty* as new labs arrive week by week. This project targets that gap with an RL policy inside a stochastic patient digital twin.

---

## 2 · The Environment — Stochastic T2DM Digital Twin

The environment (`env/digital_twin_env.py`) simulates one patient's progression over a **52-week horizon** (`max_steps=52`). Each step = one simulated week.

### Observation (partial)

`week`, `hba1c`, `fasting_glucose`, `bmi`, `systolic_bp`, `egfr`, `ckd`, `cvd`

The policy does not see hidden latent parameters directly.

### Action (JSON each week)

```json
{"type":"start","drug":"metformin","dose":1.0,"lifestyle":0.7}
{"type":"add","drug":"glp1","dose":1.0}
{"type":"dose_adjust","drug":"metformin","dose":0.5}
{"type":"stop","drug":"sglt2"}
{"type":"switch","from_drug":"metformin","to_drug":"insulin","dose":0.8}
```

### Tool modules in the repo

| Tool module | Purpose |
|---|---|
| `tools/ehr.py` | Structured patient context |
| `tools/genomics.py` | Variant-aware signals |
| `tools/interactions.py` | Drug-drug interaction checks |
| `tools/progression_forecast.py` | Near-term trajectory estimation |
| `tools/trial_sim.py` | In-silico treatment trialing |
| `tools/biomarkers.py` | Biomarker interpretation |
| `tools/resistance.py` | Treatment resistance heuristics |
| `tools/risk.py` | CV/renal risk scoring |

### Reward rubric (`reward/rubric.py`)

- Positive: HbA1c reduction, fasting glucose reduction, target shaping, BMI trend
- Negative: side-effect severity, treatment cost
- Terminal: remission bonus / failure penalty

Dense reward discourages waiting until the final step and encourages stable week-to-week control.

---

## 3 · Training — RL Against Live Simulator Rollouts

Training loop: `scripts/train_reinforce_twin.py`

Every gradient update is from **on-policy rollouts** in `DigitalTwinDiabetesEnv` (not a static dataset).

```text
for update in range(N):
  trajectories = rollout_episode(policy, env)
  loss = REINFORCE(trajectories)
  optimizer.step()
```

### Run modes

| Mode | Command | Purpose |
|---|---|---|
| Smoke (fast CPU) | `python scripts/train_reinforce_twin.py --quick` | quick pipeline check with `sshleifer/tiny-gpt2` |
| Judge/demo | `python scripts/train_reinforce_twin.py --judge-preset` | longer schedule (`distilgpt2`, 120 updates, 4 eps/update, 32 held-out seeds) |

For larger checkpoints:

```bash
python scripts/train_reinforce_twin.py --judge-schedule --model distilgpt2 --out-json logs/training_last_distil.json
python scripts/train_reinforce_twin.py --judge-schedule --load-in-4bit --model unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit --out-json logs/training_last_8b.json
```

---

## 4 · Results — Current Artifacts in Repo

### Learning curve vs baselines

![Training curve vs random and untrained LM baselines](docs/figures/training_vs_baselines.png)

*Figure 1: Training/eval trend with random and untrained baselines.*

### Final held-out bar comparison

![Final eval: random vs trained LM](docs/figures/final_random_vs_trained.png)

*Figure 2: Final held-out comparison (`mean ± std`).*

### Random baseline return distribution

![Random agent returns per episode](logs/baseline_returns.png)

*Figure 3: Random policy episode-return variability in the stochastic twin.*


### Latest logged metrics (`logs/training_last.json`)

| Metric | Value |
|---|---|
| Model | `sshleifer/tiny-gpt2` (`--quick` smoke run) |
| Random baseline (final eval) | `6.45 ± 30.81` (`n=20`) |
| Trained LM (final eval) | `-24.85 ± 21.81` (`n=8`) |
| Parse rate (final eval) | `0.00` |

> This current JSON is a **smoke artifact**. If trained and untrained values are the same, that usually means parse rate is too low and policy updates are not effective. Re-run `--judge-preset` (or `--judge-schedule` + a stronger model), then re-commit updated JSON + figures.

---

## 5 · Why This Matters

| Stakeholder | Value |
|---|---|
| Clinicians | Auditable policy suggestions across labs, risk, interactions, and longitudinal progression |
| Patients | Personalized treatment sequencing instead of one-size-fits-many escalation |
| Health systems | Reward explicitly encodes cost and safety trade-offs |
| Researchers | Reproducible open simulator for sequential decision-making experiments |

---

## 6 · Repository Structure

```text
env/                     # DigitalTwinDiabetesEnv + patient dynamics
tools/                   # Clinical tool modules
reward/                  # Reward rubric
training/                # Colab notebooks
evaluation/              # Baseline + metrics scripts
ui/                      # Streamlit dashboard
scripts/                 # Train/eval/plot/demo entrypoints
dtm_openenv/             # OpenEnv-compatible FastAPI server + models
docs/figures/            # Committed plots
logs/                    # JSON logs + baseline plot
spaces/openenv/          # Docker Space for env server
spaces/hf_gpu_training/  # Docker Space for HF GPU training job
```

---

## 7 · Reproduce Locally

### macOS / Linux

```bash
git clone <YOUR_REPO_URL>
cd DART
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
python scripts/run_env_demo.py
python scripts/run_evaluation.py
python scripts/plot_rewards.py
python scripts/run_ui.py
```

### Windows (PowerShell)

```powershell
cd "C:\path\to\DART"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
python scripts/run_env_demo.py
python scripts/run_evaluation.py
python scripts/plot_rewards.py
python scripts/run_ui.py
```

---

## 8 · Google Colab

Open one notebook from `training/`:

| Notebook | Purpose |
|---|---|
| `train_trl_colab.ipynb` | smoke + single-model run |
| `colab_judge_pipeline.ipynb` | judge pipeline (baseline vs small vs larger model) |
| `colab_http_clinical_pipeline.ipynb` | HTTP server demo pipeline |

Recommended flow:

1. Enable GPU runtime
2. Run install + setup cells
3. Run smoke once
4. Run judge schedule (`--judge-preset` or `--judge-schedule`)
5. Commit updated `logs/*.json` and `docs/figures/*.png`


---

## 9 · Hugging Face Spaces

### Streamlit UI Space

- SDK: Streamlit
- Entry: root `app.py` (delegates to `ui/app.py`)

### OpenEnv env server Space

- SDK: Docker
- Subdirectory: `spaces/openenv`
- Command: `uvicorn dtm_openenv.server.app:app --port 7860`
- Health: `/health`
- Docs: `/docs`

---

## 10 · Compliance Checklist

| Requirement | Status |
|---|---|
| Gym-style env (`reset`, `step`, `state`) | ✅ |
| OpenEnv server separation under `dtm_openenv/server/` | ✅ |
| Live environment rollouts for training | ✅ |
| Committed judge-facing plots in repo | ✅ |
| Baseline vs trained quantitative metrics | ✅ |

---

## 11 · Next Improvements

- Multi-patient curriculum training schedules
- Stronger 4-bit checkpoints with matched eval protocol
- Richer uncertainty and causal evaluation diagnostics

---

*Built for OpenEnv Hackathon 2026. PRs and forks welcome.*
