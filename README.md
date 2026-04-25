---
title: Digital Twin Medicine — RL Agent for Personalized Treatment
emoji: "🧬"
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app.py
---

## Digital Twin Medicine — RL Agent for Personalized Treatment

- **Theme**: OpenEnv Hackathon 2026 — **#3.1 World Modeling (Professional Tasks)**
- **Core capability gap**: sequential treatment decisions for *an individual patient* under uncertainty (not a one-shot predictor).
- **Deliverable**: a **digital twin environment** + **tool-augmented RL agent loop** + **end-to-end training evidence**.

## Links (fill before submission)
- **UI Space (Streamlit)**: `<ADD_SPACE_URL_HERE>`
- **OpenEnv Env Space (Docker/FastAPI)**: `<ADD_OPENENV_SPACE_URL_HERE>`
- **Google Colab training**: `<ADD_COLAB_URL_HERE>`
- **Mini-blog / video (<2 min)**: `<ADD_BLOG_OR_VIDEO_URL_HERE>`

## 1) Problem → Environment → Agent (3-minute overview)

- **Problem**: “one-size-fits-many” medicine fails because the *same* drug sequence produces different outcomes across individuals.
- **Environment**: a **partially observable** T2DM patient digital twin; each step = **1 simulated week**; state evolves stochastically with treatment effects + side effects + costs.
- **Agent task**: output a **JSON treatment action** each week (start/add/stop/switch/dose_adjust + lifestyle).
- **Tools**: EHR, genomics, interactions, progression forecast, trial sim, biomarkers, resistance, risk (all implemented in `tools/`).
- **Reward**: dense clinical improvement + trend shaping, plus penalties (toxicity/cost) and terminal outcomes.

## 2) What the agent sees / does (OpenEnv-style)

### Observation (partial)
Returned every step from `env.reset()` / `env.step()`:
- `week`, `hba1c`, `fasting_glucose`, `bmi`, `systolic_bp`, `egfr`, `ckd`, `cvd`

### Action (JSON)
Examples:
```json
{"type":"start","drug":"metformin","dose":1.0,"lifestyle":0.7}
```
```json
{"type":"add","drug":"glp1","dose":1.0}
```

## 3) Reward (dense + hard-to-game)
Implemented in `reward/rubric.py`:
- **Clinical improvement**: HbA1c + fasting glucose reduction
- **Biomarker shaping**: distance-to-target + BMI trend
- **Side effects**: severity-weighted penalty
- **Cost**: weekly cost penalty
- **Terminal**: remission bonus / failure penalty

## 4) End-to-end training (env rollouts, not a static dataset)

The training loop lives in `scripts/train_reinforce_twin.py`. Each REINFORCE update calls `rollout_episode` → `DigitalTwinDiabetesEnv.reset` / `step` with a live LM policy, so gradients come from **on-policy trajectories in the simulator**, not from a fixed offline dataset. Baselines are a **random policy** (and an **untrained LM** line on the learning-curve figure) evaluated on the same env.

**Two run modes**

| Mode | Command | Role |
|------|---------|------|
| Smoke (fast CPU check) | `python scripts/train_reinforce_twin.py --quick` | CI / layout verification; uses `sshleifer/tiny-gpt2` (may not parse valid JSON actions). |
| Judge / demo (meaningful length) | `python scripts/train_reinforce_twin.py --judge-preset` | **Recommended for reviewers:** `distilgpt2`, 120 updates, 4 episodes/update, 32 held-out eval seeds, 80 random baseline episodes. Use a GPU if you can. |

After any run, **re-commit** `logs/training_last.json` and the two PNGs under `docs/figures/` so judges see your latest numbers and plots. Optional one-liner to **stage** those paths for `git commit` (run from a git checkout):

```bash
python scripts/train_reinforce_twin.py --judge-preset --git-stage-artifacts
```

Figures are written as **high-resolution PNG** (160 DPI) with **labeled axes** (update index; episode return in the env’s reward units).

### Plots (committed in-repo)

**Learning curve + baselines** (`docs/figures/training_vs_baselines.png`):

![Training curve vs random and untrained LM baselines](docs/figures/training_vs_baselines.png)

**Final bar comparison** (held-out seeds; `docs/figures/final_random_vs_trained.png`):

![Final eval random vs trained LM](docs/figures/final_random_vs_trained.png)

**Random baseline only** (`logs/baseline_returns.png` from `scripts/plot_rewards.py`):

![Random agent returns per episode](logs/baseline_returns.png)

### Latest logged metrics (from committed `logs/training_last.json`)

These numbers update whenever you re-run training and overwrite the JSON. *Current commit (smoke `--quick` run):*

| Metric | Random baseline | Untrained LM (eval@0) | Trained LM (final held-out) |
|--------|-----------------|------------------------|-----------------------------|
| Mean episode return | 6.45 (std 30.81, *n*=20 eval episodes) | −24.85 (std 21.81, *n*=8) | −24.85 (std 21.81, *n*=8) |
| JSON action parse rate | — | 0.00 | 0.00 |

The smoke model does not emit parseable tool JSON reliably; **run `--judge-preset` with `distilgpt2`**, then refresh this table from the new `training_last.json` before submission.

### Notebook (Colab)

- `training/train_trl_colab.ipynb` — installs deps, runs smoke or judge training, displays the same PNGs.

### Weights & Biases (optional)

If you log a run to Wandb, add the **direct URL to that run** here (plots may still be mirrored in-repo as above):

- `<ADD_WANDB_RUN_URL_HERE>`

## 5) Reproduce locally (Windows PowerShell)

```powershell
cd "C:\Users\DELL\Downloads\Digital_twin"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
```

### Environment demo (single episode)
```powershell
python scripts/run_env_demo.py
```

### Baseline evaluation + plot (evidence)
```powershell
python scripts/run_evaluation.py
python scripts/plot_rewards.py
```

### UI (Streamlit dashboard)
```powershell
python scripts/run_ui.py
```
Open the printed URL (usually `http://localhost:8501`).

## 6) Reproduce locally (macOS/Linux)

```bash
cd Digital_twin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
```

## 7) Google Colab (end-to-end env training)

Open:
- `training/train_trl_colab.ipynb` — quick smoke + single-model judge preset
- `training/colab_judge_pipeline.ipynb` — **judge pipeline**: random baseline vs **DistilGPT2** vs **Unsloth/large** (same `--judge-schedule`), two bar charts + optional curve overlay
- `training/colab_http_clinical_pipeline.ipynb` — **HTTP demo**: `uvicorn dtm_openenv.server.app:app`, rule-based vs LM policies with **real `DTMAction` JSON** (not MetaGuard-style strings), optional glucose trajectory plot

Colab steps:
1. **File → Open notebook → GitHub** (paste your repo URL) and open the notebook.
2. Run install cell(s).
3. Set `REPO_ROOT` in the notebook (e.g., `/content/mano/DART`).
4. Run the quick smoke cell once:
   - `python scripts/train_reinforce_twin.py --quick`
5. For submission-quality curves, run the **judge preset** (GPU recommended); equivalent to the long manual flags:
   - `python scripts/train_reinforce_twin.py --judge-preset`
   - Optional: `python scripts/train_reinforce_twin.py --judge-preset --git-stage-artifacts` then `git commit` from the repo root.
6. **Commit** these paths so reviewers see plots and numbers in GitHub (not only inside Colab):
   - `logs/training_last.json`
   - `docs/figures/training_vs_baselines.png`
   - `docs/figures/final_random_vs_trained.png`

### 7b) Colab: random baseline + small GPT vs ~8B (Unsloth 4-bit) on one figure

Use the **same eval protocol** for every run (defaults already match if you only change `--model` and `--out-json`).

1. **Runtime:** `Runtime → Change runtime type → GPU` (T4 minimum; **A100** much safer for 8B + training).
2. **Install:** `pip install -q -r requirements_hackathon.txt` (includes `bitsandbytes`, `transformers`, `torch`).
3. **HF access:** Many Llama checkpoints are **gated**. On the model page click *Access repository*, then in Colab run `huggingface-cli login` and paste a **read** token, or set the `HF_TOKEN` secret and `huggingface_hub.login()`.
4. **Pick the model:** On [huggingface.co/unsloth](https://huggingface.co/unsloth) search for **8B** and prefer a **`bnb-4bit`** (or `unsloth-bnb-4bit`) weight so one GPU fits. Example: `unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit`.
5. **Train the small baseline LM** (separate JSON so you can compare later):

   `python scripts/train_reinforce_twin.py --judge-schedule --model distilgpt2 --out-json logs/training_last_distil.json`

6. **Train the 8B model** (4-bit load; add `--trust-remote-code` only if the model card says so):

   `python scripts/train_reinforce_twin.py --judge-schedule --load-in-4bit --model unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit --out-json logs/training_last_8b.json`

7. **One comparison PNG** (random baseline + both learning curves + final bars):

   `python scripts/plot_compare_training_runs.py --run logs/training_last_distil.json:DistilGPT2 --run logs/training_last_8b.json:8B-Instruct-4bit --out docs/figures/compare_small_vs_8b.png`

8. **Commit:** add `logs/training_last_distil.json`, `logs/training_last_8b.json`, `docs/figures/compare_small_vs_8b.png`, and embed the image in your README or writeup.

If you run out of VRAM, lower workload before changing eval seeds (e.g. `--updates 60 --episodes-per-update 2`) so the two runs stay **identical** in schedule.

## 7c) Spend Hugging Face **credits** on training (GPU Space, step-by-step)

Downloading weights into Colab does **not** use Hub credits. To use the **$30 (or any) HF balance**, run training on **Hugging Face’s paid GPU** — here via a **Docker Space** that executes `train_reinforce_twin.py` once, then **exits** so billing stops.

### A) Credits and billing

1. Open **[huggingface.co/settings/billing](https://huggingface.co/settings/billing)** and confirm your **balance / credits** (or redeem the voucher your organizers sent).
2. Read **[Using GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)** — you are billed **per minute** while the Space hardware is **Starting** or **Running** on a paid GPU.

### B) Create the training Space

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**.
2. **Owner:** your user or org. **Space name:** e.g. `dart-gpu-training`. **License:** match your repo.
3. **SDK:** **Docker** (not Gradio).
4. **Hardware:** leave **CPU basic** until the first build succeeds, then upgrade (next step). This avoids paying GPU during a broken build loop.
5. **Create** the Space, then connect **GitHub** (or Git) so this repository is linked — same flow as your other Space.

### C) Point the Space at the training Dockerfile

1. In the Space: **Settings → Dev mode** (or **Files and versions**): set the **Dockerfile path** to  
   `DART/spaces/hf_gpu_training/Dockerfile`  
   (if your **Git root** is the parent repo that contains `DART/`).
2. If your **Git root is only `DART`** (no `DART/` subfolder), set a **Docker build argument** in Space **Settings → Variables** (build-time):  
   `DART_PREFIX` = `.`  
   and set the Dockerfile path to `spaces/hf_gpu_training/Dockerfile` (paths relative to DART root).

### D) Secrets (gated models)

1. **Settings → Secrets**: add **`HF_TOKEN`** with a **read** token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) so `from_pretrained` can download gated weights.

### E) What command runs (Space “Variables”)

1. **Settings → Variables** (runtime): add **`TRAIN_ARGS`** (single line), for example:

   `--judge-schedule --model distilgpt2 --out-json logs/training_last.json`

   For an 8B 4-bit run (VRAM permitting):

   `--judge-schedule --load-in-4bit --model unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit --out-json logs/training_last_8b.json`

2. Optional — save outputs to a **Dataset** repo so files survive after the job ends: create an empty **Dataset** on HF (e.g. `yourname/dart-training-runs`), then add variable **`HF_UPLOAD_REPO`** = `yourname/dart-training-runs`. The entrypoint runs `scripts/upload_hf_space_artifacts.py` after training.

### F) Turn on the GPU (this uses credits)

1. **Settings → Hardware**: choose **Nvidia T4 - small** (or larger if you need VRAM; see [GPU pricing](https://huggingface.co/docs/hub/spaces-gpus)).
2. **Save / Rebuild** and open the **Logs** tab until you see `wrote .../training_last.json` and plot paths.
3. When the job finishes, the container **exits** so you are not left with an idle GPU loop — **still** open **Settings → Hardware** and set back to **CPU basic** (or **Pause** the Space) so you do not pay for a restarted Space you forgot about.

### G) Get plots into your GitHub README

1. If you used **`HF_UPLOAD_REPO`**, open that **Dataset** on the Hub and **download** `training_last.json` and the PNGs into your laptop repo under `logs/` and `docs/figures/`, then `git add` / `git commit` / `git push`.
2. Or copy the same files from the Space **build logs** only if you attached them as build artifacts (not reliable) — **Dataset upload is recommended**.

Template files for this Space live under **`spaces/hf_gpu_training/`** in this repo.

## 8) Hugging Face Spaces (discoverable + runnable)

### Space A — UI (Streamlit)
- SDK: **Streamlit**
- Entry: root `app.py` (delegates to `ui/app.py`)

### Space B — OpenEnv Env Server (Docker/FastAPI) — required
- SDK: **Docker**
- **Repository subdirectory**: `spaces/openenv`
- Runs: `uvicorn dtm_openenv.server.app:app --port 7860`

Verify endpoints:
- `/health` → `{"ok": true}`
- `/docs` → interactive FastAPI docs

## 9) Engineering compliance checklist (judges’ table stakes)

- **Gym-style API**: `env.digital_twin_env.DigitalTwinDiabetesEnv` implements `reset`, `step`, `state` ✅
- **Valid `openenv.yaml`**: `spec_version: 1` + client/action/observation/state entries ✅
- **Client/server separation**: env server code lives under `dtm_openenv/server/` ✅
- **No reserved tool names used as tools**: tools are in `tools/*.py` (not named reset/step/state/close) ✅

```text
env/           # digital twin environment + patient twin
tools/         # tool system modules
reward/        # reward rubric
training/      # notebooks
evaluation/    # baseline + metrics
ui/            # Streamlit dashboard UI
dtm_openenv/   # OpenEnv-style server + models
scripts/       # entrypoints
logs/          # reward curves, run logs
spaces/openenv # Docker Space for env server
```

