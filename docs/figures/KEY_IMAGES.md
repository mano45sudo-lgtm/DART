# Key figures (Colab → `docs/figures/`)

After **`training/DART_Colab_submission.ipynb`** (or `python scripts/generate_readme_demo_figures.py`), these files are produced:

| File | Source script |
|------|----------------|
| `training_curve.png` | `plot_colab_publication.py` |
| `final_comparison_bars.png` | `plot_colab_publication.py` |
| `behavior_glucose.png` | `plot_colab_publication.py` |
| `self_repair_episodes.png` | `plot_colab_publication.py` |
| `judge_clinical_state.png` | `plot_colab_judge_insights.py` |
| `judge_step_and_cumulative_return.png` | `plot_colab_judge_insights.py` |
| `judge_action_mix.png` | `plot_colab_judge_insights.py` |
| `judge_rubric_episode_totals.png` | `plot_colab_judge_insights.py` |
| `judge_outcome_distributions.png` | `plot_colab_judge_insights.py` |
| `judge_council_glucose_example.png` | `plot_colab_judge_insights.py` |

With `--also-svg`, matching **`.svg`** files are written (text; fine for some Git remotes).

**GitHub:** commit `docs/figures/*` and `logs/colab_experiment.json`, then `git push origin main`.

**Hugging Face Space (host files without `git push` of PNG):**

```bash
export HF_TOKEN=hf_...   # read/write for your account
python scripts/upload_figures_to_hf_space.py --repo mano678/DART_1
```

- **Browse on Hub:** [spaces/mano678/DART_1 → docs/figures](https://huggingface.co/spaces/mano678/DART_1/tree/main/docs/figures)
- **Direct file (pattern):** `https://huggingface.co/spaces/mano678/DART_1/resolve/main/docs/figures/<filename>.png`
