"""Optional: push training JSON + PNGs to a Hub Dataset (for HF GPU Spaces)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


def main() -> None:
    repo_id = os.environ.get("HF_UPLOAD_REPO", "").strip()
    if not repo_id:
        print("upload_hf_space_artifacts: HF_UPLOAD_REPO not set; skipping.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("upload_hf_space_artifacts: huggingface_hub missing")
        return

    paths = [
        repo_root / "logs" / "training_last.json",
        repo_root / "docs" / "figures" / "training_vs_baselines.png",
        repo_root / "docs" / "figures" / "final_random_vs_trained.png",
    ]
    api = HfApi()
    for p in paths:
        if not p.is_file():
            print("upload_hf_space_artifacts: skip missing", p)
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("uploaded", p.name, "->", repo_id)


if __name__ == "__main__":
    main()
