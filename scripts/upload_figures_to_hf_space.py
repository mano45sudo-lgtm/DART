#!/usr/bin/env python3
"""
Upload `docs/figures/*.{png,svg,json optional}` to a Hugging Face **Space** via the Hub API.
This avoids `git push` to the Hub (which may reject binary PNG) while still hosting images on the Space.

Requires:  pip install huggingface_hub
Token:     huggingface-cli login   or   export HF_TOKEN=hf_...

Usage:
  python scripts/upload_figures_to_hf_space.py
  python scripts/upload_figures_to_hf_space.py --repo mano678/DART_1 --figures-dir docs/figures
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="mano678/DART_1", help="Space id (user/name)")
    p.add_argument("--branch", type=str, default="main")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "docs" / "figures",
    )
    p.add_argument(
        "--patterns",
        nargs="*",
        default=("*.png", "*.svg"),
        help="Which file globs to upload (default: png and svg)",
    )
    p.add_argument(
        "--readme",
        action="store_true",
        help="Also upload repo root README.md (Space card + project readme on Hub).",
    )
    p.add_argument(
        "--streamlit",
        action="store_true",
        help="Also upload app.py, ui/app.py, requirements.txt, and LICENSE for the Space build.",
    )
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip docs/figures upload (only --readme / --streamlit parts).",
    )
    args = p.parse_args()
    fdir: Path = args.figures_dir
    if not args.skip_figures:
        if not fdir.is_dir():
            print("missing figures dir:", fdir, file=sys.stderr)
            sys.exit(1)
    try:
        from huggingface_hub import HfApi, login
    except ImportError as e:
        print("install huggingface_hub:", e, file=sys.stderr)
        sys.exit(1)

    tok = (os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        login(token=tok, add_to_git_credential=False)
    api = HfApi()
    if not args.skip_figures:
        files: list[Path] = []
        for pat in args.patterns:
            files.extend(sorted(fdir.glob(pat)))
        files = [x for x in files if x.is_file()]
        if not files:
            print("no files matched under", fdir, file=sys.stderr)
            sys.exit(1)
        for fp in files:
            rel = f"docs/figures/{fp.name}"
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=rel,
                repo_id=args.repo,
                repo_type="space",
                revision=args.branch,
                commit_message=f"Add/update figure: {fp.name}",
            )
            print("uploaded", rel)
    if args.readme:
        rm = repo_root / "README.md"
        if rm.is_file():
            api.upload_file(
                path_or_fileobj=str(rm),
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="space",
                revision=args.branch,
                commit_message="Sync README from GitHub",
            )
            print("uploaded README.md")
        else:
            print("skip README: missing", rm, file=sys.stderr)
    if args.streamlit:
        for rel_name in ("app.py", "ui/app.py", "requirements.txt", "LICENSE"):
            src = repo_root / rel_name
            if not src.is_file():
                print("skip streamlit: missing", src, file=sys.stderr)
                continue
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=rel_name,
                repo_id=args.repo,
                repo_type="space",
                revision=args.branch,
                commit_message=f"Sync Streamlit: {rel_name}",
            )
            print("uploaded", rel_name)
    print("done. View:", f"https://huggingface.co/spaces/{args.repo}/tree/{args.branch}/docs/figures")


if __name__ == "__main__":
    main()
