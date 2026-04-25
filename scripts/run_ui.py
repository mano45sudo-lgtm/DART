from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

if __name__ == "__main__":
    # Run via: python scripts/run_ui.py
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(repo_root / "ui" / "app.py")]
    raise SystemExit(stcli.main())

