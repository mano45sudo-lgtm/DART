from __future__ import annotations

"""
Hugging Face Spaces entrypoint (Streamlit).

Spaces expects `app.py` at repo root. `st.set_page_config` must be the **first**
Streamlit call in the process; delegating via `run_path(..., __main__)` breaks that,
so we load `ui/app.py` as a module and call `_run_app_ui()` after configuring here.
"""

import importlib.util
import sys
from pathlib import Path

import streamlit as st

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

st.set_page_config(page_title="Digital Twin Medicine (T2DM)", layout="wide")

_spec = importlib.util.spec_from_file_location("_dart_ui_app", _root / "ui" / "app.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod._run_app_ui()
