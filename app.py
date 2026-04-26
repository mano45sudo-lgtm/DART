"""Hugging Face Spaces entrypoint (Streamlit).

Spaces runs this file as the main script. Streamlit calls ``on_script_start()``
before your module body runs, so any ``st.*`` delta emitted during imports can
block ``set_page_config``. Keep the smallest possible preamble before
``import streamlit`` and ``st.set_page_config`` (see Streamlit ``ScriptRunContext``).

``ui/app.py`` is loaded with ``exec_module`` under a non-``__main__`` name so it
does **not** call ``set_page_config`` again (that only runs for ``streamlit run ui/app.py``).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Digital Twin Medicine (T2DM)", layout="wide")

import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location("_dart_ui_app", _root / "ui" / "app.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# dataclasses (and other tools) resolve cls.__module__ via sys.modules; exec_module
# does not register the module until after the body finishes, so register first.
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
_mod._run_app_ui()
