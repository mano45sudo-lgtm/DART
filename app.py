from __future__ import annotations

"""
Hugging Face Spaces entrypoint (Streamlit).

Spaces expects `app.py` at repo root for Streamlit SDK.
This file delegates to `ui/app.py`.
"""

import runpy

runpy.run_path("ui/app.py", run_name="__main__")

