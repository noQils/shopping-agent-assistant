"""Streamlit entrypoint for the shopping assistant app.

Streamlit reruns this file after each interaction. Use run_module so the
actual app body executes on every rerun instead of being skipped by Python's
normal import cache.
"""

from runpy import run_module

run_module("src.app", run_name="__main__")
