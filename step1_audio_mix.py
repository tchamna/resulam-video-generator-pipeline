#!/usr/bin/env python3
"""Compatibility wrapper around `step1_audio_processing_v2`.

This project accumulated multiple "step1" variants. Keep `step1_audio_mix.py`
importable (tools may import it) while centralizing the actual implementation
in `step1_audio_processing_v2.py`.
"""
from __future__ import annotations

from step1_audio_processing_v2 import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("step1_audio_processing_v2", run_name="__main__")

