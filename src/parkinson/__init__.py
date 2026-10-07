"""Parkinson OFF motor score prediction package.

Exposes `PROJECT_ROOT`: the absolute path to the project root,
derived from this file's location. Any module that needs to resolve
a project-relative path (data files, configs) imports this constant
instead of hard-coding a CWD-relative string.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
