"""Utilities for exporting figures and tables for reports.

These helpers are intentionally small so they can be reused from
both notebooks and scripts.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def get_export_dir(base_dir: Optional[str] = None, subdir: str = "exports") -> str:
    """Return (and create) the directory for exported artifacts.

    Parameters
    ----------
    base_dir:
        Base directory for exports. Defaults to the current working
        directory when ``None``.
    subdir:
        Name of the subdirectory inside ``base_dir``.
    """

    if base_dir is None:
        base_dir = os.getcwd()

    export_dir = os.path.join(base_dir, subdir)
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


def save_figure(filename: str, export_dir: str) -> str:
    """Save the current matplotlib figure to ``export_dir/filename``.

    Returns the absolute path of the saved file.
    """

    path = os.path.join(export_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, filename: str, export_dir: str) -> str:
    """Save a DataFrame as CSV to ``export_dir/filename`` and return the path."""

    path = os.path.join(export_dir, filename)
    df.to_csv(path, index=True)
    return path
