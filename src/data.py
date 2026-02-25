"""Data acquisition utilities.

This project uses IBM's Telco Customer Churn sample dataset (CSV).
We download it directly from a public GitHub raw URL.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .config import DATA_URL, RAW_DATA_PATH

def download_data(url: str = DATA_URL, out_path: str = RAW_DATA_PATH, force: bool = False) -> str:
    """Download the dataset to `out_path`.

    Args:
        url: Public URL to a CSV.
        out_path: Local path where CSV will be saved.
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded file as a string.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        return str(out)

    with urllib.request.urlopen(url) as resp:
        content = resp.read()

    out.write_bytes(content)
    return str(out)
