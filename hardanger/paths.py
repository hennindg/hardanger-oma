r"""
Where the data, results and literature tables are.

The recordings (.mat files) are NOT part of the repository (several GB, CC-BY 4.0 data set,
DOI 10.21400/5ng8980s). Download them with tools/download_data.py, or point the scripts at an
existing copy by setting environment variables before running a script, e.g. in PowerShell:

    $env:HARDANGER_DATA = "D:\hardanger_data"
    $env:HARDANGER_RESULTS = "D:\hardanger_results"

Defaults: data/ and results/ inside the repository folder (both are ignored by git).
"""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_FOLDER = Path(os.environ.get("HARDANGER_DATA", REPOSITORY / "data"))
RESULTS_FOLDER = Path(os.environ.get("HARDANGER_RESULTS", REPOSITORY / "results"))
LITERATURE_FOLDER = REPOSITORY / "literature"
