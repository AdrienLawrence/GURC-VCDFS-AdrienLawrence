"""Run Bench201 commands from the repository root, regardless of IDE working directory."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def launch(*arguments):
    subprocess.run([sys.executable, "-B", *map(str, arguments)], cwd=ROOT, check=True)
