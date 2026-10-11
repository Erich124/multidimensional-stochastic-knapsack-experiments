"""Run all Round 3 stages in priority order."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    stages = [
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["-m", "experiments.run_exact"],
        ["-m", "experiments.run_stress_search"],
        ["-m", "experiments.run_scaling"],
        ["-m", "experiments.make_tables"],
        ["-m", "experiments.make_figures"],
    ]
    for args in stages:
        subprocess.run([sys.executable] + args, check=True)


if __name__ == "__main__":
    main()

