"""
Runs one of the paper's scripts unchanged, with the judge routed to the
team's shared judge (see judge_shim.py).

Usage (from p1_wang/):
    .venv/bin/python run_paper.py <script> [script args...]
e.g.
    .venv/bin/python run_paper.py extract_activations --model Qwen/Qwen2.5-7B-Instruct

<script> is a file name in third_party/CI-Steering/src without ".py".
Relative paths in the paper's scripts (data/, outputs/) resolve against
p1_wang/, so run this from p1_wang/.
"""
import runpy
import sys
from pathlib import Path

import compat
import judge_shim

PAPER_SRC = Path(__file__).resolve().parent / "third_party" / "CI-Steering" / "src"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = PAPER_SRC / f"{sys.argv[1]}.py"
    if not script.exists():
        sys.exit(f"No such paper script: {script}")

    compat.install()
    judge_shim.install()
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
