#!/usr/bin/env bash
# One-time setup for P1 (from p1_wang/): paper code at a pinned commit,
# external benchmarks, and the Python environment.
set -euo pipefail
cd "$(dirname "$0")"

clone() {  # clone <url> <dir> <commit>
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "$2" checkout -q "$3"
}

clone https://github.com/wang2226/CI-Steering.git third_party/CI-Steering 5c5edc3261f9cdf11319b47107894e08b2b40c60
# Upstream syntax error in src/control/rep_tuning.py (see NOTES.md)
git -C third_party/CI-Steering apply --check ../../patches/ci-steering-fixes.patch 2>/dev/null \
  && git -C third_party/CI-Steering apply ../../patches/ci-steering-fixes.patch
clone https://github.com/skywalker023/confaide.git data/confaide 3174c63e689226189675a740b7bf9053f10bae50
clone https://github.com/HKUST-KnowComp/PrivaCI-Bench.git data/privaci_bench cb49d187095062bfff554121d4fa31440a32578e
ln -sfn third_party/CI-Steering/configs configs

uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

.venv/bin/python run_paper.py generate_stimuli \
  --num-pairs-per-type 50 --num-function-pairs 200 --num-ci-per-condition 100
echo "Setup done. Log in to Hugging Face (hf auth login) for Llama and Gemma access."
