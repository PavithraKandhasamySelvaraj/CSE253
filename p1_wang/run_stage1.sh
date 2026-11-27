#!/usr/bin/env bash
# Stage 1: the paper's own pipeline (third_party/CI-Steering, unchanged),
# with judging routed to the shared judge. Then stage 2 (benchmark.py).
#
# Usage (from p1_wang/):  ./run_stage1.sh <hf-model-id> [steps...]
#   steps default to all of: extract read ci synthetic confaide privaci utility benchmark
# Example:  ./run_stage1.sh Qwen/Qwen2.5-7B-Instruct extract read ci
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: ./run_stage1.sh <hf-model-id> [steps...]}"
shift
STEPS="${*:-extract read ci synthetic confaide privaci utility benchmark}"
SHORT="${MODEL##*/}"
PY=.venv/bin/python
ACT=outputs/activations/$SHORT
READ=outputs/reading/$SHORT
CI=outputs/ci_decomposition/$SHORT
ALPHAS="0.5 1.0 2.0 4.0"   # paper's sweep; it reports alpha = 1.0
mkdir -p logs

run() {  # run <step-name> <paper-script> [args...]
  local step=$1; shift
  echo "=== $step ($SHORT) $(date '+%F %T')"
  "$PY" run_paper.py "$@" 2>&1 | tee "logs/${SHORT}_${step}.log"
}

for step in $STEPS; do
  case $step in
    extract)   run extract extract_activations --config configs/default.yaml --model "$MODEL" --batch-size 8 ;;
    read)      run read read_representations --activations-dir "$ACT" --output-dir "$READ" ;;
    ci)        run ci ci_decomposition --activations-dir "$ACT" --output-dir "$CI" ;;
    synthetic) run synthetic ci_steering_evaluation --model "$MODEL" --reader-dir "$READ/probe_reader" \
                 --ci-dir "$CI" --alphas $ALPHAS --output-dir "outputs/ci_steering_comparison/$SHORT" ;;
    confaide)  run confaide confaide_ci_steering --model "$MODEL" --reader-dir "$READ/probe_reader" \
                 --ci-dir "$CI" --alphas $ALPHAS --output-dir "outputs/confaide_ci/$SHORT" ;;
    privaci)   run privaci privaci_ci_steering --model "$MODEL" --reader-dir "$READ/probe_reader" \
                 --ci-dir "$CI" --alphas $ALPHAS --output-dir "outputs/privaci_ci/$SHORT" ;;
    utility)   run utility utility_evaluation --model "$MODEL" --reader-dir "$READ/probe_reader" \
                 --ci-dir "$CI" --output-dir "outputs/utility_eval/$SHORT" ;;
    benchmark) echo "=== benchmark ($SHORT) $(date '+%F %T')"
               "$PY" benchmark.py --model "$MODEL" 2>&1 | tee "logs/${SHORT}_benchmark.log" ;;
    *) echo "unknown step: $step" >&2; exit 1 ;;
  esac
done
