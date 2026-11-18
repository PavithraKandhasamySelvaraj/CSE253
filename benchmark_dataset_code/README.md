# Shared Stage 2 benchmark infrastructure

This folder holds the pieces of the shared benchmark protocol that must be
**bit-for-bit identical across all five papers** (P1-P5), not regenerated
independently by each person. Everything else — the actual test items,
counterfactual pairs, and attribute-specific prompts for each paper — stays
paper-specific, since P1-P5 each study a different sensitive attribute and
scenario.

## What's shared, and why

| File | Used for | Why it must be identical across papers |
|---|---|---|
| `alpaca_calibration_1000.json` | Setting each paper's detection threshold (99th percentile of scores on these prompts, per P4's procedure) | If each paper calibrates against a different random sample, the resulting AUROC/recall@1%FPR numbers aren't comparable to each other — you'd each be measuring against a different bar |
| `mmlu_utility_1000.json` | Utility cost: does an intervention damage general capability? | Same logic — "fixed 1,000 questions" per the protocol only means something if it's the same 1,000 for everyone |
| `judge.py` + `shared_judge_config.json` | Every LLM-judge call made by any paper's Stage 2 code | The protocol says "use one fixed judge for every method" — a shared config file alone isn't enough if everyone still writes their own calling code with subtly different temperature, retries, or truncation. Import `judge.py`, don't reimplement it. |

## What's NOT shared (and shouldn't be)

- The actual test items / prompts specific to each paper's attribute (P3's
  public-figure names, P1's leakage test strings, P2's candidate profiles,
  etc.) — these are necessarily different because the papers study different
  things.
- The judge *prompt content* for each paper's specific classification task
  (P3 asks "did this disclose sexual orientation," P1 asks something about
  leakage, etc.) — `judge.py` only holds the model/backend/temperature
  constant, not what question you're asking it.
- Each paper's own probe/steering/detection code.

## How to build these (run ONCE, by whoever sets this up first)

```bash
pip install -q datasets
python 01_build_alpaca_calibration_set.py
python 02_build_mmlu_utility_subset.py
```

Both use a fixed seed (0) and print exactly what was sampled. Once built,
**distribute the two resulting JSON files to every paper's team member as-is**
— do not have each person re-run these scripts with their own environment,
since library version differences in `datasets` could in principle return
different row ordering even with the same seed. Generate once, share the
files directly (e.g. as a shared Kaggle Dataset, a shared Drive folder, or
committed to a shared git repo).

## How each paper's Stage 2 code should use these

**Detection threshold calibration** (`alpaca_calibration_1000.json`):
1. Load the 1,000 prompts.
2. Run your paper's own target model + probe/detector on each one (these are
   neutral, non-sensitive prompts — your detector should mostly score them low).
3. Take the 99th percentile of your detector's scores on this set as your
   threshold.
4. Score your actual test items (P3: the counterfactual name pairs) against
   this threshold to compute recall@1%FPR.

**Utility cost** (`mmlu_utility_1000.json`):
1. Run your target model (unsteered/baseline) on all 1,000 questions, record accuracy.
2. Run your target model with your intervention applied (steering, filtering,
   etc.) on the same 1,000 questions, record accuracy.
3. Report the accuracy delta as your utility cost.
4. Also run the Alpaca-1000 set through your judge for a helpfulness score
   (1-5) on steered vs. unsteered outputs, per the protocol.

**Judge calls** (`judge.py`):
```python
import sys
sys.path.insert(0, "/path/to/benchmark_shared")   # wherever you put this folder
from judge import call_judge, get_judge_identity

raw_response = call_judge(your_fully_formatted_prompt)
my_label = my_own_parsing_function(raw_response)   # your own paper-specific parsing

# When writing your config.yaml, log which judge was actually used:
judge_id = get_judge_identity()   # e.g. "openai:gpt-4o-mini:temp0"
```

## The one team decision this forces: judge backend

`shared_judge_config.json` currently defaults to `gpt-4o-mini` via the OpenAI
API, matching the protocol's stated default. This needs an API key and costs
real money (typically a few tens of dollars total across all five papers'
Stage 2 judging — not per paper — depending on how many judge calls each
paper's benchmark makes).

If the team instead wants to run a free local judge for Stage 2 (as P3's
Stage 1 did with `gemma-2-9b-it`), **make that decision once, as a team, and
change `shared_judge_config.json` in this shared location** — not
independently per paper. If some papers use GPT-4o-mini and others use a
local model, the cross-paper comparison is broken in exactly the way this
whole shared-infrastructure exercise is trying to avoid. Whichever way the
team decides, every paper should end up with the same `backend` value in
this one file.

Note: if you do switch to local, re-run a judge validation step using the shared judge
settings, since quantized local judges can have materially lower agreement
with human labels than GPT-4o-mini.

## Checklist for whoever is setting this up

- [ ] Run both build scripts once, confirm the printed sample counts look right
- [ ] Share `alpaca_calibration_1000.json`, `mmlu_utility_1000.json`,
      `judge.py`, and `shared_judge_config.json` with every paper's team member
- [ ] Get team agreement on the judge backend (OpenAI vs. local) before
      anyone starts their Stage 2 judge-dependent work
- [ ] Each paper's `config.yaml` should record `get_judge_identity()`'s output,
      so it's traceable which judge configuration produced which numbers
- [ ] If the judge backend ever changes later, everyone must re-run their
      Stage 2 judging — numbers from different judge backends cannot be
      mixed in the same results table
