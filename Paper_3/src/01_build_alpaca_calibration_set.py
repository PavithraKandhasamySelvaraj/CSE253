"""
Shared Stage 2 infrastructure — run ONCE for the whole team, not once per paper.

Builds the ~1,000 neutral Alpaca prompts used by every paper's benchmark port
to set its detection threshold (99th percentile of scores on this set, per
P4's procedure in the shared protocol). If each paper samples its own 1,000
prompts, the thresholds — and therefore the AUROC/recall@1%FPR numbers —
are not actually comparable across papers. This file exists exactly once,
with a fixed seed, and every paper's Stage 2 code reads from it.

Output: alpaca_calibration_1000.json
  A list of 1,000 {"id": int, "instruction": str, "input": str} objects.
  "input" is often an empty string — that's normal for Alpaca; it means the
  instruction is self-contained with no separate input to append.

Each paper's own code is responsible for turning these into whatever prompt
format its target model and scoring pipeline need (e.g. P3 would NOT ask
these as "what is the sexual orientation of X" questions — these are generic,
unrelated instructions used purely to measure what the detector's score
distribution looks like on ordinary text with no sensitive content).
"""
import json
import random

SEED = 0
N_PROMPTS = 1000
OUTPUT_PATH = "alpaca_calibration_1000.json"


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing the `datasets` library...")
        import os
        os.system("pip install -q datasets")
        from datasets import load_dataset

    print("Downloading tatsu-lab/alpaca (one-time download, ~45 MB)...")
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    print(f"Full Alpaca dataset size: {len(ds)}")

    random.seed(SEED)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    chosen_indices = sorted(indices[:N_PROMPTS])

    prompts = []
    for i, idx in enumerate(chosen_indices):
        row = ds[idx]
        prompts.append({
            "id": i,
            "source_index": idx,   # the index in the full Alpaca dataset, for traceability
            "instruction": row["instruction"],
            "input": row["input"],
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {OUTPUT_PATH}")
    print(f"Seed used: {SEED} (fixed — do not change, or the set won't match other papers' copies)")
    print("\nSample of 3 prompts:")
    for p in prompts[:3]:
        print(f"  [{p['id']}] instruction: {p['instruction'][:80]!r}  input: {p['input'][:40]!r}")

    print(f"\nIMPORTANT: distribute {OUTPUT_PATH} to every paper's team member as-is.")
    print("Do NOT regenerate this file per-paper — everyone must use this exact same file")
    print("for their threshold calibration to be comparable across papers.")


if __name__ == "__main__":
    main()
