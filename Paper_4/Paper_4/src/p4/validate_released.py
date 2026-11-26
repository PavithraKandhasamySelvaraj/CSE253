"""Stage-1 metric check, no GPU: recompute Apollo's results_table.csv for the released
Llama-3.3-70B layer-22 Instructed-Pairs probe from their per-token scores, using OUR metric code.
Released labels are used (agent.md §4.4: "use their labels, not ours")."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.metrics import detection_metrics, reduce_mean
from src.p4.apollo import APOLLO

EX = APOLLO / "example_results" / "instructed_pairs"
COLS = ["n_honest", "n_ambiguous", "n_deceptive", "auroc", "auroc_vs_alpaca",
        "recall 1.0%", "fpr 1.0%", "recall 0.2%", "fpr 0.2%"]


def main() -> tuple[pd.DataFrame, float]:
    scores = json.loads((EX / "scores.json").read_text())
    control = json.loads((EX / "control_scores.json").read_text())["alpaca__plain"]
    ctrl = reduce_mean([np.array(s, dtype=np.float64) for s in control["scores"]])
    ref = pd.read_csv(EX / "results_table.csv", index_col=0)

    rows = {}
    for name, d in scores.items():
        s = reduce_mean([np.array(x, dtype=np.float64) for x in d["scores"]])
        labels = [lab.lower() for lab in d["labels"]]
        rows[name] = detection_metrics(s, labels, ctrl)
    ours = pd.DataFrame(rows).T
    # Apollo's "0.2%" columns are really FPR=0.0025 (format rounding); align names
    ours = ours.rename(columns={"recall 0.25%": "recall 0.2%", "fpr 0.25%": "fpr 0.2%"})
    ours = ours[COLS].astype(float)
    ref = ref[COLS].astype(float)
    diff = (ours - ref.loc[ours.index]).abs()
    return ours, float(diff.max().max())


if __name__ == "__main__":
    ours, maxdiff = main()
    pd.set_option("display.width", 200)
    print(ours.round(4))
    print(f"\nmax |ours - released results_table.csv| = {maxdiff:.2e}")
