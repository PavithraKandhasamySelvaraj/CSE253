"""
Step 10 — Run this AFTER you've downloaded judge_validation_blind.csv,
filled in the "human_label" column by reading each response yourself, and
re-uploaded it back to the same path (overwriting the blank version).

Merges your human labels with the judge's own labels (from
judge_validation_full.csv, saved internally in step 9) and computes
agreement: overall accuracy, Cohen's kappa, a per-class breakdown, and a
list of every disagreement so you can eyeball what the judge got wrong.

Robust to encoding mangling ("mojibake") introduced by Excel/Google Sheets
round-trips on names with accented characters (e.g. "García Márquez" turning
into "GarcÃ­a MÃ¡rquez") — matches are attempted on the raw item_id first,
then on a de-mojibake'd + accent-stripped version, then falls back to
matching by row position (both CSVs were written in the same order).

Output: config.DATA_DIR/judge_validation_results.json
"""
import csv
import unicodedata

import config
import utils

LABELS = ["Disclosed", "Unavailable", "Ambiguous", "Refused"]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def denoise(s):
    """Best-effort fix for UTF-8-read-as-Latin-1 mojibake, then strip
    diacritics, so 'GarcÃ­a' and 'García' and 'Garcia' all normalize the
    same way for matching purposes."""
    try:
        s2 = s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        s2 = s
    s3 = "".join(c for c in unicodedata.normalize("NFKD", s2) if not unicodedata.combining(c))
    return s3.strip().lower()


def main():
    full_path = f"{config.DATA_DIR}/judge_validation_full.csv"
    blind_path = f"{config.DATA_DIR}/judge_validation_blind.csv"

    full_rows_list = load_csv(full_path)
    blind_rows = load_csv(blind_path)

    # exact-match index
    full_by_exact = {r["item_id"]: r for r in full_rows_list}
    # denoised-match index (handles mojibake from Excel/Sheets round-trips)
    full_by_denoised = {denoise(r["item_id"]): r for r in full_rows_list}

    same_length_same_order = len(full_rows_list) == len(blind_rows)

    merged = []
    missing_human_label = 0
    bad_label = 0
    unmatched = 0
    match_method_counts = {"exact": 0, "denoised": 0, "position": 0}

    for i, row in enumerate(blind_rows):
        human_label = row.get("human_label", "").strip()
        if not human_label or human_label.lower() == "nan":
            missing_human_label += 1
            continue
        if human_label not in LABELS:
            print(f"WARNING: item {row['item_id']} has unrecognized human_label "
                  f"'{human_label}' — must be one of {LABELS}. Skipping.")
            bad_label += 1
            continue

        full_row = full_by_exact.get(row["item_id"])
        method = "exact"
        if full_row is None:
            full_row = full_by_denoised.get(denoise(row["item_id"]))
            method = "denoised"
        if full_row is None and same_length_same_order:
            full_row = full_rows_list[i]
            method = "position"
        if full_row is None:
            print(f"WARNING: could not match item_id '{row['item_id']}' to "
                  f"judge_validation_full.csv by any method — skipping this row.")
            unmatched += 1
            continue

        match_method_counts[method] += 1
        merged.append({
            "item_id": row["item_id"],
            "response": row["response"],
            "judge_label": full_row["judge_label"],
            "human_label": human_label,
        })

    print(f"Match methods used: {match_method_counts}")
    if unmatched:
        print(f"WARNING: {unmatched} rows could not be matched at all and were dropped.")
    print(f"Merged {len(merged)} labeled items "
          f"({missing_human_label} still blank, {bad_label} had an invalid label)")

    if missing_human_label > 0:
        print(f"\nNOTE: {missing_human_label} items in judge_validation_blind.csv still "
              f"have no human_label. Agreement is computed only on the {len(merged)} "
              f"items that ARE labeled. Finish labeling the rest for the full ~100-item "
              f"validation the protocol asks for.")

    if not merged:
        print("No labeled items yet — nothing to compute. Label some rows and re-run.")
        return

    n_agree = sum(1 for m in merged if m["judge_label"] == m["human_label"])
    accuracy = n_agree / len(merged)

    def cohen_kappa(y1, y2, labels):
        n = len(y1)
        po = sum(1 for a, b in zip(y1, y2) if a == b) / n
        pe = 0.0
        for label in labels:
            p1 = sum(1 for y in y1 if y == label) / n
            p2 = sum(1 for y in y2 if y == label) / n
            pe += p1 * p2
        if pe == 1.0:
            return 1.0
        return (po - pe) / (1 - pe)

    judge_labels_list = [m["judge_label"] for m in merged]
    human_labels_list = [m["human_label"] for m in merged]
    kappa = cohen_kappa(judge_labels_list, human_labels_list, LABELS)

    per_class = {}
    for label in LABELS:
        human_as_label = [m for m in merged if m["human_label"] == label]
        if not human_as_label:
            per_class[label] = {"n": 0, "judge_agreement_rate": None}
            continue
        agree = sum(1 for m in human_as_label if m["judge_label"] == label)
        per_class[label] = {
            "n": len(human_as_label),
            "judge_agreement_rate": agree / len(human_as_label),
        }

    confusion = {h: {j: 0 for j in LABELS} for h in LABELS}
    for m in merged:
        confusion[m["human_label"]][m["judge_label"]] += 1

    disagreements = [
        {"item_id": m["item_id"], "judge_label": m["judge_label"],
         "human_label": m["human_label"], "response_excerpt": m["response"][:200]}
        for m in merged if m["judge_label"] != m["human_label"]
    ]

    results = {
        "n_validated": len(merged),
        "n_still_blank": missing_human_label,
        "n_unmatched_dropped": unmatched,
        "match_method_counts": match_method_counts,
        "overall_accuracy": accuracy,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "confusion_matrix_rows_human_cols_judge": confusion,
        "disagreements": disagreements,
    }

    utils.save_json(results, f"{config.DATA_DIR}/judge_validation_results.json")

    print(f"\n===== JUDGE VALIDATION RESULTS ({len(merged)} items) =====")
    print(f"Overall accuracy (judge vs. human): {accuracy:.3f}")
    print(f"Cohen's kappa: {kappa:.3f}  "
          f"(>0.8 excellent, 0.6-0.8 good, 0.4-0.6 moderate, <0.4 weak)")
    print("\nPer-class (how often the judge matched human for items humans labeled X):")
    for label, stats in per_class.items():
        rate = f"{stats['judge_agreement_rate']:.3f}" if stats["judge_agreement_rate"] is not None else "n/a (0 items)"
        print(f"  {label:15s} n={stats['n']:3d}  agreement={rate}")

    print(f"\n{len(disagreements)} disagreements — see judge_validation_results.json for full list.")
    print("First few disagreements:")
    for d in disagreements[:5]:
        print(f"  {d['item_id']}: judge={d['judge_label']}, human={d['human_label']}")
        print(f"    \"{d['response_excerpt']}...\"")

    print(f"\nSaved full results to {config.DATA_DIR}/judge_validation_results.json")
    print("Add the overall_accuracy and cohen_kappa numbers to NOTES.md.")


if __name__ == "__main__":
    main()
