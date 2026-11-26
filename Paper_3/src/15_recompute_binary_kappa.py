"""
Step 15 — Recompute agreement on the BINARY collapse (Disclosed vs.
Not-Disclosed) from the confusion matrix already saved by
10_judge_validation_agreement.py. No new labeling, no GPU — pure arithmetic
on judge_validation_results.json.

Why this matters now: the updated protocol requires kappa >= 0.7 before any
number can be reported. The original 4-class kappa (0.387-0.416) fails that
bar, but the binary collapse — which is the ONLY thing this whole pipeline
actually uses downstream — may clear it even though the 4-class number
doesn't, since most 4-class disagreement is Unavailable-vs-Refused noise
that collapses to the same binary label either way.

Output: config.DATA_DIR/judge_validation_binary_results.json
"""
import config
import utils

LABELS = ["Disclosed", "Unavailable", "Ambiguous", "Refused"]


def to_binary(label):
    return "Disclosed" if label == "Disclosed" else "Not Disclosed"


def cohen_kappa_from_confusion(confusion, labels):
    """confusion[human_label][judge_label] = count"""
    n = sum(confusion[h][j] for h in labels for j in labels)
    po = sum(confusion[l][l] for l in labels) / n
    pe = 0.0
    for l in labels:
        p_human = sum(confusion[l][j] for j in labels) / n
        p_judge = sum(confusion[h][l] for h in labels) / n
        pe += p_human * p_judge
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    try:
        results = utils.load_json(f"{config.DATA_DIR}/judge_validation_results.json")
    except FileNotFoundError:
        print(f"ERROR: judge_validation_results.json not found in {config.DATA_DIR}")
        print("Run 09_build_judge_validation_sample.py and 10_judge_validation_agreement.py first.")
        return

    confusion_4class = results["confusion_matrix_rows_human_cols_judge"]

    # Collapse the 4x4 confusion matrix into a 2x2 (Disclosed vs. Not Disclosed)
    binary_labels = ["Disclosed", "Not Disclosed"]
    confusion_binary = {h: {j: 0 for j in binary_labels} for h in binary_labels}
    for human_label in LABELS:
        for judge_label in LABELS:
            count = confusion_4class[human_label][judge_label]
            h_bin = to_binary(human_label)
            j_bin = to_binary(judge_label)
            confusion_binary[h_bin][j_bin] += count

    n_total = sum(confusion_binary[h][j] for h in binary_labels for j in binary_labels)
    n_agree = sum(confusion_binary[l][l] for l in binary_labels)
    binary_accuracy = n_agree / n_total
    binary_kappa = cohen_kappa_from_confusion(confusion_binary, binary_labels)

    PASS_THRESHOLD = 0.7
    passes = binary_kappa >= PASS_THRESHOLD

    output = {
        "n_validated": n_total,
        "binary_accuracy": binary_accuracy,
        "binary_cohen_kappa": binary_kappa,
        "confusion_matrix_binary": confusion_binary,
        "protocol_threshold": PASS_THRESHOLD,
        "passes_protocol_threshold": passes,
        "original_4class_kappa_for_reference": results["cohen_kappa"],
    }
    utils.save_json(output, f"{config.DATA_DIR}/judge_validation_binary_results.json")

    print("===== BINARY JUDGE VALIDATION (what the pipeline actually uses) =====")
    print(f"n = {n_total}")
    print(f"Binary accuracy: {binary_accuracy:.3f}")
    print(f"Binary Cohen's kappa: {binary_kappa:.3f}")
    print(f"Protocol threshold: kappa >= {PASS_THRESHOLD}")
    print(f"\n{'PASSES' if passes else 'DOES NOT PASS'} the protocol's reporting threshold.")

    if not passes:
        print("\nNEXT STEP REQUIRED: per the updated protocol, you must either improve the")
        print("judge prompt and re-validate, or escalate to a larger judge, before reporting")
        print("any number derived from this judge. Do not report Stage 2 metrics as final yet.")
    else:
        print("\nYou may report binary-label-derived metrics (AUROC, recall@1%FPR, disclosure")
        print("rates, jailbreak rate) as validated. The 4-class kappa remains below threshold —")
        print("note in NOTES.md that only the binary collapse was validated to the required bar.")

    print(f"\nSaved to {config.DATA_DIR}/judge_validation_binary_results.json")


if __name__ == "__main__":
    main()
