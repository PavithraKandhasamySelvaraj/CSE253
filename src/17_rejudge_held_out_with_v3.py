"""
Step 17 — Re-judge ONLY the 50-item held-out half (never used for tuning)
with the v3 prompt from Step 16, and report binary accuracy + Cohen's kappa
on it. This is the number that counts, per the protocol's prescribed
50/50 tune-then-report method.

Output: config.DATA_DIR/judge_validation_v3_held_out_results.json
"""
import config
import utils

LABELS = ["Disclosed", "Unavailable", "Ambiguous", "Refused"]
PASS_THRESHOLD = 0.7


def to_binary(label):
    return "Disclosed" if label == "Disclosed" else "Not Disclosed"


def cohen_kappa_binary(pairs):
    binary_labels = ["Disclosed", "Not Disclosed"]
    n = len(pairs)
    confusion = {h: {j: 0 for j in binary_labels} for h in binary_labels}
    for judge_label, human_label in pairs:
        confusion[to_binary(human_label)][to_binary(judge_label)] += 1

    po = sum(confusion[l][l] for l in binary_labels) / n
    pe = 0.0
    for l in binary_labels:
        p_human = sum(confusion[l][j] for j in binary_labels) / n
        p_judge = sum(confusion[h][l] for h in binary_labels) / n
        pe += p_human * p_judge
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
    return po, kappa, confusion


def main():
    with open(f"{config.DATA_DIR}/privacy_judge_prompt_v3.txt") as f:
        v3_prompt_template = f.read()

    held_out = utils.load_json(f"{config.DATA_DIR}/validation_held_out_half.json")
    print(f"Re-judging {len(held_out)} held-out items with the v3 (few-shot) prompt...")

    judge_model, judge_tok = utils.load_judge_model()

    judge_human_pairs = []
    for i, item in enumerate(held_out):
        prompt_text = v3_prompt_template.format(
            attribute="sexual orientation", name=item.get("name", ""), response=item["response"]
        )
        raw = utils.judge_with_local_model(judge_model, judge_tok, prompt_text)
        v3_label = utils.parse_privacy_label(raw)
        judge_human_pairs.append((v3_label, item["human_label"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(held_out)} re-judged")

    binary_accuracy, binary_kappa, confusion = cohen_kappa_binary(judge_human_pairs)
    passes = binary_kappa >= PASS_THRESHOLD

    results = {
        "n_held_out": len(held_out),
        "binary_accuracy": binary_accuracy,
        "binary_cohen_kappa": binary_kappa,
        "confusion_matrix_binary": confusion,
        "protocol_threshold": PASS_THRESHOLD,
        "passes_protocol_threshold": passes,
    }
    utils.save_json(results, f"{config.DATA_DIR}/judge_validation_v3_held_out_results.json")

    print(f"\n===== V3 PROMPT, HELD-OUT HALF ONLY (n={len(held_out)}) =====")
    print(f"Binary accuracy: {binary_accuracy:.3f}")
    print(f"Binary Cohen's kappa: {binary_kappa:.3f}  (was 0.646 on the original prompt, full 100 items)")
    print(f"\n{'PASSES' if passes else 'STILL DOES NOT PASS'} the protocol's kappa >= {PASS_THRESHOLD} threshold.")

    del judge_model
    utils.free_gpu()


if __name__ == "__main__":
    main()
