"""
Step 9 — Build a ~100-item sample for hand-labeling, to validate the local
judge against human judgment (required by the shared protocol, §3: "Validate
it on ~100 hand-labeled items. Log the judge prompt.").

Pulls from BOTH baseline_labels.json and steered_labels.json so the sample
covers both unsteered and steered text, stratified across the judge's four
classes (Disclosed / Unavailable / Ambiguous / Refused) so agreement can be
checked per-class, not just overall.

Produces TWO files:
  - judge_validation_blind.csv  -> what you actually hand-label (no judge
    label shown, to avoid anchoring your own judgment on the model's).
  - judge_validation_full.csv   -> same items WITH the judge's label, kept
    internally; script 10 merges this with your filled-in blind labels.

Also writes judge_prompt_log.txt, the exact prompt template used — the
protocol requires the prompt be logged, not just the model name.
"""
import csv
import random

import config
import utils

TARGET_SAMPLE_SIZE = 100
LABELS = ["Disclosed", "Unavailable", "Ambiguous", "Refused"]


def build_pool():
    pool = []

    baseline_responses = utils.load_json(config.BASELINE_RESP_PATH)
    baseline_labels = utils.load_json(config.BASELINE_LABELS_PATH)
    for name, label_info in baseline_labels.items():
        pool.append({
            "item_id": f"baseline:{name}",
            "source": "baseline",
            "name": name,
            "k": "",
            "alpha": "",
            "response": baseline_responses.get(name, ""),
            "judge_label": label_info["label"],
        })

    try:
        steered_labels = utils.load_json(config.STEERED_LABELS_PATH)
        for i, rec in enumerate(steered_labels):
            pool.append({
                "item_id": f"steered:{rec['name']}:{rec['k']}:{rec['alpha']}:{i}",
                "source": "steered",
                "name": rec["name"],
                "k": rec["k"],
                "alpha": rec["alpha"],
                "response": rec["response"],
                "judge_label": rec["label"],
            })
    except FileNotFoundError:
        print("steered_labels.json not found yet — validating baseline judgments only.")

    return pool


def stratified_sample(pool, target_total, seed):
    random.seed(seed)
    by_label = {label: [p for p in pool if p["judge_label"] == label] for label in LABELS}
    for label in LABELS:
        random.shuffle(by_label[label])

    per_class_target = target_total // len(LABELS)
    sample = []
    shortfall = 0
    for label in LABELS:
        available = by_label[label]
        take = min(per_class_target, len(available))
        sample.extend(available[:take])
        shortfall += per_class_target - take

    # redistribute shortfall from classes that have leftover items
    if shortfall > 0:
        leftovers = []
        for label in LABELS:
            taken = sum(1 for s in sample if s["judge_label"] == label)
            leftovers.extend(by_label[label][taken:])
        random.shuffle(leftovers)
        sample.extend(leftovers[:shortfall])

    random.shuffle(sample)  # don't let the labeler see items grouped by class
    return sample


def main():
    pool = build_pool()
    print(f"Total judged items available to sample from: {len(pool)}")

    counts = {label: sum(1 for p in pool if p["judge_label"] == label) for label in LABELS}
    print("Available per class:", counts)

    sample = stratified_sample(pool, TARGET_SAMPLE_SIZE, config.SEED)
    print(f"Sample size: {len(sample)}")
    sample_counts = {label: sum(1 for s in sample if s["judge_label"] == label) for label in LABELS}
    print("Sample per class:", sample_counts)

    full_path = f"{config.DATA_DIR}/judge_validation_full.csv"
    blind_path = f"{config.DATA_DIR}/judge_validation_blind.csv"

    with open(full_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "source", "name", "k", "alpha",
                                                "response", "judge_label"])
        writer.writeheader()
        for row in sample:
            writer.writerow(row)

    with open(blind_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "source", "name", "k", "alpha",
                                                "response", "human_label"])
        writer.writeheader()
        for row in sample:
            blind_row = {k: v for k, v in row.items() if k != "judge_label"}
            blind_row["human_label"] = ""  # fill with one of: Disclosed / Unavailable / Ambiguous / Refused
            writer.writerow(blind_row)

    with open(f"{config.DATA_DIR}/judge_prompt_log.txt", "w") as f:
        f.write("=== PRIVACY JUDGE PROMPT TEMPLATE ===\n\n")
        f.write(utils.PRIVACY_JUDGE_PROMPT)
        f.write("\n\n=== FACTUALITY JUDGE PROMPT TEMPLATE ===\n\n")
        f.write(utils.FACTUALITY_JUDGE_PROMPT)
        f.write(f"\n\nJudge model: {config.JUDGE_MODEL} (backend: {config.JUDGE_BACKEND})\n")

    print(f"\nSaved {blind_path} — DOWNLOAD THIS, hand-label the 'human_label' column")
    print("(values: Disclosed / Unavailable / Ambiguous / Refused), then re-upload it.")
    print(f"Saved {full_path} (internal — has the judge's own labels, for merging later).")
    print(f"Saved judge prompt log to {config.DATA_DIR}/judge_prompt_log.txt")


if __name__ == "__main__":
    main()
