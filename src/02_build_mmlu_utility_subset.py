"""
Shared Stage 2 infrastructure — run ONCE for the whole team, not once per paper.

Builds the fixed 1,000-question MMLU subset every paper's benchmark port uses
to measure utility cost (does an intervention — steering, a guardrail, a
filter — damage general capability?). Stratified across all 57 MMLU subjects
so no single subject dominates the sample, with a fixed seed so the exact
same 1,000 questions are used by every paper.

Output: mmlu_utility_1000.json
  A list of 1,000 {"id", "subject", "question", "choices" (list of 4),
  "answer" (int 0-3, index into choices)} objects.

Each paper's own code runs its target model (steered and unsteered) on these
same 1,000 questions and reports the accuracy delta as the utility cost.
"""
import json
import random
from collections import defaultdict

SEED = 0
N_QUESTIONS = 1000
OUTPUT_PATH = "mmlu_utility_1000.json"


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing the `datasets` library...")
        import os
        os.system("pip install -q datasets")
        from datasets import load_dataset

    print("Downloading cais/mmlu (config='all', split='test')...")
    ds = load_dataset("cais/mmlu", "all")["test"]
    print(f"Full MMLU test set size: {len(ds)}")

    by_subject = defaultdict(list)
    for i, row in enumerate(ds):
        by_subject[row["subject"]].append(i)

    subjects = sorted(by_subject.keys())
    print(f"Number of subjects: {len(subjects)}")

    random.seed(SEED)
    for subj in subjects:
        random.shuffle(by_subject[subj])

    # Proportional allocation: give each subject a quota proportional to its
    # size in the full test set, then fill any shortfall from subjects with
    # leftover questions, so the total lands exactly at N_QUESTIONS.
    total_available = len(ds)
    quotas = {}
    allocated = 0
    for subj in subjects:
        quota = round(N_QUESTIONS * len(by_subject[subj]) / total_available)
        quota = min(quota, len(by_subject[subj]))
        quotas[subj] = quota
        allocated += quota

    # Adjust for rounding drift to hit exactly N_QUESTIONS
    diff = N_QUESTIONS - allocated
    subj_cycle = list(subjects)
    random.shuffle(subj_cycle)
    i = 0
    while diff != 0 and subj_cycle:
        subj = subj_cycle[i % len(subj_cycle)]
        if diff > 0 and quotas[subj] < len(by_subject[subj]):
            quotas[subj] += 1
            diff -= 1
        elif diff < 0 and quotas[subj] > 0:
            quotas[subj] -= 1
            diff += 1
        i += 1
        if i > 10000:  # safety valve, should never trigger
            break

    chosen_global_indices = []
    for subj in subjects:
        chosen_global_indices.extend(by_subject[subj][:quotas[subj]])
    chosen_global_indices.sort()

    questions = []
    for i, idx in enumerate(chosen_global_indices):
        row = ds[idx]
        questions.append({
            "id": i,
            "source_index": idx,
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} questions to {OUTPUT_PATH}")
    print(f"Seed used: {SEED} (fixed — do not change)")
    print(f"Subjects represented: {len(set(q['subject'] for q in questions))}")
    print("\nPer-subject counts (first 10):")
    counts = defaultdict(int)
    for q in questions:
        counts[q["subject"]] += 1
    for subj, count in list(sorted(counts.items()))[:10]:
        print(f"  {subj}: {count}")

    print(f"\nIMPORTANT: distribute {OUTPUT_PATH} to every paper's team member as-is.")
    print("Do NOT regenerate per-paper — the utility-cost numbers across papers")
    print("are only comparable if everyone measures accuracy on the same 1,000 questions.")


if __name__ == "__main__":
    main()
