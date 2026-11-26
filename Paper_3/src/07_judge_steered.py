"""
Step 7 — Judge every steered response, then run factuality verification on a
sample of the ones labeled "Disclosed".

Output:
  config.STEERED_LABELS_PATH -> list of records with an added "label" field
  config.FACTUALITY_PATH     -> list of {"name","k","alpha","response","factuality_label"}
"""
import random

import config
import utils


def main():
    random.seed(config.SEED)
    records = utils.load_json(config.STEERED_RESP_PATH)
    judge_model, judge_tok = utils.load_judge_model()

    for i, rec in enumerate(records):
        raw = utils.judge_with_local_model(
            judge_model, judge_tok,
            utils.PRIVACY_JUDGE_PROMPT.format(
                attribute="sexual orientation", name=rec["name"], response=rec["response"]
            ),
        )
        rec["label"] = utils.parse_privacy_label(raw)
        if i % 200 == 0:
            print(f"{i}/{len(records)} steered responses judged")
            utils.save_json(records, config.STEERED_LABELS_PATH)

    utils.save_json(records, config.STEERED_LABELS_PATH)
    print(f"Saved judged steered responses to {config.STEERED_LABELS_PATH}")

    # ---- Factuality sample ----
    disclosed = [r for r in records if r["label"] == "Disclosed"]
    print(f"\n{len(disclosed)} steered responses labeled Disclosed, out of {len(records)}")
    random.shuffle(disclosed)
    sample = disclosed[:config.FACTUALITY_SAMPLE_SIZE]

    factuality_results = []
    for rec in sample:
        raw = utils.judge_with_local_model(
            judge_model, judge_tok,
            utils.FACTUALITY_JUDGE_PROMPT.format(
                attribute="sexual orientation", name=rec["name"], response=rec["response"]
            ),
        )
        label = utils.parse_factuality_label(raw)
        factuality_results.append({**rec, "factuality_label": label})

    utils.save_json(factuality_results, config.FACTUALITY_PATH)
    n_factual = sum(1 for r in factuality_results if r["factuality_label"] == "factual")
    print(f"Factuality sample: {n_factual}/{len(factuality_results)} judged factual")
    print("REMINDER: manually spot-check a few of these before trusting the number —")
    print("a local 4-bit 9B judge is noisier than the paper's GPT-4o-mini.")
    print(f"Saved factuality sample to {config.FACTUALITY_PATH}")

    del judge_model
    utils.free_gpu()


if __name__ == "__main__":
    main()
