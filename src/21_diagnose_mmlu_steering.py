"""
Step 21 — Diagnose the -17.1pt MMLU drop from Step 19: is it real capability
loss, or is steering pushing generation off-format so extract_letter() can't
find a clean A/B/C/D in the first 5 tokens? Runs a small sample (50 items),
logs RAW steered output text so you can see the actual failure mode.
"""
import random
import re

import numpy as np

import config
import utils

N_SAMPLE = 50
LETTERS = ["A", "B", "C", "D"]


def build_mmlu_prompt(item, tok):
    choices_text = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, item["choices"]))
    text = f"{item['question']}\n{choices_text}\nAnswer with just the letter."
    return utils.build_llama2_chat_prompt(text, tok)


def extract_letter(text):
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def main():
    random.seed(config.SEED)
    items = utils.load_json("mmlu_utility_1000.json")
    sample = random.sample(items, N_SAMPLE)

    ranking = utils.load_json(config.HEAD_RANKING_PATH)
    weights = np.load(config.PROBES_PATH)["weights"]
    chosen = utils.load_json(f"{config.DATA_DIR}/chosen_configs.json")[0]
    k, alpha = chosen["k"], chosen["alpha"]
    top = ranking[:k]
    steer_map = {}
    for r in top:
        steer_map.setdefault(r["layer"], []).append((r["head"], weights[r["layer"], r["head"], :]))

    model, tok = utils.load_target_model()
    prompts = [build_mmlu_prompt(it, tok) for it in sample]

    # Use a LONGER max_new_tokens here (20 instead of 5) so if the model is just
    # rambling before giving a letter, we can still see it — Step 19 used 5,
    # which may itself be part of the problem.
    with utils.SteeringHook(model, model.config.num_attention_heads,
                             model.config.hidden_size // model.config.num_attention_heads,
                             steer_map, alpha):
        texts = utils.generate_batch(model, tok, prompts, max_new_tokens=20)

    n_no_letter = 0
    n_wrong_letter = 0
    n_correct = 0
    print(f"{'ITEM':<4} {'CORRECT':<8} {'EXTRACTED':<10} {'RAW OUTPUT'}")
    for it, text in zip(sample, texts):
        correct_letter = LETTERS[it["answer"]]
        extracted = extract_letter(text)
        status = "CORRECT" if extracted == correct_letter else ("NO_LETTER" if extracted is None else "WRONG")
        if extracted is None:
            n_no_letter += 1
        elif extracted != correct_letter:
            n_wrong_letter += 1
        else:
            n_correct += 1
        print(f"{it['id']:<4} {correct_letter:<8} {str(extracted):<10} {text[:100]!r}  [{status}]")

    print(f"\n===== SUMMARY (n={N_SAMPLE}, max_new_tokens=20) =====")
    print(f"Correct: {n_correct} ({n_correct/N_SAMPLE:.1%})")
    print(f"Wrong (clean letter, just incorrect): {n_wrong_letter} ({n_wrong_letter/N_SAMPLE:.1%})")
    print(f"No letter extracted at all (format failure): {n_no_letter} ({n_no_letter/N_SAMPLE:.1%})")
    print("\nIf 'no letter extracted' is a large share of the errors, the -17.1pt drop from")
    print("Step 19 is partly/mostly a format-extraction artifact (max_new_tokens=5 cutting")
    print("off the answer), not pure capability loss. Re-run Step 19 with max_new_tokens=20")
    print("if so, and report the corrected delta.")

    del model
    utils.free_gpu()


if __name__ == "__main__":
    main()
