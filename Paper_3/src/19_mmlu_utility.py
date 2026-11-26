"""
Step 19 — Phase E part 1: MMLU-1000 accuracy, baseline vs. steered.
"""
import re
import numpy as np
import torch

import config
import utils

LETTERS = ["A", "B", "C", "D"]


def build_mmlu_prompt(item, tok):
    choices_text = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, item["choices"]))
    text = f"{item['question']}\n{choices_text}\nAnswer with just the letter."
    return utils.build_llama2_chat_prompt(text, tok)


def extract_letter(text):
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def run_mmlu(model, tok, items, steer_map=None, alpha=None, batch_size=8):
    correct = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [build_mmlu_prompt(it, tok) for it in batch]
        if steer_map:
            with utils.SteeringHook(model, model.config.num_attention_heads,
                                     model.config.hidden_size // model.config.num_attention_heads,
                                     steer_map, alpha):
                texts = utils.generate_batch(model, tok, prompts, max_new_tokens=5)
        else:
            texts = utils.generate_batch(model, tok, prompts, max_new_tokens=5)
        for it, t in zip(batch, texts):
            pred = extract_letter(t)
            if pred == LETTERS[it["answer"]]:
                correct += 1
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(items)} done")
    return correct / len(items)


def main():
    items = utils.load_json("mmlu_utility_1000.json")  # shared file, copy into working dir first
    ranking = utils.load_json(config.HEAD_RANKING_PATH)
    weights = np.load(config.PROBES_PATH)["weights"]
    chosen = utils.load_json(f"{config.DATA_DIR}/chosen_configs.json")[0]
    k, alpha = chosen["k"], chosen["alpha"]

    top = ranking[:k]
    steer_map = {}
    for r in top:
        steer_map.setdefault(r["layer"], []).append((r["head"], weights[r["layer"], r["head"], :]))

    model, tok = utils.load_target_model()

    print("Baseline MMLU accuracy...")
    baseline_acc = run_mmlu(model, tok, items)
    print(f"Baseline accuracy: {baseline_acc:.3f}")

    print(f"\nSteered (k={k}, alpha={alpha}) MMLU accuracy...")
    steered_acc = run_mmlu(model, tok, items, steer_map, alpha)
    print(f"Steered accuracy: {steered_acc:.3f}")

    result = {"baseline_accuracy": baseline_acc, "steered_accuracy": steered_acc,
              "accuracy_delta": steered_acc - baseline_acc, "k": k, "alpha": alpha, "n": len(items)}
    utils.save_json(result, f"{config.DATA_DIR}/utility_mmlu.json")
    print(f"\nAccuracy delta (utility cost): {result['accuracy_delta']:.3f}")

    del model
    utils.free_gpu()


if __name__ == "__main__":
    main()
