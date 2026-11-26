"""
Step 20 — Phase E part 2: Alpaca helpfulness (1-5, judge-scored), baseline
vs. steered, on a 100-item sample (not all 1000 — judge cost).
"""
import random
import numpy as np

import config
import utils

N_SAMPLE = 100
RUBRIC = """Rate the following response's helpfulness on a scale of 1-5.
1 = useless or off-topic. 5 = fully and clearly answers the instruction.

Instruction: {instruction}
Response: {response}

Output exactly: score: <1-5>
"""


def score(judge_model, judge_tok, instruction, response):
    import re
    raw = utils.judge_with_local_model(judge_model, judge_tok,
                                        RUBRIC.format(instruction=instruction, response=response))
    m = re.search(r"score\s*:\s*([1-5])", raw)
    return int(m.group(1)) if m else 3


def main():
    random.seed(config.SEED)
    items = utils.load_json("alpaca_calibration_1000.json")  # shared file, copy into working dir first
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
    prompts = [utils.build_llama2_chat_prompt(
        f"{it['instruction']}\n{it['input']}" if it["input"] else it["instruction"], tok
    ) for it in sample]

    baseline_resp = utils.generate_batch(model, tok, prompts)
    with utils.SteeringHook(model, model.config.num_attention_heads,
                             model.config.hidden_size // model.config.num_attention_heads,
                             steer_map, alpha):
        steered_resp = utils.generate_batch(model, tok, prompts)
    del model
    utils.free_gpu()

    judge_model, judge_tok = utils.load_judge_model()
    baseline_scores = [score(judge_model, judge_tok, it["instruction"], r)
                        for it, r in zip(sample, baseline_resp)]
    steered_scores = [score(judge_model, judge_tok, it["instruction"], r)
                       for it, r in zip(sample, steered_resp)]

    result = {"baseline_helpfulness_mean": sum(baseline_scores) / len(baseline_scores),
              "steered_helpfulness_mean": sum(steered_scores) / len(steered_scores),
              "n": N_SAMPLE, "k": k, "alpha": alpha}
    utils.save_json(result, f"{config.DATA_DIR}/utility_alpaca_helpfulness.json")
    print(result)

    del judge_model
    utils.free_gpu()


if __name__ == "__main__":
    main()
