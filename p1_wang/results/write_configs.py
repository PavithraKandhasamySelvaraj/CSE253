"""
Writes results/<model>/config.yaml (agent.md: one config per run) and copies the
stage-1 summary JSONs from outputs/ (git-ignored) into results/<model>/stage1/.

Usage (from p1_wang/):  .venv/bin/python results/write_configs.py
"""
import json
import shutil
from pathlib import Path

import yaml

P1 = Path(__file__).resolve().parent.parent
OUT = P1 / "outputs"
MODELS = {
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
JUDGE = {"model": "google/gemma-2-9b-it", "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
         "precision": "bf16", "temperature": 0, "backend": "local (benchmark_dataset/judge.py via judge_shim.py)"}
# Stage-1 summary files: <source relative to outputs/> -> <name in results/<model>/stage1/>
SUMMARIES = {
    "reading/{m}/reading_results.json": "reading_results.json",
    "reading/{m}/layer_summary.json": "reading_layer_summary.json",
    "reading/{m}/probe_reader/probe_scores.json": "probe_scores.json",
    "reading/{m}/pca_reader/layer_scores.json": "pca_layer_scores.json",
    "ci_decomposition/{m}/ci_decomposition_results.json": "ci_decomposition_results.json",
    "ci_steering_comparison/{m}/ci_comparison_results.json": "synthetic_ci_comparison_results.json",
    "confaide_ci/{m}/{m}/confaide_ci_transfer_results.json": "confaide_ci_transfer_results.json",
    "utility_eval/{m}/utility_evaluation.json": "utility_evaluation.json",
}


def stage1_layers(short: str) -> list[int]:
    for line in open(P1 / "logs" / f"{short}_synthetic.log"):
        if "Steering layers:" in line:
            return json.loads(line.split("Steering layers:")[1].strip())
    raise ValueError(f"no steering layers in {short}_synthetic.log")


def main():
    for short, (hf_id, rev) in MODELS.items():
        res = P1 / "results" / short
        dst = res / "stage1"
        dst.mkdir(parents=True, exist_ok=True)
        for src, name in SUMMARIES.items():
            shutil.copy2(OUT / src.format(m=short), dst / name)

        bench = json.load(open(res / "metrics.json"))["config"]
        synth = json.load(open(dst / "synthetic_ci_comparison_results.json"))
        conf = json.load(open(dst / "confaide_ci_transfer_results.json"))
        util = json.load(open(dst / "utility_evaluation.json"))
        cfg = {
            "paper": "P1 Wang, Xiong & Shu, arXiv 2604.00209",
            "model": {"id": hf_id, "revision": rev, "dtype": bench["dtype"], "device": bench["device"]},
            "code": {"upstream": "https://github.com/wang2226/CI-Steering",
                     "upstream_commit": "5c5edc3261f9cdf11319b47107894e08b2b40c60",
                     "local_fixes": "patches/ci-steering-fixes.patch"},
            "judge": JUDGE,
            "seed": 42,
            "generation": {"decoding": "greedy (temperature 0)", "max_new_tokens": 256},
            "stage1": {
                "reader": "probe (paper's default)",
                "steering_layers": stage1_layers(short),
                "top_k_layers": synth["top_k_layers"],
                "alphas": synth["alphas"],
                "synthetic": {"n_eval": synth["n_eval"]},
                "confaide": {"set": "CONFAIDE Tier 3", "n_scenarios": conf["n_scenarios"]},
                "privaci": "skipped (not in the common benchmark)",
                "utility": {"alphas": util.get("alpha_values"), "n_alpaca": util["num_instructions"],
                            "n_over_refusal": util["num_appropriate_scenarios"]},
                "summaries": "stage1/",
            },
            "benchmark": {k: bench[k] for k in ("alpha", "top_k_layers", "probe_layer", "probe_layer_rule",
                                                 "steering_layers", "n_pairs", "n_alpaca_calibration",
                                                 "n_alpaca_helpfulness", "n_mmlu")},
            "results": {"benchmark": "metrics.json", "per_item_scores": "scores.json"},
            "status": "preliminary until judge validation passes (NOTES.md)",
        }
        with open(res / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=100)
        print(f"{short}: config.yaml + {len(SUMMARIES)} summaries")


if __name__ == "__main__":
    main()
