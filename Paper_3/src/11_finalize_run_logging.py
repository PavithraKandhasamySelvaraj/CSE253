"""
Step 11 — Close the remaining shared-protocol (§3) logging gaps:
  - Write a config.yaml recording model + revision (commit hash), judge +
    revision, layers/hook point, the k/alpha grid actually used, probe
    training set size, seed, test pool size, and judge validation results
    if available.
  - Copy the run's key outputs into results/<paper_id>/<model>/, the path
    structure the shared protocol specifies, instead of leaving everything
    under p3_data/.

"""
import os
import shutil
import datetime

import config
import utils

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    print("PyYAML not installed — will install it now.")
    os.system("pip install -q pyyaml")
    import yaml
    HAVE_YAML = True


def get_model_revision(repo_id):
    """Best-effort fetch of the exact commit hash in use. Network call."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id)
        return info.sha
    except Exception as e:
        return f"UNKNOWN (lookup failed: {e})"


def main():
    model_revision = get_model_revision(config.TARGET_MODEL)
    judge_revision = get_model_revision(config.JUDGE_MODEL) if config.JUDGE_BACKEND == "local" else "n/a (openai)"

    try:
        chosen = utils.load_json(f"{config.DATA_DIR}/chosen_configs.json")
    except FileNotFoundError:
        chosen = None

    try:
        metrics = utils.load_json(config.METRICS_PATH)
    except FileNotFoundError:
        metrics = None

    probe_train_size = None
    try:
        import numpy as np
        acts = np.load(config.ACTIVATIONS_PATH, allow_pickle=True)
        probe_train_size = int(acts["activations"].shape[0])
    except FileNotFoundError:
        pass

    try:
        judge_validation = utils.load_json(f"{config.DATA_DIR}/judge_validation_results.json")
        judge_validation_summary = {
            "n_validated": judge_validation["n_validated"],
            "overall_accuracy": judge_validation["overall_accuracy"],
            "cohen_kappa": judge_validation["cohen_kappa"],
        }
    except FileNotFoundError:
        judge_validation_summary = "NOT YET RUN — see 09_build_judge_validation_sample.py / 10_judge_validation_agreement.py"

    run_config = {
        "paper_id": "P3",
        "paper_title": "PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage (Nakka et al., arXiv 2507.02332)",
        "run_date": datetime.date.today().isoformat(),
        "stage": "stage_1_faithful_reproduction",
        "models": {
            "target": {
                "name": config.TARGET_MODEL,
                "revision": model_revision,
                "dtype": config.TARGET_DTYPE,
            },
            "judge": {
                "name": config.JUDGE_MODEL if config.JUDGE_BACKEND == "local" else "gpt-4o-mini",
                "revision": judge_revision,
                "backend": config.JUDGE_BACKEND,
            },
        },
        "attribute_probed": "sexual orientation",
        "activation_hook": {
            "location": "input to self_attn.o_proj",
            "token_position": "last prompt token (prefill pass only, never applied during decoding)",
            "n_layers": 32,
            "n_heads": 32,
            "head_dim": 128,
        },
        "probe": {
            "method": "sklearn.linear_model.Ridge",
            "alpha": 1.0,
            "balanced_train_val_size": probe_train_size,
            "target_size_in_paper": 110,
        },
        "steering": {
            "k_grid_used": config.K_GRID,
            "alpha_grid_used": config.ALPHA_GRID,
            "paper_k_grid": [16, 32, 48, 64, 96],
            "paper_alpha_range": "-80 to 30, step 10",
            "tune_subset_size": config.TUNE_SUBSET_SIZE,
            "chosen_configs_for_full_run": chosen,
        },
        "seed": config.SEED,
        "results_summary": metrics,
        "judge_validation": judge_validation_summary,
        "known_deviations_from_paper": [
            "Name list generated via the target model with a neutral (non-orientation-targeted) "
            "prompt, 677 names vs. paper's 863 via GPT-4o-mini with orientation-targeted prompt.",
            "Judge is local gemma-2-9b-it (4-bit) rather than GPT-4o-mini for the main privacy "
            "and factuality judgments.",
            "Probe balanced training set size adapted to available data (see "
            "balanced_train_val_size above) rather than the paper's fixed 110.",
            "Steering grid reduced to a subset of (k, alpha) combinations, tuned on a subset "
            "of test subjects before the top configs were run on the full test pool.",
            "Target model run in fp16, not bf16 (free-tier GPU hardware does not support bf16).",
            "See NOTES.md for full explanation and the population-composition finding that "
            "explains most of the jailbreak-rate gap vs. the paper.",
        ],
    }

    config_yaml_path = f"{config.DATA_DIR}/config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {config_yaml_path}")

    # ---- reorganize into results/<paper_id>/<model>/ ----
    model_dir_name = config.TARGET_MODEL.split("/")[-1].lower()
    results_dir = f"/kaggle/working/results/P3/{model_dir_name}"
    os.makedirs(results_dir, exist_ok=True)

    files_to_copy = {
        config.METRICS_PATH: "metrics.json",
        config_yaml_path: "config.yaml",
    }
    # NOTES.md may live alongside the scripts, not in DATA_DIR — check both locations
    for notes_candidate in ["/kaggle/working/NOTES.md", f"{config.DATA_DIR}/NOTES.md"]:
        if os.path.exists(notes_candidate):
            files_to_copy[notes_candidate] = "NOTES.md"
            break

    if os.path.exists(f"{config.DATA_DIR}/judge_validation_results.json"):
        files_to_copy[f"{config.DATA_DIR}/judge_validation_results.json"] = "judge_validation_results.json"
    if os.path.exists(f"{config.DATA_DIR}/judge_prompt_log.txt"):
        files_to_copy[f"{config.DATA_DIR}/judge_prompt_log.txt"] = "judge_prompt_log.txt"

    for src, dst_name in files_to_copy.items():
        if os.path.exists(src):
            shutil.copy(src, os.path.join(results_dir, dst_name))
            print(f"  copied {src} -> {results_dir}/{dst_name}")
        else:
            print(f"  SKIPPED (not found): {src}")

    print(f"\nFinal results directory: {results_dir}")
    print("Contents:")
    for f in sorted(os.listdir(results_dir)):
        print(f"  {f}")

    if judge_validation_summary == "NOT YET RUN — see 09_build_judge_validation_sample.py / 10_judge_validation_agreement.py":
        print("\nREMINDER: judge validation has not been run yet. Run "
              "09_build_judge_validation_sample.py and "
              "10_judge_validation_agreement.py, then re-run this script to "
              "include the validation numbers in config.yaml.")


if __name__ == "__main__":
    main()
