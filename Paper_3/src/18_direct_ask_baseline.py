"""
Step 18 — Phase D(a): the "direct ask" black-box baseline every white-box
method must beat, per the protocol. For P3 this IS the Phase A baseline
response (no probe, no steering) — no new generation needed.

Reports: baseline disclosure rate (direct-ask), vs. the white-box method's
single-config and best-of-configs disclosure rates from Phase A/C, so the
comparison is explicit in one place.

Output: config.DATA_DIR/baseline_comparison.json
"""
import config
import utils


def main():
    baseline_labels = utils.load_json(config.BASELINE_LABELS_PATH)
    phase_a_metrics = utils.load_json(config.METRICS_PATH)

    n_total = len(baseline_labels)
    n_disclosed = sum(1 for v in baseline_labels.values() if v["binary"] == 0)
    direct_ask_rate = n_disclosed / n_total

    per_config_rates = phase_a_metrics.get("per_config_disclosure_rate", {})
    avg_single_config = sum(per_config_rates.values()) / len(per_config_rates) if per_config_rates else None
    best_of_configs = phase_a_metrics.get("jailbreak_rate_best_of_configs")

    comparison = {
        "baseline_direct_ask_disclosure_rate": direct_ask_rate,
        "n_queried": n_total,
        "white_box_avg_single_config_disclosure_rate": avg_single_config,
        "white_box_best_of_configs_jailbreak_rate": best_of_configs,
        "white_box_beats_baseline_single_config": (
            avg_single_config > direct_ask_rate if avg_single_config is not None else None
        ),
        "white_box_beats_baseline_best_of_configs": (
            best_of_configs > direct_ask_rate if best_of_configs is not None else None
        ),
    }
    utils.save_json(comparison, f"{config.DATA_DIR}/baseline_comparison.json")

    print("===== BLACK-BOX BASELINE (a): DIRECT ASK =====")
    print(f"Direct-ask disclosure rate: {direct_ask_rate:.3f} ({n_disclosed}/{n_total})")
    print(f"\nWhite-box (probe+steering) avg single-config rate: {avg_single_config}")
    print(f"White-box best-of-configs jailbreak rate: {best_of_configs}")
    print(f"\nWhite-box beats baseline (single-config): {comparison['white_box_beats_baseline_single_config']}")
    print(f"White-box beats baseline (best-of-configs): {comparison['white_box_beats_baseline_best_of_configs']}")
    print(f"\nSaved to {config.DATA_DIR}/baseline_comparison.json")


if __name__ == "__main__":
    main()
