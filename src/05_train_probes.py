"""
Step 5 — Train a ridge-regression probe for every (layer, head), rank by
validation F1 (paper: sklearn Ridge(alpha=1.0), train/val split of the
balanced 110-example set, default sklearn settings otherwise).

Output:
  config.PROBES_PATH -> npz with "weights" [n_layers, n_heads, head_dim]
  config.HEAD_RANKING_PATH -> json list of {"layer":, "head":, "f1":, "auroc":}
    sorted best-first
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
import utils


def main():
    data = np.load(config.ACTIVATIONS_PATH, allow_pickle=True)
    acts = data["activations"]
    cd_labels = data["cd_labels"]

    n, n_layers, n_heads, head_dim = acts.shape
    print(f"Loaded activations: N={n}, layers={n_layers}, heads={n_heads}, dim={head_dim}")

    idx_train, idx_val = train_test_split(
        np.arange(n), test_size=0.5, random_state=config.SEED, stratify=cd_labels
    )

    weights = np.zeros((n_layers, n_heads, head_dim), dtype=np.float32)
    ranking = []

    for l in range(n_layers):
        for h in range(n_heads):
            X = acts[:, l, h, :]
            X_train, X_val = X[idx_train], X[idx_val]
            y_train, y_val = cd_labels[idx_train], cd_labels[idx_val]

            probe = Ridge(alpha=1.0)  # default sklearn settings otherwise, per the paper
            probe.fit(X_train, y_train)
            weights[l, h, :] = probe.coef_

            val_pred_continuous = probe.predict(X_val)
            val_pred_binary = (val_pred_continuous >= 0.5).astype(int)

            f1 = f1_score(y_val, val_pred_binary, zero_division=0)
            try:
                auroc = roc_auc_score(y_val, val_pred_continuous)
            except ValueError:
                auroc = float("nan")

            ranking.append({"layer": l, "head": h, "f1": float(f1), "auroc": float(auroc)})

        if l % 8 == 0:
            print(f"Trained probes through layer {l}/{n_layers}")

    ranking.sort(key=lambda r: r["f1"], reverse=True)

    np.savez_compressed(config.PROBES_PATH, weights=weights)
    utils.save_json(ranking, config.HEAD_RANKING_PATH)

    print("\nTop 5 heads by validation F1:")
    for r in ranking[:5]:
        print(f"  layer={r['layer']:2d} head={r['head']:2d}  F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}")
    print("\nSanity check: paper reports a best F1 around 0.83 and AUROC around 0.89 for Llama-2-7B.")
    print(f"Saved probe weights to {config.PROBES_PATH} and ranking to {config.HEAD_RANKING_PATH}")


if __name__ == "__main__":
    main()
