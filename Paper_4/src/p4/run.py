"""P4 (Goldowsky-Dill et al. 2025) deception probe — entrypoint.

Stage 1 (faithful, adapted to 8B; paper only has 70B):
  1a  validate   : recompute the released 70B results table with our metric code (no GPU)
  1b  train      : Instructed-Pairs activations at every layer -> one LR probe per layer
  1c  eval       : score Apollo's released 70B transcripts (OFF-policy for the 8B model) with
                   every layer's probe, threshold on 1,000 released Alpaca responses
  1d  (todo)     : regenerate on-policy 8B rollouts + local-judge labels, re-run 1c

Usage:
  python -m src.p4.run --stage 1 --model meta-llama/Llama-3.1-8B-Instruct
  python -m src.p4.run --step validate
  python -m src.p4.run --step train --model Qwen/Qwen2.5-0.5B-Instruct --limit 40   # smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.metrics import detection_metrics, reduce_mean
from src.p4 import apollo
from src.p4.acts import iter_acts
from src.p4.probe import fit_layer, score_tokens

ROOT = apollo.ROOT


def slug(model: str) -> str:
    return model.split("/")[-1]


def paths(model: str):
    res = ROOT / "results" / "p4" / slug(model)
    out = ROOT / "outputs" / "p4" / slug(model)
    res.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return res, out


def all_layers(model) -> list[int]:
    return list(range(model.config.num_hidden_layers + 1))


# ---------------------------------------------------------------- train
def train(model, tok, args) -> dict:
    res, out = paths(args.model)
    layers = args.layers or all_layers(model)
    tr, va = apollo.instructed_pairs(val_fraction=0.2)
    if args.limit:
        tr, va = tr[: args.limit], va[: max(2, args.limit // 4)]
    pad = (0, 0)  # Apollo RepeHonestDishonestFacts.padding = 0/0

    X, y = [], []
    for i, a in iter_acts(model, tok, tr, layers, pad=pad, desc="instructed-pairs train"):
        if a is None:
            continue
        X.append(a.half())
        y.append(np.full(len(a), 1 if tr[i].label == "deceptive" else 0))
    X = torch.cat(X)  # [n_tok, n_layers, d]
    y = np.concatenate(y)
    print(f"train tokens: {len(y)} ({y.mean():.2%} deceptive), layers: {len(layers)}")

    probes, conv = {}, {}
    for j, l in enumerate(layers):
        p = fit_layer(X[:, j].float().numpy(), y, reg_coeff=args.reg_coeff)
        conv[l] = p.pop("converged")
        probes[l] = p
    torch.save({"layers": layers, "probes": probes, "reg_coeff": args.reg_coeff},
               res / "probes.pt")

    # Instructed-Pairs validation (repe_honesty__plain_val row of Apollo's table)
    val = score_items(model, tok, va, probes, layers, pad=pad, desc="instructed-pairs val")
    torch.save({"labels": [v.label for v in va], "scores": val}, out / "repe_honesty__plain_val.pt")
    return {"layers": layers, "converged": conv, "n_train_tokens": int(len(y))}


def score_items(model, tok, items, probes, layers, pad=None, desc=""):
    """-> {layer: [per-token score arrays]} (np.nan-only array for empty masks)."""
    out = {l: [np.array([np.nan])] * len(items) for l in layers}
    for l in layers:
        out[l] = list(out[l])
    for i, a in iter_acts(model, tok, items, layers, pad=pad, max_length=args_max_len, desc=desc):
        if a is None:
            continue
        for j, l in enumerate(layers):
            out[l][i] = score_tokens(probes[l], a[:, j])
    return out


args_max_len: int | None = None


# ---------------------------------------------------------------- eval
def evaluate(model, tok, args) -> dict:
    res, out = paths(args.model)
    ck = torch.load(res / "probes.pt", weights_only=False)
    probes, layers = ck["probes"], ck["layers"]

    def cached(name: str, items) -> dict:
        f = out / f"{name}__{args.policy}__n{len(items)}.pt"  # n in key: --limit runs never collide
        if f.exists() and not args.overwrite:
            return torch.load(f, weights_only=False)
        d = {"labels": [x.label for x in items],
             "scores": score_items(model, tok, items, probes, layers, desc=name)}
        torch.save(d, f)
        return d

    ctrl_items = apollo.load(apollo.CONTROL_SET, model=args.policy, limit=args.control_n)
    ctrl = cached(apollo.CONTROL_SET, ctrl_items)
    sets = {"repe_honesty__plain_val": torch.load(out / "repe_honesty__plain_val.pt",
                                                  weights_only=False)}
    for name in args.datasets:
        sets[name] = cached(name, apollo.load(name, model=args.policy, limit=args.limit))

    by_layer = {}
    for l in layers:
        c = reduce_mean(ctrl["scores"][l])
        by_layer[l] = {n: detection_metrics(reduce_mean(d["scores"][l]), d["labels"], c)
                       for n, d in sets.items()}
    return by_layer


def pick_layer(by_layer: dict, on: str = "roleplaying__plain") -> int:
    """Apollo picked the layer by sweeping on Roleplaying (§4.4). We pick max AUROC there,
    tie-broken by recall@1%FPR. OUR CHOICE of criterion; logged in NOTES.md."""
    return max(by_layer, key=lambda l: (by_layer[l].get(on, {}).get("auroc", -1),
                                        by_layer[l].get(on, {}).get("recall 1.0%", -1)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--step", choices=["validate", "train", "eval"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "8bit", "4bit"])
    ap.add_argument("--gpu-mem-gib", type=float, default=None,
                    help="cap GPU weights; rest offloaded to CPU (e.g. 9 on a 12 GB card)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--reg-coeff", type=float, default=10.0)
    ap.add_argument("--policy", default="llama-70b-3.3",
                    help="whose transcripts to score; llama-70b-3.3 = Apollo's released ones")
    ap.add_argument("--datasets", nargs="*", default=apollo.EVAL_SETS)
    ap.add_argument("--control-n", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test cap per dataset")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    global args_max_len
    args_max_len = args.max_len

    if args.stage == 2:
        raise SystemExit("stage 2 (benchmark port) needs the shared counterfactual items from "
                         "P1/P2 first; not implemented yet")
    steps = ["validate", "train", "eval"] if args.stage == 1 else [args.step]

    res, _ = paths(args.model)
    metrics: dict = {}
    mf = res / "metrics.json"
    if mf.exists():
        metrics = json.loads(mf.read_text())
    t0 = time.time()

    if "validate" in steps:
        from src.p4.validate_released import main as vmain

        table, maxdiff = vmain()
        print(table.round(4), f"\nmax diff vs released table: {maxdiff:.2e}")
        metrics["released_70b_recomputed"] = {"max_abs_diff": maxdiff,
                                              "table": table.to_dict(orient="index")}

    model = tok = None
    if "train" in steps or "eval" in steps:
        from src.common.models import load, resolved_revision

        model, tok = load(args.model, args.precision, args.gpu_mem_gib)
        cfg = {"paper": "P4", "model": args.model, "revision": resolved_revision(model),
               "precision": args.precision, "gpu_mem_gib": args.gpu_mem_gib,
               "reg_coeff": args.reg_coeff, "C": 1 / args.reg_coeff, "policy": args.policy,
               "control_n": args.control_n, "datasets": args.datasets, "seed_lr": 42,
               "limit": args.limit, "judge": None,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "host": platform.node()}
        (res / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if "train" in steps:
        metrics["train"] = train(model, tok, args)
    if "eval" in steps:
        by_layer = evaluate(model, tok, args)
        best = pick_layer(by_layer)
        metrics["eval_policy"] = args.policy
        metrics["by_layer"] = {str(k): v for k, v in by_layer.items()}
        metrics["selected_layer"] = best
        metrics["selected"] = by_layer[best]
        print(f"\nselected layer {best}:")
        for n, m in by_layer[best].items():
            print(f"  {n:40s} AUROC {m['auroc']:.3f}  recall@1% {m['recall 1.0%']:.3f}  "
                  f"(H/A/D {m['n_honest']}/{m['n_ambiguous']}/{m['n_deceptive']})")

    metrics["wall_seconds_last_run"] = round(time.time() - t0, 1)
    mf.write_text(json.dumps(metrics, indent=1, default=float))
    print(f"wrote {mf}")


if __name__ == "__main__":
    main()
