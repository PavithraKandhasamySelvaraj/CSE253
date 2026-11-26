"""Adapter over the Apollo deception-detection repo's dataset loaders.

We reuse their loaders verbatim (exact prompts, labels, detect flags, trim_reasoning, shuffle
seed) and only re-implement tokenization / activations / probes / metrics model-agnostically.
Repo must be cloned at external/deception-detection (see README).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APOLLO = ROOT / "external" / "deception-detection"
if str(APOLLO) not in sys.path:
    sys.path.insert(0, str(APOLLO))


@dataclass
class Item:
    segments: list[tuple[str, str, bool]]  # (role, content, detect)
    label: str  # honest | deceptive | ambiguous | skip
    meta: dict = field(default_factory=dict)


def _repo():
    from deception_detection.repository import DatasetRepository  # noqa: E402

    return DatasetRepository()


def load(partial_id: str, model: str = "prewritten", trim_reasoning: bool = True,
         limit: int | None = None) -> list[Item]:
    """Mirrors Experiment.get_dataset: repe is not shuffled, everything else is (seed 42)."""
    ds = _repo().get(
        partial_id,
        model=model,
        trim_reasoning=trim_reasoning,
        shuffle_upon_init=not partial_id.startswith("repe"),
    )
    items = []
    meta = getattr(ds, "metadata", None)
    for i, (dlg, lab) in enumerate(zip(ds.dialogues, ds.labels, strict=True)):
        m = {k: v[i] for k, v in meta.items()} if meta else {}
        items.append(Item([(x.role, x.content, bool(x.detect)) for x in dlg], lab.value, m))
    return items[:limit] if limit else items


def instructed_pairs(val_fraction: float = 0.2) -> tuple[list[Item], list[Item]]:
    """Apollo cfg: train_data=repe_honesty__plain, val = last int(N*0.2) dialogues (pairs are
    interleaved honest/deceptive, so the tail split keeps pairs together)."""
    items = load("repe_honesty__plain")
    n_val = int(len(items) * val_fraction)
    return items[:-n_val], items[-n_val:]


# ids as in Apollo's repe.yaml / instructed_pairs cfg
EVAL_SETS = [
    "roleplaying__plain",
    "insider_trading__upscale",
    "insider_trading_doubledown__upscale",
    "sandbagging_v2__wmdp_mmlu",
]
CONTROL_SET = "alpaca__plain"
