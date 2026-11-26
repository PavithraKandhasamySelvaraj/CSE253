"""Residual-stream activations at detect-mask tokens, via forward hooks (no LM head, no full
hidden_states tuple -> fits long insider-trading transcripts on a 12 GB GPU).

Layer convention (matches Apollo's `hidden_states[l]` indexing, e.g. "layer 22" for 70B):
  l = 0      -> embedding output
  l = 1..L   -> output of decoder block l-1 (= resid_post of block l-1, pre final norm)
HF's hidden_states[L] has the final norm applied; ours does not. Only matters at l = L.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from src.common.chat import encode, model_family


class _Stop(Exception):
    pass


@torch.no_grad()
def iter_acts(model, tok, items, layers: list[int], pad=None, max_length: int | None = None,
              desc: str = "") -> Iterator[tuple[int, torch.Tensor | None]]:
    """Yield (item_idx, acts[n_detect, len(layers), d] float32 on cpu) per item.
    Items with an empty detect mask yield None."""
    from tqdm import tqdm

    fam = model_family(model.config._name_or_path)
    base = model.model
    blocks = base.layers
    max_l = max(layers)
    cache: dict[int, torch.Tensor] = {}
    state: dict = {}

    def grab(l):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[l] = h[0, state["mask"].to(h.device)].float().cpu()
            if l == max_l:
                raise _Stop  # skip the remaining blocks
        return hook

    hooks = []
    if 0 in layers:
        hooks.append(base.embed_tokens.register_forward_hook(grab(0)))
    for l in layers:
        if l >= 1:
            hooks.append(blocks[l - 1].register_forward_hook(grab(l)))
    dev = next(base.parameters()).device
    try:
        for i, it in enumerate(tqdm(items, desc=desc)):
            enc = encode(tok, it.segments, fam, pad, max_length)
            if not enc.detect_mask.any():
                yield i, None
                continue
            state["mask"] = enc.detect_mask
            cache.clear()
            try:
                base(input_ids=enc.input_ids[None].to(dev), use_cache=False)
            except _Stop:
                pass
            yield i, torch.stack([cache[l] for l in layers], dim=1)
    finally:
        for h in hooks:
            h.remove()
