"""Render a dialogue with the model's chat template and build a token-level detect mask.

A dialogue is a list of (role, content, detect) segments. Adjacent same-role segments are
merged into one chat message (this is how Apollo marks only part of a response for detection,
e.g. Instructed-Pairs detects the fact minus its last 5 words).

Model-agnostic re-implementation of Apollo's TokenizedDataset._get_detection_mask: instead of
a per-family regex, we locate each merged message's (stripped) content in the rendered text
sequentially, then map char spans -> tokens with offset_mapping. `pad_left` / `pad_right` extend
the mask over the turn header / end-of-turn tokens, like Apollo's `padding` dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Apollo: llama {"left": 4, "right": 1} for eval sets, {0, 0} for Instructed-Pairs.
# Qwen value is OUR CHOICE: "<|im_start|>assistant\n" = 3 tokens, "<|im_end|>" = 1.
DEFAULT_PADDING = {"llama": (4, 1), "qwen": (3, 1), "gemma": (4, 1), "mistral": (2, 1)}

# Apollo's released Llama-3 transcripts were rendered with this fixed date.
LLAMA_DATE = "26 Jul 2024"


def model_family(model_name: str) -> str:
    n = model_name.lower()
    for fam in ("llama", "qwen", "gemma", "mistral"):
        if fam in n:
            return fam
    return "other"


@dataclass
class Encoded:
    input_ids: torch.Tensor  # [seq]
    detect_mask: torch.Tensor  # [seq] bool
    text: str


def encode(
    tokenizer,
    segments: list[tuple[str, str, bool]],
    family: str,
    pad: tuple[int, int] | None,
    max_length: int | None = None,
) -> Encoded:
    # merge same-role segments, remembering each segment's char span inside the merged message
    merged: list[dict] = []
    for role, content, detect in segments:
        if merged and merged[-1]["role"] == role:
            m = merged[-1]
            m["spans"].append((len(m["content"]), len(m["content"]) + len(content), detect))
            m["content"] += content
        else:
            merged.append({"role": role, "content": content, "spans": [(0, len(content), detect)]})

    fold_system = family in ("gemma", "mistral")
    msgs = [{"role": m["role"], "content": m["content"]} for m in merged]
    if fold_system and msgs and msgs[0]["role"] == "system":
        sys = msgs.pop(0)
        msgs[0]["content"] = sys["content"] + "\n\n" + msgs[0]["content"]
        merged.pop(0)  # system content is never a detect target

    kwargs = {"date_string": LLAMA_DATE} if family == "llama" else {}
    text: str = tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    ids = torch.tensor(enc["input_ids"])
    offsets = enc["offset_mapping"]
    mask = torch.zeros(len(ids), dtype=torch.bool)

    def char_to_tok(c: int) -> int:
        # first token whose span ends after c
        for i, (s, e) in enumerate(offsets):
            if e > c:
                return i
        return len(offsets)

    left, right = pad if pad is not None else DEFAULT_PADDING.get(family, (0, 0))
    cursor = 0
    for m in merged:
        content = m["content"]
        stripped = content.strip()  # templates (e.g. Llama-3) trim message content
        lead = len(content) - len(content.lstrip())
        start = text.find(stripped, cursor) if stripped else cursor
        if start < 0:
            raise ValueError(f"could not locate message content in rendered template: {stripped[:80]!r}")
        for s, e, detect in m["spans"]:
            if not detect:
                continue
            cs = start + max(0, min(len(stripped), s - lead))
            ce = start + max(0, min(len(stripped), e - lead))
            if ce <= cs:
                continue
            ts, te = char_to_tok(cs), char_to_tok(ce)
            mask[max(0, ts - left) : min(len(ids), te + right)] = True
        cursor = start + len(stripped)

    return Encoded(ids, mask, text)
