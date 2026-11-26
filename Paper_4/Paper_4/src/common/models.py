"""Model loading. bf16 by default (§3). On a <16 GB GPU, bf16 still works via accelerate CPU
offload (exact activations, slower). 8bit/4bit are available but change activations: log it."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_name: str, precision: str = "bf16", gpu_mem_gib: float | None = None,
         revision: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    kw: dict = {"revision": revision, "device_map": "auto"}
    if precision == "bf16":
        kw["dtype"] = torch.bfloat16
    elif precision in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=precision == "8bit",
            load_in_4bit=precision == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(precision)
    if gpu_mem_gib is not None and torch.cuda.is_available():
        kw["max_memory"] = {0: f"{gpu_mem_gib}GiB", "cpu": "64GiB"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
    return model, tok


def resolved_revision(model) -> str | None:
    return getattr(model.config, "_commit_hash", None)
