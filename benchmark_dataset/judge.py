"""
Shared Stage 2 judge wrapper. Every paper's benchmark-port code should
import and call `call_judge(prompt_text)` from THIS module, instead of each
person writing their own OpenAI/local-model calling code. This is what
actually enforces "use one fixed judge for every method" — a shared config
file is necessary but not sufficient if everyone's wrapper code differs
subtly (different temperature, different retry logic, different truncation).

Usage (identical for every paper):
    from judge import call_judge
    raw_text = call_judge("your fully-formatted judge prompt here")
    label = your_own_parsing_function(raw_text)   # parsing IS paper-specific —
                                                     # P3 parses Disclosed/Refused/etc,
                                                     # P1 parses its own leakage labels, etc.

What this module does NOT decide for you: the judge PROMPT content itself.
Each paper asks the judge a different question (P3: did this response disclose
the attribute? P1: did this response leak the target string? etc.) — that
prompt text stays paper-specific. What this module holds constant is the
model, backend, and temperature used to answer whatever prompt you send it.

Reads its settings from shared_judge_config.json, which must sit in the same
directory as this file (or pass config_path explicitly). Do not hardcode a
different model/temperature in your own paper's code — change the shared
config file instead, so the change applies to everyone at once.
"""
import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_judge_config.json")

_config = None
_local_model = None
_local_tokenizer = None


def _load_config(config_path=None):
    global _config
    path = config_path or _CONFIG_PATH
    with open(path) as f:
        _config = json.load(f)
    return _config


def _get_config():
    if _config is None:
        _load_config()
    return _config


def _call_openai(prompt_text, model, temperature):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return r.choices[0].message.content


def _load_local_model(model_name, quantization):
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        import torch
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
        )
    else:
        _local_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    _local_model.eval()
    return _local_model, _local_tokenizer


def _call_local(prompt_text, model_name, quantization, temperature):
    import torch
    model, tok = _load_local_model(model_name, quantization)
    messages = [{"role": "user", "content": prompt_text}]
    templated = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(templated, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=1 if not do_sample else None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen_only = out[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_only[0], skip_special_tokens=True).strip()


def call_judge(prompt_text, config_path=None):
    """
    The one function every paper's Stage 2 code should call for judge
    decisions. Returns the judge's raw text response — parsing it into your
    paper's specific labels is your own code's job.
    """
    cfg = _load_config(config_path) if config_path else _get_config()

    if cfg["backend"] == "openai":
        return _call_openai(prompt_text, cfg["model"], cfg["temperature"])
    elif cfg["backend"] == "local":
        return _call_local(prompt_text, cfg["local_model"], cfg["local_quantization"], cfg["temperature"])
    else:
        raise ValueError(f"Unknown judge backend in shared_judge_config.json: {cfg['backend']}")


def get_judge_identity():
    """Returns a short string identifying the current judge, for logging into
    each paper's config.yaml — e.g. 'openai:gpt-4o-mini:temp0'."""
    cfg = _get_config()
    if cfg["backend"] == "openai":
        return f"openai:{cfg['model']}:temp{cfg['temperature']}"
    else:
        return f"local:{cfg['local_model']}:{cfg['local_quantization']}:temp{cfg['temperature']}"


if __name__ == "__main__":
    cfg = _get_config()
    print("Current shared judge configuration:")
    print(json.dumps(cfg, indent=2))
    print(f"\nIdentity string for config.yaml logging: {get_judge_identity()}")
