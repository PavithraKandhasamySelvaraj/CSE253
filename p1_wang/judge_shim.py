"""
Routes the CI-Steering code's OpenAI judge calls to the team's shared judge.

The paper's code (third_party/CI-Steering) calls
`OpenAI(...).chat.completions.create(...)` with GPT-4o-mini in two places:
src/evaluation/ci_eval.py (leakage judge) and src/utility_evaluation.py
(quality scorer). Instead of editing that code, `install()` replaces the
`openai.OpenAI` class with a stand-in whose `create` forwards the prompt to
`benchmark_dataset/judge.py:call_judge`, so every judge call goes through the
one shared judge configuration (benchmark_dataset/shared_judge_config.json).

The paper's prompts are kept word for word. Two unavoidable differences:
- call_judge takes a single user message (and Gemma has no system role), so
  the system prompt is prepended to the user message.
- There is no JSON mode; the paper's own fallback parser extracts the JSON.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = REPO_ROOT / "benchmark_dataset"


class _Completions:
    def create(self, model=None, messages=None, **_ignored):
        from judge import call_judge

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        prompt = f"{system}\n\n{user}" if system else user
        text = call_judge(prompt)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class SharedJudgeClient:
    """Stand-in for openai.OpenAI that only supports chat.completions.create."""

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_Completions())


def install():
    sys.path.insert(0, str(SHARED_DIR))
    import openai
    import judge

    # Test-only override (e.g. a smoke test with an ungated model). Real runs
    # must leave JUDGE_CONFIG unset so the shared config is used.
    if os.environ.get("JUDGE_CONFIG"):
        judge._load_config(os.environ["JUDGE_CONFIG"])
        print(f"[judge_shim] WARNING: using test judge config {os.environ['JUDGE_CONFIG']}")

    openai.OpenAI = SharedJudgeClient
    # The paper's scripts refuse to start without a key; no key is ever used.
    os.environ.setdefault("OPENAI_API_KEY", "unused-shared-local-judge")


def judge_identity():
    sys.path.insert(0, str(SHARED_DIR))
    from judge import get_judge_identity

    return get_judge_identity()
