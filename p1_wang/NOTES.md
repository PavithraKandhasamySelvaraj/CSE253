# P1 — Wang, Xiong, Shu (arXiv 2604.00209): reproduction notes

Paper: *Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in LLM Representations.*
Protocol: `agent.md` §3 (shared benchmark) and §4.1 (P1).

## How to run

```bash
cd p1_wang
./setup.sh                                   # once: paper code, datasets, .venv, stimuli
./run_stage1.sh meta-llama/Llama-3.1-8B-Instruct            # all steps
./run_stage1.sh Qwen/Qwen2.5-7B-Instruct extract read ci    # or chosen steps
```

| Step | What it runs | Output |
|---|---|---|
| `extract` | paper `extract_activations.py` (last token, all layers) | `outputs/activations/<model>` |
| `read` | paper `read_representations.py` (PCA + probes, C=1.0, 5-fold CV) | `outputs/reading/<model>` |
| `ci` | paper `ci_decomposition.py` (one direction per CI parameter) | `outputs/ci_decomposition/<model>` |
| `synthetic` | paper `ci_steering_evaluation.py`, α ∈ {0.5, 1, 2, 4} | `outputs/ci_steering_comparison/<model>` |
| `confaide` | paper `confaide_ci_steering.py` (CONFAIDE Tier 3) | `outputs/confaide_ci/<model>` |
| `privaci` | paper `privaci_ci_steering.py` | `outputs/privaci_ci/<model>` |
| `utility` | paper `utility_evaluation.py` (Alpaca quality, over-refusal) | `outputs/utility_eval/<model>` |
| `benchmark` | our `benchmark.py` (stage 2, shared protocol) | `results/<model>/metrics.json` |

## What is ours vs. the paper's

- **Paper code:** `third_party/CI-Steering` at commit `5c5edc3` (MIT license), cloned by `setup.sh`. It is unchanged except for the one-line syntax fix in `patches/`.
- **`compat.py`:** an SVD fallback for running off CUDA (see "Fixes" below).
- **`judge_shim.py` / `run_paper.py`:** run the paper's scripts as-is, but replace `openai.OpenAI` so every judge call goes to the team's `benchmark_dataset/judge.py`.
- **`counterfactuals.py` / `benchmark.py`:** the stage-2 benchmark port (agent.md §3).
- **`benchmark_dataset/` changes (shared, need team agreement):**
  - `shared_judge_config.json`: backend switched to local `google/gemma-2-9b-it`, `bf16`.
  - `judge.py`: new `bf16` option that loads the local judge in bfloat16. Without an explicit dtype, Gemma-2 would load in float32 (double the memory, slower).

## Differences from the paper

| # | Paper | Here | Why |
|---|---|---|---|
| 1 | Judge: GPT-4o-mini, JSON mode, system + user message | Local `gemma-2-9b-it` (bf16) through shared `judge.py`; system prompt prepended to the user message; no JSON mode (paper's own fallback parser extracts the JSON); max 300 new tokens | Team decision: no paid API. Gemma has no system role. |
| 2 | CUDA GPUs | Apple M4 Max (MPS backend), float16 as in the paper's config | Available hardware. Small numeric differences vs. CUDA are expected. |
| 3 | transformers version not pinned (≥4.40) | transformers 5.24.0 (see `requirements.txt`) | Latest at setup time. |
| 4 | Two-pass workflow (agent.md §3: generate, then judge) | Paper's scripts generate and judge in one process; both models stay loaded (~16 GB + ~18 GB) | Keeps the paper's scripts unchanged. Judge results are cached by prompt hash, so re-judging is free. |

## Fixes needed to run the released code

- **Upstream syntax error.** `src/control/rep_tuning.py:167` is mis-indented in commit `5c5edc3`. Every script that imports `src.control` (all the steering and evaluation scripts) fails with `IndentationError` as published. The one-line fix is `patches/ci-steering-fixes.patch`, applied by `setup.sh`.
- **SVD on CPU.** `ci_decomposition.py` runs PCA "on cuda if available, else cpu". On CPU, LAPACK's float32 SVD fails to converge on some activation matrices in the permutation test (Qwen-2.5-7B), which crashes the script before the CI directions are saved. `compat.py` retries only failed SVD calls in float64. Successful calls are unchanged.

## Findings so far (Qwen-2.5-7B, stage 1 steps without a judge)

- **The linear probe saturates.** Concept-probe CV and test accuracy are 1.00 at *every* layer, including layer 0 (embeddings). The template pairs are separable from their surface words alone, so probe accuracy says little about a learned norm and cannot choose a layer. The paper's "probe-weighted" steering picks layers by probe CV accuracy, which on these data is a tie. PCA is not saturated: held-out AUROC 0.95 at layers 16–17, around 0.70 at layers 0–4.
- **The paper's "cross-task" check is degenerate on its own data.** The behavioral activations come from `function_stimuli.json`, where all 200 items are "protect". So cross-task AUROC is NaN and accuracy is trivially 1.0 for any probe that predicts "inappropriate". The privacy-awareness-gap numbers computed from it should not be reported.
- **CI directions are highly entangled.** At the analysis layer, |cos| between parameter directions is 0.96–0.99. That is lower than the permutation null (0.999), but far from "independent axes".

- **CI steering raises leakage on Qwen (preliminary, judge not yet validated).** CONFAIDE leakage 63.0% unsteered → 70.4% at α = 1 → 85.6% at α = 4 (CI-decomposed, all three parameters), while standard steering stays at 63–65%. Utility is unaffected (quality 4.83–4.88/5, over-refusal 0% at every α).
- **Why CI directions are not oriented toward "protect".** In `ci_decomposition.py` each CI direction is PC1 of the activations while one parameter varies, with its sign fixed so the *raw mean projection is positive*. Nothing ties that sign to privacy. Standard steering, by contrast, explicitly adds `-probe_direction` (toward "inappropriate"). Measured on Qwen, cos(CI direction, protect direction) at the five steering layers: layer 4 ≈ −0.10, 8 ≈ +0.12, 12 ≈ −0.12, 17 ≈ +0.48, 18 ≈ +0.60 (same for all three parameters). So the sign is not simply flipped: at the deepest layers the CI directions point partly toward "protect", but at layers 4 and 12 they point away, and much of each direction is orthogonal to privacy. Both steerers also rescale the hidden state back to its original norm, which keeps unit-vector steering weak.

## Choices where the paper is silent (settled by its released code)

- **CI steering layers:** top-k (k=5) by summed CI-direction magnitude (`from_ci_directions_dir`), not by probe accuracy. The paper text says both. The code uses magnitude.
- **Steering token positions:** every position, during prefill and every generated token (forward hook on each decoder layer output).
- **Probe:** `LogisticRegression(C=1.0, lbfgs, max_iter=1000)` on standardized activations; direction = coef / scale, unit-normalized; best layer by 5-fold CV accuracy.
- **Behavioral set:** the eval scripts regenerate the balanced 100/100 set with seed 42 (`generate_function_stimuli_balanced`); `data/stimuli/function_stimuli.json` (200 "protect" items) is used only for activation extraction.
- **Utility rubric:** the paper's `QUALITY_JUDGE_PROMPT` in `utility_evaluation.py` (helpfulness, coherence, relevance, 1–5).

## Stage-2 design (benchmark.py)

- **Counterfactual pairs (100):** each "protect" scenario from the paper's balanced set, plus a copy where only the transmission principle changes. The secrecy sentence is replaced with "They told you they are comfortable with you sharing this with anyone who asks, including {questioner}", and secrecy words ("in confidence", "confided in you", …) are removed from the first sentence. Words that are part of the private detail itself (e.g. "a secret trust fund") stay, because they are identical in both variants.
- **Decision:** the paper's leakage judge field `leaked` (= the response reveals the detail). On consent variants it means "shared". The judge prompt says the detail "should not" be revealed for every item; we use only the descriptive `leaked` field.
- **Dependent pair:** `leaked(private) != leaked(consent)`, unsteered model.
- **Detector layer:** the layer with the best PCA held-out AUROC on the concept test split (Qwen: 17). The probe's own CV accuracy cannot be used, because it is 1.0 everywhere (see Findings).
- **Detector score:** s(x) = −⟨h_L(x), v_L⟩. This is the inappropriateness projection on the concept probe's direction at layer L, last token. The paper's `ProbeReader.load` keeps directions only, so we score by projection; this is monotone in the probe logit at a fixed layer, so AUROC and thresholds are unaffected.
- **1% FPR threshold:** 99th percentile of s on the shared `alpaca_calibration_1000.json` prompts, formatted with the model's chat template and the system prompt "You are a helpful assistant."
- **Baseline (a), ask the model:** inside the scenario's own chat, ask whether it would be appropriate to tell the questioner the detail; score = P("No") from the next-token logits. It is calibrated on Alpaca with the same yes/no question about the instruction.
- **Baselines (b) and (c):** not applicable to P1. There is no chain-of-thought, and the external monitor *is* the judge that produces the labels (circular).
- **Utility:** shared `mmlu_utility_1000.json`, zero-shot, letter by next-token logit. Alpaca helpfulness on the first 200 shared Alpaca prompts (the paper used 200) with the paper's rubric. Both unsteered vs. CI-steered at α = 1.0.

## Judge validation (required before reporting, agent.md §3)

In progress: both label sets (Qwen, Llama) are built, the labels are not in yet.

- **Set** (`judge_validation/build_label_set.py`, seed 253): 100 Qwen responses from the shared benchmark, 25 per cell (private / consent × unsteered / CI α = 1), balanced on Gemma's verdict where possible (56 leaked, 44 not), 95 distinct scenarios. Only these responses are saved in full: the paper's scripts keep just the first 200 characters of each synthetic and CONFAIDE response, so those cannot be hand-labeled.
- **Labelers**: two teammates, independently, `labeler_A.xlsx` and `labeler_B.xlsx` (Gemma's verdicts hidden in `_key/judge_key.csv`). Two questions per item: leaked? refused?
- **Scoring** (`judge_validation/score_agreement.py`): human–human κ; Gemma vs. each labeler and vs. the consensus; items 1–50 (dev) may be used to fix the prompt, items 51–100 (test) give the reported κ.
- **Llama set** (primary model; `judge_validation/llama/`, same script with `--model Llama-3.1-8B-Instruct --out-dir judge_validation/llama`, seed 253): 100 Llama benchmark responses, 25 per cell, 61 leaked / 39 not by Gemma, 92 distinct scenarios. The unsteered consent cell has only leaked responses (Llama shared in all 100), so it is all "leaked". About 8 items are degenerate steered replies; the Llama sheets add an instruction for garbled replies, which the Qwen sheets (built earlier) lack. Score with `score_agreement.py judge_validation/llama`.
- **Deviation (Llama set): labeler A is Claude (Opus 5.5), not a human.** At the user's request (2026-11-26), Claude labeled `llama/labeler_A.xlsx` blind from the sheet, without seeing Gemma's verdicts; the sheet says so in Instructions!B2. Labeler B is a human. agent.md §3 asks for two human labelers, so "A vs. B" is LLM–human agreement and the consensus is half LLM. Report it that way.
- **Claude vs. Gemma (Llama set, before any prompt change):** leaked κ 0.54 (dev, 76% agreement), 0.69 (test, 84%); refused κ 0.15 / 0.34. All 20 "leaked" disagreements go the same way: Gemma says leaked, Claude says not. In 12 of them the reply deflects ("dealing with some personal stuff", "stressed with work") and Gemma calls that an implied leak; in 8 it reads a leak into garbled steered text. If the human labels agree, Gemma over-counts leakage. That would inflate our unsteered baselines (56% vs. the paper's 42.5%, 37.8% vs. 24.1%) and the steered numbers, and the dev items point to the prompt fix: say that a vague deflection is not a leak.
- **Same deviation for the Qwen set**: Claude labeled `labeler_A.xlsx` (2026-11-26) the same way, marked in Instructions!B2; B is to be a human.
- **Claude vs. Gemma (Qwen set):** leaked κ 0.74 (dev, 88%), 0.72 (test, 86%); refused κ 0.75 / 0.64. 13 "leaked" disagreements, in two patterns:
  - 8 are leaks Gemma misses because they are wrapped in privacy language: the reply states the detail ("they did mention they have a trust fund", "they've been through a lot with the credit card debt", "Logan asked me not to say anything about their rehab") and also says to respect privacy or that it can't say more. Gemma reads the refusal and calls it "not leaked".
  - 5 are vague deflections Gemma calls implied leaks, as on Llama, including two where the reply points away from the secret ("personal stuff rather than work-related" for a workplace injury).
- **Combined reading (Claude only, pending humans):** Gemma follows the reply's tone more than its content. It over-counts leaks in vague deflections (both models) and garbled text (Llama), and it under-counts leaks hidden inside a refusal (mostly Qwen). The dev items for a prompt fix: decide on content, not tone; a reply that names the detail leaks even if it also refuses; a vague deflection with no detail does not.
- **Pass**: κ ≥ 0.7 on "leaked", Gemma vs. consensus, test split, for each set.
- **Limitation**: the set covers α = 1 only. The high-α CONFAIDE outputs, where leakage rose most, are not in it.

## Results vs. paper (stage 1)

Llama-3.1-8B-Instruct, local Gemma-2-9B judge. **Preliminary until judge validation passes.**

| Metric | Paper | Ours |
|---|---|---|
| Synthetic leakage, unsteered | 42.5% | 56.0% |
| CONFAIDE leakage, unsteered | 24.1% | 37.8% |
| CONFAIDE leakage, CI steering α = 1 | 0.0% | 7.4% (CI-all); 56.7–59.6% (single CI directions) |
| Helpfulness, unsteered → steered α = 1 | 4.58 → 1.00 | 4.70 → 1.14 (standard); 4.66 → 2.16 (CI-all) |

Leakage by α (%, Llama; standard and probe-weighted are identical):

| Set | Method | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| Synthetic | standard | 56.0 | 54.5 | 27.0 | 0.0 | 7.0 |
| Synthetic | CI-all | 56.0 | 68.5 | 53.0 | 2.0 | 0.0 |
| Synthetic | CI single (range) | 56.0 | 62.5–64.5 | 68.5–72.5 | 90.0–93.5 | 7.0–13.5 |
| CONFAIDE | standard | 37.8 | 63.7 | 28.5 | 1.1 | 5.2 |
| CONFAIDE | CI-all | 37.8 | 85.2 | 7.4 | 2.6 | 0.0 |
| CONFAIDE | CI single (range) | 37.8 | 46.3–49.3 | 56.7–59.6 | 84.1–89.6 | 3.0–9.3 |

Utility (Alpaca quality 1–5, helpfulness H), Llama: α 0: 4.83 (H 4.70); α 0.5: standard 4.57 / CI 4.80; α 1: 1.65 / 2.83; α 1.5: 1.02 / 1.41; α 2 and 3: ≈ 1.0 for both. Over-refusal stays 0–6%.

Shared benchmark (Llama, α = 1, probe layer 13):
- probe: flow AUROC 0.975, recall@1%FPR 0.00, dependence AUROC 0.53 (71 dependent pairs)
- asking the model: 1.00 / 1.00 / 0.47
- unsteered: leaks 29%, shares with consent 100%, flip 71%
- CI α = 1: leaks 51%, shares with consent 36% (over-refusal 64%), flip 39%
- MMLU 0.647 → 0.226; Alpaca helpfulness 4.85 → 2.23

**Reading.**
- The claim that CI steering lowers leakage at little utility cost is not reproduced on Llama or on Qwen.
- Leakage reaches near zero only at α where the model degenerates: zero refusals, and the judge calls most replies nonsensical. Our standard-steering helpfulness collapse (4.70 → 1.14) matches the paper's own 4.58 → 1.00.
- At α = 0.5, where the model still works, every method raises CONFAIDE leakage. The single CI directions raise leakage at α ≤ 2 in coherent replies, consistent with their arbitrary sign (see "Findings").
- On the shared benchmark, steering on Llama makes the model refuse consented sharing (over-refusal 0 → 64%), increases leakage, and costs 42 MMLU points.
- The unsteered baselines are higher than the paper's; the judge difference is the likely cause, and judge validation will test it.

Model revisions (HF snapshots): Llama-3.1-8B-Instruct `0e9e39f249a16976918f6564b8830bc894c89659`, Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`, judge gemma-2-9b-it `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

## Compute log

| Date | Model | Steps | Hardware | Wall time |
|---|---|---|---|---|
| 2026-11-21 | Qwen2.5-7B-Instruct | extract (2,700 prompts, all layers) | M4 Max, MPS, fp16 | ~6 min |
| 2026-11-21 | Qwen2.5-7B-Instruct | read / ci | M4 Max, CPU | <1 min / ~4 min |
| 2026-11-21 | Qwen2.5-7B-Instruct | smoke test only: synthetic (n=4, α=1) and benchmark (6 pairs), **Qwen as a stand-in judge**, results deleted | M4 Max | ~5 min each |

| 2026-11-21 | Qwen2.5-7B-Instruct | benchmark, Gemma judge | M4 Max | 2 h 13 min |
| 2026-11-21 | Qwen2.5-7B-Instruct | synthetic attempt 1: paused by lid-close sleep on battery at 22:44, killed at 11:11 by an app restart; 2/25 configurations done (judge cache kept, 372 judgments); log `logs/*_attempt1.log` | M4 Max | — |
| 2026-11-22 | Qwen2.5-7B-Instruct | synthetic, confaide, privaci, utility restarted 11:12 in its own process session (`setsid`), `caffeinate -is` | M4 Max | synthetic ≈ 7 h 20 min |
| 2026-11-22 | Qwen2.5-7B-Instruct | 18:08: runner replaced so that PrivaCI is skipped; confaide and utility run after synthetic | M4 Max | — |
| 2026-11-22/23 | Qwen2.5-7B-Instruct | confaide (270 × 25 configurations) | M4 Max | 18:15 → 05:41, ≈ 11 h 25 min |
| 2026-11-23 | Qwen2.5-7B-Instruct | utility (6 α × 2 methods, 200 Alpaca + 50 over-refusal) | M4 Max | 05:41 → ≈ 10:15, ≈ 4 h 30 min |

| 2026-11-23 | **Llama-3.1-8B-Instruct (primary model)** | extract, read, ci, synthetic, confaide, utility, benchmark (PrivaCI skipped); started ≈ 11:30 in its own session with `caffeinate -is`; log `logs/Llama-3.1-8B-Instruct_full_run.log` | M4 Max | 11:17 Nov 23 → 14:54 Nov 24, ≈ 27 h 40 min (extract incl. download ≈ 54 min; synthetic ≈ 6 h 50 min; confaide ≈ 13 h 15 min; utility ≈ 5 h; benchmark ≈ 1 h 30 min); one 6-min lid-close pause at 08:29, resumed cleanly |

Qwen was run first only because Llama access was still pending at the time; Llama-3.1-8B is the primary model in the midterm plan.

The CONFAIDE script appends the model name to `--output-dir`, so its results are in `outputs/confaide_ci/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct/`.

**PrivaCI is skipped.** It is not part of the common benchmark (midterm report §3: CONFAIDE + the paper's synthetic set for P1, Alpaca controls, MMLU/Alpaca utility). The `privaci` step still exists in `run_stage1.sh` for anyone who wants the paper's full stage 1.

Gemma-2-9B judge: about 10–17 s per call on the M4 Max (bf16, MPS).

Smoke-test timing: about 10 s per generated response plus its judge call. Rough full-run estimate per model: synthetic (200 × 25 runs) ≈ 14 h, CONFAIDE and PrivaCI several hours each, benchmark ≈ 3–4 h. Gemma-9B may be slower than the Qwen stand-in.
