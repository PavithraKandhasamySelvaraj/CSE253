# CSE 253: Benchmarking Methods for Detecting Sensitive-Information Dependence and Deceptive Behavior in LLMs

Team: Mukund Ojha, Mignot Mesele, Mathis Aubert, Christian Li, Pavithra Selvaraj (UC Santa Cruz).

## Research question

Can we detect when an LLM's decisions depend on sensitive personal information (race, gender, sexual orientation, or a private detail shared in another context), and whether the model is deceptive about that dependence?

Output inspection alone is not enough: models change hiring and sentencing decisions with demographic signals while their explanations never mention them. Interpretability methods (linear probes, steering directions, concept editing) read or modify internal representations instead. Each published method uses its own task, models, data and metrics, so they cannot be compared.

**Our contribution:** a common benchmark that reimplements five existing methods and evaluates them under the same conditions: the same models, data, judge and metrics.

## The five methods

| ID | Paper | Method | Attribute / behavior |
|---|---|---|---|
| P1 | Wang, Xiong & Shu, arXiv 2604.00209 [1] | CI privacy probing (linear probes, PCA) and CI-parametric steering | Contextual privacy norms |
| P2 | Karvonen & Marks, arXiv 2506.10922 [2] | Demographic directions and affine concept editing | Race, gender (hiring) |
| P3 | Nakka et al., arXiv 2507.02332 [3] | Attention-head probing and steering for disclosure | Sensitive personal information |
| P4 | Goldowsky-Dill et al., arXiv 2502.03407 [6] | Deception probes | Strategic deception |
| P5 | Chen et al., arXiv 2406.07882 [10] | User-model reading probes | User demographics |

Selection criteria: open-weight models we can run, a per-input score or decision comparable across methods, a check against a counterfactual behavioral ground truth, and relevance to either the dependence or the deception part of the question.

Not reproduced as methods: the three black-box papers (AgentDAM [7], the sentencing study [8], Staab et al. [9]) supply behavioral baselines and evaluation data. The SAE study [4] is correlational and needs precomputed SAEs. NeuroFaith [5] is a stretch goal.

## Common evaluation protocol

- **Models:** Llama-3.1-8B-Instruct (primary) and Qwen-2.5-7B-Instruct (second model, to test generalization).
- **Tasks:** counterfactual pairs where only the sensitive attribute changes:
  - hiring resumes with race and gender signals [2]
  - contextual privacy scenarios from CONFAIDE and the synthetic set of [1]
  - user-attribute conversations [10]
  - sensitive-disclosure prompts [3]

  A pair whose decision flips is labeled *dependent*.
- **Shared pipeline:** residual-stream activations cached once per model for every layer. All methods use the same probe training (logistic regression), direction estimation (difference of means) and train/test splits.
- **Metrics:**
  - Detection: AUROC and recall at 1% false-positive rate, with neutral Alpaca prompts as the control set [6].
  - Behavior: decision-flip rate, interview-rate gap, leakage rate.
  - Interventions: reduction in bias or leakage, plus the utility cost on an MMLU subset and on Alpaca helpfulness.
- **Black-box baselines every white-box method must beat:**
  - (a) asking the model directly about the attribute [10]
  - (b) checking whether the chain-of-thought mentions the attribute [2, 8]
  - (c) an external LLM monitor that reads the full transcript [6]
- **Deception about dependence:** a case is deceptive when the decision depends on the attribute (the counterfactual decision flips) but the explanation does not mention it. We test whether the deception probes [6] fire on these cases. This is a proxy: an explanation that omits the attribute is not necessarily an intentional lie.
- **Judge:** one fixed LLM judge for every method, validated against a hand-labeled sample before any number is reported.

Each method is reported twice: on its original task, to check that our reimplementation matches the paper, and on the shared suite.

## Plan

1. Review and categorize ten papers.
2. Select five methods.
3. Define the common evaluation protocol.
4. Reimplement the five methods in one environment.
5. Compare them and release the benchmark.

Steps 1–3 are covered in the midterm report.

**Known obstacles:**
- **Heterogeneous setups:** handled by the common protocol.
- **Compute:** the deception probes of [6] were trained on a 70B model; we retrain them on Llama-3.1-8B.
- **Data:** some data is not public, so we use public benchmarks and regenerate synthetic data where needed.
- **Judge cost and variance:** a single local judge, validated on a hand-labeled sample.
- **Synthetic training data:** it may encode stereotypes, so we test transfer to non-synthetic data where possible.

## Repository layout

| Path | Contents |
|---|---|
| `benchmark_dataset/` | Shared, identical-for-everyone pieces: Alpaca calibration set, MMLU utility subset, `judge.py` and `shared_judge_config.json` |
| `benchmark_dataset_code/` | Scripts that build the shared sets (see its README) |
| `Paper_3/` | P3 reproduction (Nakka et al.); see `Paper_3/NOTES.md` |
| `p1_wang/` | P1 reproduction (Wang et al.); on branch `p1-wang-ci-steering`, see `p1_wang/NOTES.md` |

## References

1. H. Wang, L. Xiong, K. Shu. Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations. arXiv:2604.00209, 2026.
2. A. Karvonen, S. Marks. Robustly Improving LLM Fairness in Realistic Settings via Interpretability. arXiv:2506.10922, 2025.
3. K. K. Nakka, X. Jiang, D. Usynin, X. Zhou. PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage. arXiv:2507.02332, 2025.
4. K. Simbeck, M. Mahran. Mechanistic Interpretability with SAEs: Probing Religion, Violence, and Geography in Large Language Models. arXiv:2509.17665, 2025.
5. M. Bhan, J.-N. Vittaut, N. Chesneau, S. Chandar, M.-J. Lesot. NeuroFaith: Evaluating LLM Self-Explanation Faithfulness via Internal Representation Alignment. arXiv:2506.09277, 2025.
6. N. Goldowsky-Dill, B. Chughtai, S. Heimersheim, M. Hobbhahn. Detecting Strategic Deception Using Linear Probes. arXiv:2502.03407, 2025.
7. A. Zharmagambetov, C. Guo, I. Evtimov, M. Pavlova, R. Salakhutdinov, K. Chaudhuri. AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents. NeurIPS Datasets and Benchmarks, 2025. arXiv:2503.09780.
8. J. No. Is Artificial Intelligence Fair for All? Sexual Orientation Bias in LLM Sentencing. ICML, 2026.
9. R. Staab, M. Vero, M. Balunović, M. Vechev. Beyond Memorization: Violating Privacy via Inference with Large Language Models. ICLR, 2024. arXiv:2310.07298.
10. Y. Chen, A. Wu, T. DePodesta, C. Yeh, K. Li, N. Castillo Marin, O. Patel, J. Riecke, S. Raval, O. Seow, M. Wattenberg, F. Viégas. Designing a Dashboard for Transparency and Control of Conversational AI. arXiv:2406.07882, 2024.
11. N. Mireshghallah, H. Kim, X. Zhou, Y. Tsvetkov, M. Sap, R. Shokri, Y. Choi. Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory. ICLR, 2024.
