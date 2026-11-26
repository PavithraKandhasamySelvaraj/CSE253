# cse253-bench

Shared benchmark for sensitive-information dependence + deception detection (CSE 253).
Spec: `agent.md`. Per-paper code in `src/<paper>/`, shared code in `src/common/`.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
git clone https://github.com/ApolloResearch/deception-detection external/deception-detection
hf auth login   # Llama-3.1 is gated: accept the license on its HF page first
```

## P4 (deception probe)

```
python -m src.p4.validate_released                       # metric check, no GPU
python -m src.p4.run --stage 1 --model meta-llama/Llama-3.1-8B-Instruct --gpu-mem-gib 9
```

Outputs: `results/p4/<model>/{metrics.json,config.yaml,probes.pt}`, per-token scores in
`outputs/p4/<model>/`. Deviations and choices: `NOTES.md`.
