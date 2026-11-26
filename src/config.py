"""
Shared settings — Stage 2 (benchmark port), Phase A.

"""
import os

# ---- Model for this run ----------------------------------------------------
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL  = "google/gemma-2-9b-it"      # unchanged from Stage 1
JUDGE_BACKEND = "local"

TARGET_DTYPE = "float16" 
MAX_NEW_TOKENS = 200

# ---- Paths: model-specific, so Stage 1's Llama-2 data is never touched ----
MODEL_SHORT_NAME = TARGET_MODEL.split("/")[-1].lower()
DATA_DIR = f"/kaggle/working/p3_data_{MODEL_SHORT_NAME}"
os.makedirs(DATA_DIR, exist_ok=True)

# Reuse the SAME 677-name list from Stage 1 (Llama-2 run) for a fair
# cross-model comparison — no point paying to regenerate it, and the paper's
# own benchmark compares models on the same question set.
NAMES_SOURCE_PATH = "/kaggle/working/p3_data/names.json"   # Stage 1's original location

NAMES_PATH          = f"{DATA_DIR}/names.json"
BASELINE_RESP_PATH  = f"{DATA_DIR}/baseline_responses.json"
BASELINE_LABELS_PATH= f"{DATA_DIR}/baseline_labels.json"
ACTIVATIONS_PATH    = f"{DATA_DIR}/activations.npz"
PROBES_PATH         = f"{DATA_DIR}/probes.npz"
HEAD_RANKING_PATH   = f"{DATA_DIR}/head_ranking.json"
STEERED_RESP_PATH   = f"{DATA_DIR}/steered_responses.json"
STEERED_LABELS_PATH = f"{DATA_DIR}/steered_labels.json"
FACTUALITY_PATH      = f"{DATA_DIR}/factuality.json"
METRICS_PATH         = f"{DATA_DIR}/metrics.json"
CONFIG_LOG_PATH       = f"{DATA_DIR}/config_used.json"

# ---- Data sizes --------------------------------------------------------
N_NAMES_TARGET = 800          # not used this run — names are reused, not regenerated
N_CATEGORIES_PER_CALL = 18
NAMES_PER_CATEGORY_REQUEST = 50

TRAIN_SET_SIZE = 110          # auto-shrinks to whatever's available, same as Stage 1
SEED = 0

# ---- Steering sweep — TRIMMED for the 6-hour Phase A budget ---------------
# Stage 1 used 3 k-values x 4 alpha-values = 12 configs. This run uses
# 2 x 2 = 4 configs, tuned on a smaller subset, with only the SINGLE best
# config run on the full test pool (the shared protocol also asks for a
# single-configuration disclosure rate, not just best-of, so this doubles
# as that number with no extra work).
K_GRID = [16, 96]
ALPHA_GRID = [-80, -40]
TUNE_SUBSET_SIZE = 100
FULL_RUN_TOP_CONFIGS = 1

FACTUALITY_SAMPLE_SIZE = 50

HF_HOME = os.environ.get("HF_HOME", "/root/hf_cache") 
os.environ.setdefault("HF_HOME", HF_HOME)
