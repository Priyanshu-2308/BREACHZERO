"""Shared paths and configuration constants."""

# Seed is fixed to keep split creation and initializations reproducible across runs.
SEED = 42
# Sequence length is capped at 256 tokens to cover standard prompts while limiting latency.
MAX_LEN = 256
# DistilBERT uncased is used as the primary guard for lightweight inference.
BASE_MODEL = "distilbert-base-uncased"
# Multilingual model is evaluated to test non-English and code-mixed prompt injections.
MULTI_MODEL = "distilbert-base-multilingual-cased"

DATA_DIR = "data"
ATTACK_DIR = "data/attacks"
MODEL_DIR = "models"
RESULTS_DIR = "results"


