from pathlib import Path
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "dataset" / "locomo10.json"
RESULTS_DIR = ROOT / "results"

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 30
DEFAULT_RUNS = 3

# Categories included in main eval (cat 5 excluded — adversarial / unanswerable)
EVAL_CATEGORIES = {1, 2, 3, 4}

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}
