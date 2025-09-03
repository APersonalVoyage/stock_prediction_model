from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    # src/stock_predictor/config.py -> parents[2] is the project root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEFAULT_TEST_SIZE_DAYS = 252  # ~1 trading year
DEFAULT_VALIDATION_SIZE_DAYS = 126  # ~half trading year
