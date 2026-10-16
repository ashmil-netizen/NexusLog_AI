import json
from functools import lru_cache

import joblib
import pandas as pd

from app.config import MODEL_DIR, REPORT_DIR


@lru_cache(maxsize=3)
def load_model(stage: str):
    path = MODEL_DIR / f"stage_{stage}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model for stage {stage} is unavailable. Run scripts/train_model.py after preparing the dataset."
        )
    return joblib.load(path)


def load_report(name: str):
    path = REPORT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment output '{name}' is unavailable. Run scripts/train_model.py after preparing the dataset."
        )
    with path.open(encoding="utf-8") as report_file:
        return json.load(report_file)


@lru_cache(maxsize=1)
def load_test_predictions():
    path = REPORT_DIR / "test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Saved test predictions are unavailable. Run scripts/train_model.py after preparing the dataset."
        )
    return pd.read_csv(path, dtype={"order_id": str})
