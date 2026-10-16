from functools import lru_cache

import pandas as pd

from app.config import PROCESSED_DIR


@lru_cache(maxsize=1)
def load_orders() -> pd.DataFrame:
    path = PROCESSED_DIR / "orders_processed.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Processed order data is unavailable. Run scripts/prepare_dataset.py after downloading and inspecting the dataset."
        )
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
