"""Inspect downloaded Olist CSV schemas and data quality without training a model."""

import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATE_COLUMNS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
}
TARGET_CANDIDATES = ["order_delivered_customer_date", "order_estimated_delivery_date"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("inspect_dataset")


def inspect_file(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "data_types": {key: str(value) for key, value in frame.dtypes.items()},
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "basic_statistics": json.loads(frame.describe(include="all").to_json(default_handler=str)),
        "candidate_target_variables": [name for name in TARGET_CANDIDATES if name in frame.columns],
        "relevant_date_columns": [name for name in DATE_COLUMNS if name in frame.columns],
    }


def main() -> int:
    csv_paths = sorted(RAW_DIR.glob("*.csv"))
    if not csv_paths:
        LOGGER.error("No CSV files found in %s. Run scripts/download_dataset.py first.", RAW_DIR)
        return 1
    report = {}
    for path in csv_paths:
        try:
            report[path.name] = inspect_file(path)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            LOGGER.error("Failed inspecting %s: %s", path.name, exc)
            return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
