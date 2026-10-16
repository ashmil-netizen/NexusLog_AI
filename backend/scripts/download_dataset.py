"""Download the Olist public dataset using KaggleHub and verify its CSV files."""

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_CSVS = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_customers_dataset.csv",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("download_dataset")


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        LOGGER.error("KaggleHub is not installed. Run: python -m pip install -r requirements.txt")
        return 1

    try:
        dataset_path = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce")).resolve()
    except Exception as exc:
        LOGGER.error(
            "KaggleHub could not download the dataset (%s). Check network access and KaggleHub's "
            "authentication guidance for this public dataset; never add credentials to this project.",
            exc,
        )
        return 1

    LOGGER.info("KaggleHub dataset directory: %s", dataset_path)
    if not dataset_path.is_dir():
        LOGGER.error("KaggleHub returned a path that is not a directory.")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        LOGGER.error("No CSV files were found in the downloaded dataset directory.")
        return 1

    for source in csv_files:
        target = RAW_DIR / source.name
        if source.resolve() == target.resolve():
            LOGGER.info("Already in raw-data directory; not copying %s", source.name)
            continue
        if target.exists():
            LOGGER.info("Keeping existing raw file; not overwriting %s", target.name)
            continue
        shutil.copy2(source, target)
        LOGGER.info("Copied %s into the project raw-data directory", source.name)

    missing = [name for name in EXPECTED_CSVS if not (RAW_DIR / name).is_file()]
    if missing:
        LOGGER.error("Required dataset CSV files are missing: %s", ", ".join(missing))
        return 1

    for filename in EXPECTED_CSVS:
        path = RAW_DIR / filename
        try:
            sample = pd.read_csv(path, nrows=5)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            LOGGER.error("Could not read %s: %s", filename, exc)
            return 1
        LOGGER.info("Verified %s (%d sample rows, %d columns)", filename, len(sample), len(sample.columns))

    LOGGER.info("Dataset download and required-file verification succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
