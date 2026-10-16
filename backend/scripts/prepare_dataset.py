"""Create one audited, order-level dataset with explicit milestone features."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.features import STAGE_FEATURES

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REQUIRED = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}
DATE_FIELDS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("prepare_dataset")


def _read(name: str) -> pd.DataFrame:
    path = RAW_DIR / REQUIRED[name]
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path.name}. Run scripts/download_dataset.py first.")
    return pd.read_csv(path, low_memory=False)


def build_dataset(orders: pd.DataFrame, items: pd.DataFrame, payments: pd.DataFrame, customers: pd.DataFrame):
    required_order_fields = {"order_id", "order_status", *DATE_FIELDS}
    missing_fields = required_order_fields.difference(orders.columns)
    if missing_fields:
        raise ValueError(f"Orders table is missing required fields: {sorted(missing_fields)}")

    duplicate_order_rows = int(orders.duplicated("order_id", keep="first").sum())
    conflicting = orders.loc[orders.duplicated("order_id", keep=False)].groupby("order_id").nunique(dropna=False)
    if not conflicting.empty and (conflicting > 1).any(axis=None):
        raise ValueError("Conflicting duplicate order IDs found; resolve the source records before preparing data.")
    orders = orders.drop_duplicates("order_id").copy()

    for field in DATE_FIELDS:
        orders[field] = pd.to_datetime(orders[field], errors="coerce")

    item_aggregates = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        item_price_total=("price", "sum"),
        item_freight_total=("freight_value", "sum"),
        seller_count=("seller_id", "nunique"),
    )
    payment_aggregates = payments.groupby("order_id", as_index=False).agg(
        payment_type=("payment_type", lambda values: values.mode().iloc[0] if not values.mode().empty else None),
        payment_installments=("payment_installments", "max"),
        payment_value_total=("payment_value", "sum"),
    )
    customer_fields = customers[["customer_id", "customer_state"]].drop_duplicates("customer_id")
    merged = orders.merge(item_aggregates, on="order_id", how="left", validate="one_to_one")
    merged = merged.merge(payment_aggregates, on="order_id", how="left", validate="one_to_one")
    merged = merged.merge(customer_fields, on="customer_id", how="left", validate="many_to_one")

    removed = {
        "duplicate_order_rows": duplicate_order_rows,
        "canceled_orders": int((merged["order_status"] == "canceled").sum()),
        "other_non_delivered_orders": int(
            ((merged["order_status"] != "delivered") & (merged["order_status"] != "canceled")).sum()
        ),
        "missing_delivery_or_estimate": 0,
        "invalid_timestamps": 0,
        "missing_purchase_timestamp": int(merged["order_purchase_timestamp"].isna().sum()),
    }
    eligible = merged.loc[merged["order_status"] == "delivered"].copy()
    missing_outcome = (
        eligible["order_delivered_customer_date"].isna()
        | eligible["order_estimated_delivery_date"].isna()
    )
    removed["missing_delivery_or_estimate"] = int(missing_outcome.sum())
    eligible = eligible.loc[~missing_outcome].copy()
    invalid = (
        eligible["order_purchase_timestamp"].isna()
        | (eligible["order_delivered_customer_date"] < eligible["order_purchase_timestamp"])
        | (eligible["order_estimated_delivery_date"] < eligible["order_purchase_timestamp"])
        | (
            eligible["order_approved_at"].notna()
            & (
                (eligible["order_approved_at"] < eligible["order_purchase_timestamp"])
                | (eligible["order_approved_at"] > eligible["order_delivered_customer_date"])
            )
        )
        | (
            eligible["order_delivered_carrier_date"].notna()
            & (
                (eligible["order_delivered_carrier_date"] < eligible["order_purchase_timestamp"])
                | (eligible["order_delivered_carrier_date"] > eligible["order_delivered_customer_date"])
            )
        )
    )
    removed["invalid_timestamps"] = int(invalid.sum())
    eligible = eligible.loc[~invalid].copy()

    purchase = eligible["order_purchase_timestamp"]
    approval = eligible["order_approved_at"]
    handover = eligible["order_delivered_carrier_date"]
    eligible["purchase_month"] = purchase.dt.month
    eligible["purchase_dayofweek"] = purchase.dt.dayofweek
    eligible["purchase_hour"] = purchase.dt.hour
    eligible["approval_elapsed_hours"] = (approval - purchase).dt.total_seconds() / 3600
    eligible["handover_elapsed_hours"] = (handover - purchase).dt.total_seconds() / 3600
    eligible["late_delivery"] = (
        eligible["order_delivered_customer_date"] > eligible["order_estimated_delivery_date"]
    ).astype("int8")
    eligible["order_id"] = eligible["order_id"].astype(str)
    eligible = eligible.sort_values("order_purchase_timestamp").reset_index(drop=True)

    if eligible["order_id"].duplicated().any():
        raise ValueError("Order-level joins produced duplicate order IDs.")
    audit = {
        "input_orders_after_duplicate_resolution": int(len(orders)),
        "eligible_delivered_orders": int(len(eligible)),
        "removed_records": removed,
        "late_delivery_count": int(eligible["late_delivery"].sum()),
        "on_time_count": int((eligible["late_delivery"] == 0).sum()),
        "delay_rate": float(eligible["late_delivery"].mean()) if len(eligible) else None,
        "stage_eligibility": {
            stage: int(eligible[columns].notna().all(axis=1).sum())
            for stage, columns in STAGE_FEATURES.items()
        },
        "feature_availability": {
            "A": "Purchase calendar, customer state, and aggregated line items available at order placement; no approval or handover information.",
            "B": "Stage A plus approval elapsed time, payment type, maximum installments, and aggregated payment value; rows require an approval timestamp.",
            "C": "Stage B plus purchase-to-carrier-handover elapsed time; rows require a carrier-handover timestamp.",
        },
        "stage_features": STAGE_FEATURES,
        "target": "late_delivery = 1 when customer delivery timestamp is strictly later than estimated delivery timestamp; otherwise 0.",
    }
    return eligible, audit


def main() -> int:
    try:
        eligible, audit = build_dataset(_read("orders"), _read("items"), _read("payments"), _read("customers"))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 1
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(PROCESSED_DIR / "orders_processed.csv", index=False)
    (PROCESSED_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    LOGGER.info("Prepared %d unique eligible orders.", len(eligible))
    LOGGER.info("Delayed: %d; on time: %d; delay rate: %s", audit["late_delivery_count"], audit["on_time_count"], audit["delay_rate"])
    LOGGER.info("Audit written to %s", PROCESSED_DIR / "data_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
