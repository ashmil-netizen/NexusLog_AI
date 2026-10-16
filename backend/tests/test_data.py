import pandas as pd
import pytest

from app.services.features import STAGE_FEATURES
from scripts.prepare_dataset import build_dataset
from scripts.train_model import chronological_split


def sample_frames():
    orders = pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "customer_id": ["u1", "u2", "u3", "u4"],
        "order_status": ["delivered", "delivered", "canceled", "delivered"],
        "order_purchase_timestamp": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
        "order_approved_at": ["2018-01-01", None, None, "2018-01-04"],
        "order_delivered_carrier_date": ["2018-01-02", None, None, "2018-01-05"],
        "order_delivered_customer_date": ["2018-01-03", "2018-01-03", None, None],
        "order_estimated_delivery_date": ["2018-01-04", "2018-01-02", None, "2018-01-10"],
    })
    items = pd.DataFrame({
        "order_id": ["a", "a", "b", "d"], "order_item_id": [1, 2, 1, 1],
        "price": [10.0, 20.0, 5.0, 15.0], "freight_value": [1.0, 2.0, 1.0, 2.0],
        "seller_id": ["s1", "s2", "s1", "s3"],
    })
    payments = pd.DataFrame({
        "order_id": ["a", "b", "d"], "payment_type": ["credit_card", "boleto", "credit_card"],
        "payment_installments": [1, 1, 2], "payment_value": [33.0, 6.0, 17.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["u1", "u2", "u3", "u4"], "customer_state": ["SP", "RJ", "MG", "SP"],
    })
    return orders, items, payments, customers


def test_target_and_order_aggregation():
    frame, audit = build_dataset(*sample_frames())
    assert frame["order_id"].tolist() == ["a", "b"]
    assert frame["late_delivery"].tolist() == [0, 1]
    assert frame.loc[frame["order_id"] == "a", "item_count"].item() == 2
    assert frame["order_id"].is_unique
    assert audit["removed_records"]["canceled_orders"] == 1
    assert audit["removed_records"]["missing_delivery_or_estimate"] == 1


def test_stage_feature_allowlists_exclude_outcomes():
    outcome_columns = {
        "order_delivered_customer_date", "order_estimated_delivery_date",
        "late_delivery", "order_status",
    }
    assert all(not (outcome_columns & set(features)) for features in STAGE_FEATURES.values())


def test_chronological_split_orders_periods():
    frame = pd.DataFrame({
        "order_purchase_timestamp": pd.date_range("2020-01-01", periods=120, freq="D"),
    })
    train, validation, test = chronological_split(frame)
    assert train["order_purchase_timestamp"].max() < validation["order_purchase_timestamp"].min()
    assert validation["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()


def test_conflicting_duplicate_order_ids_fail():
    orders, items, payments, customers = sample_frames()
    orders.loc[1, "order_id"] = "a"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        build_dataset(orders, items, payments, customers)
