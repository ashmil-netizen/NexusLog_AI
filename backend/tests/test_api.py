from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services.models import load_test_predictions


def test_health_endpoint():
    with TestClient(main.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def sample_orders():
    return pd.DataFrame([{
        "order_id": "order-1",
        "order_status": "delivered",
        "order_purchase_timestamp": pd.Timestamp("2018-01-01"),
        "order_approved_at": "2018-01-01",
        "order_delivered_carrier_date": "2018-01-02",
        "order_estimated_delivery_date": "2018-01-04",
        "order_delivered_customer_date": "2018-01-03",
        "customer_state": "SP",
        "item_count": 1,
        "item_price_total": 10.0,
        "item_freight_total": 1.0,
        "late_delivery": 0,
        "purchase_hour": 12,
    }])


class FixedProbabilityModel:
    def predict_proba(self, _frame):
        return [[0.2, 0.8]]


def test_invalid_prediction_request_is_rejected():
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": "order-1"})
    assert response.status_code == 422


def test_unknown_order_returns_404(monkeypatch):
    monkeypatch.setattr(main, "load_orders", sample_orders)
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": "missing", "stage": "A"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("stage", "missing_field"),
    [("B", "order_approved_at"), ("C", "order_approved_at"), ("C", "order_delivered_carrier_date")],
)
def test_missing_stage_milestone_returns_422(monkeypatch, stage, missing_field):
    orders = sample_orders()
    orders.loc[0, missing_field] = None
    monkeypatch.setattr(main, "load_orders", lambda: orders)
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": "order-1", "stage": stage})
    assert response.status_code == 422


def test_orders_endpoint_paginates_and_filters(monkeypatch):
    orders = pd.concat([sample_orders(), sample_orders()], ignore_index=True)
    orders.loc[1, "order_id"] = "order-2"
    orders.loc[1, "late_delivery"] = 1
    monkeypatch.setattr(main, "load_orders", lambda: orders)
    with TestClient(main.app) as client:
        response = client.get("/api/orders", params={"page": 1, "page_size": 1, "late_delivery": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["order_id"] == "order-2"


def test_orders_endpoint_sorts_and_filters_by_region(monkeypatch):
    orders = pd.concat([sample_orders(), sample_orders()], ignore_index=True)
    orders.loc[1, "order_id"] = "order-2"
    orders.loc[1, "customer_state"] = "RJ"
    orders.loc[1, "item_count"] = 5
    monkeypatch.setattr(main, "load_orders", lambda: orders)
    with TestClient(main.app) as client:
        response = client.get(
            "/api/orders",
            params={"customer_state": "rj", "sort_by": "item_count", "sort_order": "desc"},
        )
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["sort_by"] == "item_count"
    assert response.json()["items"][0]["order_id"] == "order-2"


def test_summary_contains_dataset_and_test_prediction_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_orders", sample_orders)
    monkeypatch.setattr(main, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(main, "load_report", lambda _name: [{"stage": "A", "threshold": 0.4}])
    pd.DataFrame({
        "order_id": ["order-1", "order-2"],
        "stage": ["A", "A"],
        "predicted_late_delivery": [1, 0],
    }).to_csv(tmp_path / "test_predictions.csv", index=False)
    with TestClient(main.app) as client:
        response = client.get("/api/summary")
    assert response.status_code == 200
    assert response.json()["eligible_orders"] == 1
    assert response.json()["high_risk_order_count"] == 1
    assert response.json()["high_risk_threshold"] == 0.4


def test_high_risk_orders_separate_prediction_from_actual_outcome(monkeypatch):
    monkeypatch.setattr(main, "load_orders", sample_orders)
    monkeypatch.setattr(main, "load_test_predictions", lambda: pd.DataFrame([{
        "order_id": "order-1",
        "stage": "A",
        "delay_probability": 0.8,
        "predicted_late_delivery": 1,
        "threshold": 0.6,
    }]))
    with TestClient(main.app) as client:
        response = client.get("/api/orders/high-risk", params={"stage": "A"})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["delay_probability"] == 0.8
    assert item["actual_late_delivery"] == 0


def test_risk_distribution_uses_saved_probabilities_and_stage_threshold(monkeypatch):
    monkeypatch.setattr(main, "load_test_predictions", lambda: pd.DataFrame({
        "stage": ["A", "A", "A", "B"],
        "delay_probability": [0.1, 0.4, 0.8, 0.9],
    }))
    monkeypatch.setattr(main, "load_report", lambda _name: [{"stage": "A", "threshold": 0.6}])
    with TestClient(main.app) as client:
        response = client.get("/api/metrics/risk-distribution", params={"stage": "A"})
    assert response.status_code == 200
    assert response.json()["sample_count"] == 3
    assert response.json()["bins"] == [
        {"category": "low", "count": 1},
        {"category": "medium", "count": 1},
        {"category": "high", "count": 1},
    ]


def test_missing_model_returns_actionable_503(monkeypatch):
    monkeypatch.setattr(main, "load_orders", sample_orders)
    monkeypatch.setattr(main, "load_model", lambda _stage: (_ for _ in ()).throw(FileNotFoundError("Train first.")))
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": "order-1", "stage": "A"})
    assert response.status_code == 503
    assert "Train first." in response.json()["detail"]


def test_prediction_uses_stage_feature_schema(monkeypatch):
    monkeypatch.setattr(main, "load_orders", sample_orders)
    captured = {}

    class CapturingModel(FixedProbabilityModel):
        def predict_proba(self, frame):
            captured["columns"] = list(frame.columns)
            return super().predict_proba(frame)

    monkeypatch.setattr(main, "load_model", lambda _stage: {
        "features": ["purchase_hour"],
        "calibrated_model": CapturingModel(),
        "threshold": 0.6,
    })
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": "order-1", "stage": "A"})
    assert response.status_code == 200
    assert response.json()["delay_probability"] == 0.8
    assert captured["columns"] == ["purchase_hour"]


def test_prediction_with_trained_artifact_when_available():
    backend = Path(__file__).resolve().parents[1]
    data_path = backend / "data" / "processed" / "orders_processed.csv"
    model_path = backend / "models" / "stage_A.joblib"
    if not data_path.is_file() or not model_path.is_file():
        pytest.skip("Prepared data and trained model artifact are not available.")
    main.load_orders.cache_clear()
    orders = main.load_orders()
    order_id = str(orders.iloc[0]["order_id"])
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"order_id": order_id, "stage": "A"})
    assert response.status_code == 200
    assert 0 <= response.json()["delay_probability"] <= 1


def test_saved_test_prediction_is_returned_when_available():
    backend = Path(__file__).resolve().parents[1]
    predictions_path = backend / "reports" / "test_predictions.csv"
    if not predictions_path.is_file():
        pytest.skip("Saved test predictions are not available.")
    load_test_predictions.cache_clear()
    expected = pd.read_csv(predictions_path, dtype={"order_id": str})
    expected = expected.loc[expected["stage"] == "A"].iloc[0]
    main.load_orders.cache_clear()
    with TestClient(main.app) as client:
        response = client.get(f"/api/orders/{expected['order_id']}/risk", params={"stage": "A"})
    assert response.status_code == 200
    assert response.json()["delay_probability"] == pytest.approx(expected["delay_probability"])
    assert "actual_late_delivery" not in response.json()
