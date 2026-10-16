from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, REPORT_DIR
from app.schemas import PredictionRequest, PredictionResponse
from app.services.data import json_safe, load_orders
from app.services.features import STAGE_FEATURES, STAGE_NAMES
from app.services.models import load_model, load_report, load_test_predictions


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="NexusLog AI API",
    description="Retrospective stage-aware delivery-delay research API.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _orders():
    try:
        return load_orders()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _record(order_id: str):
    orders = _orders()
    matched = orders.loc[orders["order_id"].astype(str) == order_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail="Order not found in the eligible prepared dataset.")
    return matched.iloc[0]


def _predict(order_id: str, stage: str) -> PredictionResponse:
    record = _record(order_id)
    if stage in ("B", "C") and pd.isna(record.get("order_approved_at")):
        raise HTTPException(status_code=422, detail=f"Order is not eligible at stage {stage}: payment-approval timestamp is missing.")
    if stage == "C" and pd.isna(record.get("order_delivered_carrier_date")):
        raise HTTPException(status_code=422, detail="Order is not eligible at carrier handover: handover timestamp is missing.")
    try:
        bundle = load_model(stage)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    features = bundle["features"]
    row = pd.DataFrame([{feature: record.get(feature) for feature in features}])
    probability = float(bundle["calibrated_model"].predict_proba(row)[0][1])
    threshold = float(bundle["threshold"])
    category = "high" if probability >= threshold else "medium" if probability >= threshold * 0.5 else "low"
    return PredictionResponse(
        order_id=order_id,
        stage=stage,
        delay_probability=probability,
        risk_category=category,
        threshold=threshold,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "NexusLog AI API"}


@app.get("/api/summary")
def summary():
    orders = _orders()
    try:
        metrics = load_report("stage_metrics.json")
    except FileNotFoundError:
        metrics = None
    high_risk_order_count = None
    high_risk_threshold = None
    if metrics:
        stage_a_metrics = next((item for item in metrics if item["stage"] == "A"), None)
        if stage_a_metrics:
            high_risk_threshold = float(stage_a_metrics["threshold"])
            try:
                predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
            except FileNotFoundError:
                predictions = None
            if predictions is not None:
                stage_a_predictions = predictions.loc[predictions["stage"] == "A"]
                high_risk_order_count = int(stage_a_predictions["predicted_late_delivery"].sum())
    return {
        "eligible_orders": int(len(orders)),
        "delayed_orders": int(orders["late_delivery"].sum()),
        "delay_rate": float(orders["late_delivery"].mean()),
        "high_risk_order_count": high_risk_order_count,
        "high_risk_threshold": high_risk_threshold,
        "stage_metrics": metrics,
    }


@app.get("/api/orders")
def list_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    q: str = Query("", max_length=100),
    late_delivery: int | None = Query(None, ge=0, le=1),
    customer_state: str | None = Query(None, min_length=2, max_length=2),
    sort_by: Literal["order_purchase_timestamp", "item_count", "item_price_total", "late_delivery"] = "order_purchase_timestamp",
    sort_order: Literal["asc", "desc"] = "desc",
):
    orders = _orders()
    if q:
        orders = orders.loc[orders["order_id"].astype(str).str.contains(q, case=False, regex=False)]
    if late_delivery is not None:
        orders = orders.loc[orders["late_delivery"] == late_delivery]
    if customer_state:
        orders = orders.loc[orders["customer_state"].str.upper() == customer_state.upper()]
    orders = orders.sort_values(sort_by, ascending=sort_order == "asc", na_position="last")
    total = int(len(orders))
    start = (page - 1) * page_size
    columns = [
        "order_id",
        "order_purchase_timestamp",
        "customer_state",
        "late_delivery",
        "item_count",
        "item_price_total",
    ]
    rows = [
        {key: json_safe(value) for key, value in row.items()}
        for row in orders.iloc[start : start + page_size][columns].to_dict(orient="records")
    ]
    return {
        "items": rows,
        "page": page,
        "page_size": page_size,
        "total": total,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }


@app.get("/api/orders/high-risk")
def high_risk_orders(stage: str = Query("A", pattern="^[ABC]$"), limit: int = Query(6, ge=1, le=50)):
    orders = _orders()
    try:
        predictions = load_test_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    selected = predictions.loc[
        (predictions["stage"] == stage) & (predictions["predicted_late_delivery"] == 1)
    ].sort_values("delay_probability", ascending=False).head(limit)
    joined = selected.merge(
        orders[[
            "order_id",
            "order_purchase_timestamp",
            "customer_state",
            "late_delivery",
        ]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for item in joined.to_dict(orient="records"):
        rows.append({
            "order_id": str(item["order_id"]),
            "stage": stage,
            "delay_probability": float(item["delay_probability"]),
            "threshold": float(item["threshold"]),
            "order_purchase_timestamp": json_safe(item["order_purchase_timestamp"]),
            "customer_state": json_safe(item["customer_state"]),
            "actual_late_delivery": int(item["late_delivery"]),
        })
    return {"stage": stage, "count": len(rows), "items": rows}


@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    record = _record(order_id)
    fields = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "customer_state",
        "item_count",
        "item_price_total",
        "item_freight_total",
        "late_delivery",
    ]
    return {field: json_safe(record.get(field)) for field in fields}


@app.get("/api/orders/{order_id}/risk", response_model=PredictionResponse)
def get_order_risk(order_id: str, stage: str = Query("A", pattern="^[ABC]$")):
    _record(order_id)
    try:
        predictions = load_test_predictions()
    except FileNotFoundError:
        return _predict(order_id, stage)
    match = predictions.loc[
        (predictions["order_id"] == order_id) & (predictions["stage"] == stage)
    ]
    if not match.empty:
        saved = match.iloc[0]
        probability = float(saved["delay_probability"])
        threshold = float(saved["threshold"])
        category = "high" if probability >= threshold else "medium" if probability >= threshold * 0.5 else "low"
        return PredictionResponse(
            order_id=order_id,
            stage=stage,
            delay_probability=probability,
            risk_category=category,
            threshold=threshold,
        )
    return _predict(order_id, stage)


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return _predict(request.order_id, request.stage)


@app.get("/api/metrics")
def metrics():
    try:
        return load_report("model_comparison.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics/stages")
def stage_metrics():
    try:
        return load_report("stage_metrics.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics/calibration")
def calibration():
    try:
        return load_report("calibration.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics/cost-analysis")
def cost_analysis():
    try:
        return load_report("cost_analysis.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics/feature-importance")
def feature_importance():
    try:
        return load_report("feature_importance.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics/risk-distribution")
def risk_distribution(stage: str = Query("A", pattern="^[ABC]$")):
    try:
        predictions = load_test_predictions()
        stage_metrics = load_report("stage_metrics.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    stage_result = next((item for item in stage_metrics if item["stage"] == stage), None)
    if stage_result is None:
        raise HTTPException(status_code=404, detail=f"No evaluation results are available for stage {stage}.")
    probabilities = predictions.loc[predictions["stage"] == stage, "delay_probability"]
    threshold = float(stage_result["threshold"])
    counts = {
        "low": int((probabilities < threshold * 0.5).sum()),
        "medium": int(((probabilities >= threshold * 0.5) & (probabilities < threshold)).sum()),
        "high": int((probabilities >= threshold).sum()),
    }
    return {
        "stage": stage,
        "threshold": threshold,
        "sample_count": int(len(probabilities)),
        "bins": [{"category": category, "count": count} for category, count in counts.items()],
    }
