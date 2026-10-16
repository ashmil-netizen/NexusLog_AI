# Methodology

## Objective and target

This retrospective study evaluates whether Olist orders that were delivered later than the estimated customer delivery timestamp can be identified at three lifecycle milestones. The target is `1` only when `order_delivered_customer_date > order_estimated_delivery_date`; equality is on time. Canceled orders, non-delivered orders, missing actual/estimated delivery timestamps, and timestamp-invalid delivered records are excluded from the primary classification cohort and counted in the generated data audit.

## Stages and availability assumptions

- **A — Order placement:** `purchase_month`, `purchase_dayofweek`, `purchase_hour`, `customer_state`, `item_count`, `item_price_total`, `item_freight_total`, and `seller_count`. Item and seller features are aggregate checkout line items. These are only defensible at placement if the simulated prediction is made after checkout contents are known.
- **B — Payment approval:** Stage A plus `approval_elapsed_hours`, `payment_type`, `payment_installments`, and `payment_value_total`. A row is eligible only when `order_approved_at` exists.
- **C — Carrier handover:** Stage B plus `handover_elapsed_hours`. A row is eligible only when payment approval and `order_delivered_carrier_date` exist.

Actual customer delivery timestamp, estimated delivery timestamp, delivery duration, review information, and the target are never model inputs. Olist is historical data, not a streaming feed. The analysis does not imply these timestamps or fields are available in a production system without verification.

## Aggregation and leakage controls

Items and payments are aggregated by order before joining. Joins validate expected cardinality; conflicting duplicate order IDs stop preparation instead of being silently resolved. Only delivered orders with valid target dates are modeled. Missing required target timestamps, delivery/estimate dates before purchase, and present approval/handover timestamps before purchase or after actual delivery are excluded and counted in the audit. Missing later milestones do not disqualify a row from Stage A. Stage feature allowlists are explicit and shared by training and inference.

## Temporal evaluation

Orders are sorted by purchase time and divided into the oldest 70% training, following 15% validation, and newest 15% final test. All preprocessing is fit within the training pipeline. Validation selects between logistic regression and random forest, fits sigmoid calibration on the chosen training-fitted estimator, and selects a threshold. The test period is evaluated only after these choices. Repeated customers/sellers may span periods; that is a limitation, and this split measures temporal rather than unseen-entity generalization.

## Calibration, costs, and metrics

Sigmoid probability calibration is fitted on validation data. Test-set calibration curves use ten quantile bins and are evaluation-only. Brier score, PR-AUC, precision, recall, accuracy, and confusion matrix are reported. The alert threshold minimizes validation false-positive/false-negative cost using configurable illustrative costs (defaults: 1 and 3). Reported test costs do not feed back into threshold selection. Calibrated probabilities remain estimates, not guarantees.

Permutation importance is computed on validation data using PR-AUC. It indicates predictive association, not causation.
