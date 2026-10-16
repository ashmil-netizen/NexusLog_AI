import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import {
  Activity, AlertTriangle, ArrowUpRight, Boxes, ChartNoAxesCombined,
  ChevronLeft, ChevronRight, CircleHelp, Gauge, LayoutDashboard,
  LoaderCircle, PackageSearch, RefreshCw, Search, ShieldCheck, Truck,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import api, { errorMessage } from "./api";

const navItems = [
  ["/", "Dashboard", LayoutDashboard],
  ["/risk", "Risk analysis", Gauge],
  ["/orders", "Orders explorer", PackageSearch],
  ["/evaluation", "Model evaluation", ChartNoAxesCombined],
];

function useApi(path, params) {
  const [state, setState] = useState({ data: null, loading: true, error: "" });
  const key = JSON.stringify(params || {});
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setState({ data: null, loading: true, error: "" });
    api.get(path, { params }).then(({ data }) => {
      if (active) setState({ data, loading: false, error: "" });
    }).catch((error) => {
      if (active) setState({ data: null, loading: false, error: errorMessage(error) });
    });
    return () => { active = false; };
  }, [path, key, retry]);
  return { ...state, retry: () => setRetry((value) => value + 1) };
}

function Panel({ title, subtitle, action, children, className = "" }) {
  return <section className={`panel ${className}`}><div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}

function Loading() { return <div className="state"><LoaderCircle className="spin" size={20} /> Loading verified project data…</div>; }
function ErrorState({ error, retry }) { return <div className="state error-state"><AlertTriangle size={20} /><span>{error}</span><button className="button subtle" onClick={retry}><RefreshCw size={15} /> Retry</button></div>; }
function MetricCard({ title, value, hint, icon: Icon, trend }) {
  return <div className="metric-card"><div className="metric-top"><span>{title}</span><span className="metric-icon"><Icon size={18} /></span></div><strong>{value}</strong><div className="metric-hint">{trend && <span className="trend"><ArrowUpRight size={14} /></span>}{hint}</div></div>;
}
function StageChart({ stages = [] }) {
  const data = stages.map((item) => ({ stage: `Stage ${item.stage}`, "PR-AUC": item.pr_auc, "Brier score": item.brier_score }));
  if (!data.length) return <EmptyState text="No evaluation results available yet." />;
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} barGap={8}><CartesianGrid strokeDasharray="3 3" stroke="#242c40" vertical={false} /><XAxis dataKey="stage" stroke="#8791a8" /><YAxis stroke="#8791a8" domain={[0, 1]} /><Tooltip contentStyle={{ background: "#121a2c", border: "1px solid #29334b", borderRadius: 10 }} /><Bar dataKey="PR-AUC" fill="#9580ff" radius={[5, 5, 0, 0]} /><Bar dataKey="Brier score" fill="#35c9bb" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div>;
}
function RiskDistributionChart({ bins = [] }) {
  const colors = { low: "#35c9bb", medium: "#e6b95f", high: "#ff8294" };
  if (!bins.length) return <EmptyState text="No predicted-risk distribution is available." />;
  return <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={bins} barSize={45}><CartesianGrid strokeDasharray="3 3" stroke="#242c40" vertical={false} /><XAxis dataKey="category" stroke="#8791a8" /><YAxis stroke="#8791a8" /><Tooltip contentStyle={{ background: "#121a2c", border: "1px solid #29334b", borderRadius: 10 }} /><Bar dataKey="count" radius={[5, 5, 0, 0]}>{bins.map((item) => <Cell key={item.category} fill={colors[item.category]} />)}</Bar></BarChart></ResponsiveContainer></div>;
}
function EmptyState({ text }) { return <div className="empty"><CircleHelp size={18} /><span>{text}</span></div>; }
function Header() { return <header className="topbar"><div className="brand"><div className="brand-mark"><Boxes size={20} /></div><span>NexusLog <b>AI</b></span></div><div className="top-status"><span className="live-dot" /> Research workspace <span className="divider" /> <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API documentation</a></div></header>; }
function Layout() {
  return <div className="app-shell"><Header /><aside className="sidebar"><div className="workspace-label">WORKSPACE</div>{navItems.map(([to, label, Icon]) => <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}><Icon size={18} /><span>{label}</span>{to === "/risk" && <span className="nav-new">AI</span>}</NavLink>)}<div className="sidebar-bottom"><div className="source-chip"><ShieldCheck size={17} /><span><b>Research mode</b><small>Historical simulation</small></span></div><p>Olist Brazilian E-Commerce<br />Leakage-audited study</p></div></aside><main className="main-content"><Routes><Route path="/" element={<Dashboard />} /><Route path="/risk" element={<RiskAnalysis />} /><Route path="/orders" element={<OrdersExplorer />} /><Route path="/evaluation" element={<Evaluation />} /></Routes><footer>Predictions are retrospective estimates, not delivery guarantees.</footer></main></div>;
}

function PageTitle({ eyebrow, title, description, icon: Icon }) {
  return <div className="page-title"><div><div className="eyebrow"><Icon size={14} /> {eyebrow}</div><h1>{title}</h1><p>{description}</p></div><span className="research-tag"><ShieldCheck size={14} /> Research dataset</span></div>;
}

function Dashboard() {
  const summary = useApi("/summary");
  const highRisk = useApi("/orders/high-risk", { stage: "A", limit: 6 });
  const riskDistribution = useApi("/metrics/risk-distribution", { stage: "A" });
  if (summary.loading) return <Loading />;
  if (summary.error) return <ErrorState error={summary.error} retry={summary.retry} />;
  const value = summary.data;
  const stages = Array.isArray(value.stage_metrics) ? value.stage_metrics : [];
  const percent = value.delay_rate == null ? "—" : `${(value.delay_rate * 100).toFixed(1)}%`;
  const best = stages.length ? stages.reduce((a, b) => a.pr_auc >= b.pr_auc ? a : b) : null;
  return <><PageTitle eyebrow="OVERVIEW / DELIVERY INTELLIGENCE" title="Good morning, analyst" description="Monitor delivery performance and explore stage-aware delay risk across the historical Olist dataset." icon={Activity} />
    <div className="metric-grid"><MetricCard title="Eligible orders" value={value.eligible_orders.toLocaleString()} hint="Delivered orders analyzed" icon={Boxes} /><MetricCard title="Historical delay rate" value={percent} hint={`${(value.delayed_orders || 0).toLocaleString()} delayed deliveries`} icon={Truck} /><MetricCard title="High-risk orders" value={value.high_risk_order_count == null ? "—" : value.high_risk_order_count.toLocaleString()} hint={value.high_risk_threshold == null ? "No evaluation available" : `Stage A test set · ≥ ${(value.high_risk_threshold * 100).toFixed(1)}% threshold`} icon={AlertTriangle} /><MetricCard title="Best model PR-AUC" value={best ? best.pr_auc.toFixed(3) : "—"} hint={best ? `Stage ${best.stage} · ${best.selected_model} · test set` : "No evaluation available"} icon={Gauge} trend={Boolean(best)} /></div>
    <Panel title="Delay-risk distribution" subtitle={riskDistribution.data ? `Stage A · ${riskDistribution.data.sample_count.toLocaleString()} held-out test predictions · threshold ${(riskDistribution.data.threshold * 100).toFixed(1)}%` : "Historical model-estimated probabilities"}>{riskDistribution.loading ? <Loading /> : riskDistribution.error ? <ErrorState error={riskDistribution.error} retry={riskDistribution.retry} /> : <RiskDistributionChart bins={riskDistribution.data.bins} />}</Panel>
    <div className="content-grid"><Panel title="Stage performance and calibration" subtitle="Held-out PR-AUC and Brier score; calibration curves are available in model evaluation"><StageChart stages={stages} /></Panel><Panel title="Research protocol" subtitle="Three milestone-specific prediction snapshots"><div className="stage-list">{[["A", "Order placement", "Checkout and purchase details"], ["B", "Payment approval", "Payment and approval timing"], ["C", "Carrier handover", "Elapsed time to carrier handoff"]].map(([stage, name, detail]) => <div className="stage-row" key={stage}><span className="stage-code">{stage}</span><span><b>{name}</b><small>{detail}</small></span><span className="stage-arrow">→</span></div>)}</div><Link to="/evaluation" className="text-link">View model evaluation <ChevronRight size={15} /></Link></Panel></div>
    <Panel title="High-risk historical orders" subtitle="Stage A test predictions · actual outcome shown separately"><HighRiskTable orders={highRisk.data?.items || []} loading={highRisk.loading} error={highRisk.error} retry={highRisk.retry} /></Panel>
  </>;
}

function OrdersTable({ orders, loading, error, retry }) {
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={retry} />;
  if (!orders.length) return <EmptyState text="No order data available. Prepare the dataset first." />;
  return <div className="table-wrap"><table><thead><tr><th>ORDER ID</th><th>PURCHASE DATE</th><th>REGION</th><th>ITEMS</th><th>OUTCOME</th><th /></tr></thead><tbody>{orders.map((order) => <tr key={order.order_id}><td><Link className="order-id" to={`/risk?order=${encodeURIComponent(order.order_id)}`}>{order.order_id.slice(0, 14)}…</Link></td><td>{new Date(order.order_purchase_timestamp).toLocaleDateString()}</td><td>{order.customer_state || "—"}</td><td>{order.item_count ?? "—"}</td><td><span className="badge danger">Delayed</span></td><td><Link className="icon-link" to={`/risk?order=${encodeURIComponent(order.order_id)}`}><ChevronRight size={16} /></Link></td></tr>)}</tbody></table></div>;
}

function HighRiskTable({ orders, loading, error, retry }) {
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} retry={retry} />;
  if (!orders.length) return <EmptyState text="No saved Stage A test predictions crossed the alert threshold." />;
  return <div className="table-wrap"><table><thead><tr><th>ORDER ID</th><th>PURCHASE DATE</th><th>REGION</th><th>MODEL RISK</th><th>ACTUAL HISTORICAL OUTCOME</th><th /></tr></thead><tbody>{orders.map((order) => <tr key={order.order_id}><td><Link className="order-id" to={`/risk?order=${encodeURIComponent(order.order_id)}`}>{order.order_id.slice(0, 14)}…</Link></td><td>{new Date(order.order_purchase_timestamp).toLocaleDateString()}</td><td>{order.customer_state || "—"}</td><td><span className="badge high">{(order.delay_probability * 100).toFixed(1)}% · high</span></td><td><span className={`badge ${order.actual_late_delivery ? "danger" : "low"}`}>{order.actual_late_delivery ? "Delayed" : "On time"}</span></td><td><Link className="icon-link" to={`/risk?order=${encodeURIComponent(order.order_id)}`}><ChevronRight size={16} /></Link></td></tr>)}</tbody></table></div>;
}

function OrdersExplorer() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("");
  const [region, setRegion] = useState("");
  const [sort, setSort] = useState("order_purchase_timestamp:desc");
  const [page, setPage] = useState(1);
  const [sortBy, sortOrder] = sort.split(":");
  const params = {
    page,
    page_size: 15,
    sort_by: sortBy,
    sort_order: sortOrder,
    ...(query ? { q: query } : {}),
    ...(filter !== "" ? { late_delivery: filter } : {}),
    ...(region ? { customer_state: region.toUpperCase() } : {}),
  };
  const result = useApi("/orders", params);
  return <><PageTitle eyebrow="DATA / ORDER EXPLORER" title="Orders explorer" description="Browse eligible historical orders with paginated search and ground-truth delivery outcomes." icon={PackageSearch} /><Panel title="Order records" subtitle={result.data ? `${result.data.total.toLocaleString()} matching records` : "Search, filter, and sort the prepared dataset"}><div className="table-controls"><label className="search-box"><Search size={17} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="Search by order ID" /></label><input className="filter-input" aria-label="Filter by customer state" inputMode="text" maxLength="2" value={region} onChange={(event) => { setRegion(event.target.value.slice(0, 2)); setPage(1); }} placeholder="State code" /><select aria-label="Filter by historical outcome" value={filter} onChange={(event) => { setFilter(event.target.value); setPage(1); }}><option value="">All outcomes</option><option value="1">Delayed</option><option value="0">On time</option></select><select aria-label="Sort orders" value={sort} onChange={(event) => { setSort(event.target.value); setPage(1); }}><option value="order_purchase_timestamp:desc">Newest purchase</option><option value="order_purchase_timestamp:asc">Oldest purchase</option><option value="item_count:desc">Most items</option><option value="item_price_total:desc">Highest value</option></select></div>{result.loading ? <Loading /> : result.error ? <ErrorState error={result.error} retry={result.retry} /> : <><OrdersTable orders={result.data.items} loading={false} /><div className="pagination"><span>Page {result.data.page} · {result.data.total.toLocaleString()} orders</span><div><button className="button subtle" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}><ChevronLeft size={16} /> Previous</button><button className="button subtle" disabled={page * result.data.page_size >= result.data.total} onClick={() => setPage((p) => p + 1)}>Next <ChevronRight size={16} /></button></div></div></>}</Panel></>;
}

function RiskAnalysis() {
  const initial = new URLSearchParams(window.location.search).get("order") || "";
  const [orderId, setOrderId] = useState(initial);
  const [stage, setStage] = useState("A");
  const [result, setResult] = useState(null);
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  async function submit(event) {
    event.preventDefault();
    if (!orderId.trim()) return;
    setLoading(true); setError(""); setResult(null); setDetails(null);
    try {
      const [prediction, order] = await Promise.all([
        api.post("/predict", { order_id: orderId.trim(), stage }),
        api.get(`/orders/${encodeURIComponent(orderId.trim())}`),
      ]);
      setResult(prediction.data); setDetails(order.data);
    }
    catch (requestError) { setError(errorMessage(requestError)); }
    finally { setLoading(false); }
  }
  return <><PageTitle eyebrow="ANALYSIS / STAGE PREDICTION" title="Order risk analysis" description="Estimate delay probability using only information available at the selected historical lifecycle stage." icon={Gauge} /><div className="content-grid risk-grid"><Panel title="Run a prediction" subtitle="Historical stage-aware simulation"><form className="risk-form" onSubmit={submit}><label>Order ID<input required value={orderId} onChange={(event) => setOrderId(event.target.value)} placeholder="Paste an eligible Olist order ID" /></label><label>Prediction stage<select value={stage} onChange={(event) => setStage(event.target.value)}><option value="A">A — Order placement</option><option value="B">B — Payment approval</option><option value="C">C — Carrier handover</option></select></label><button className="button primary" disabled={loading}>{loading ? <LoaderCircle className="spin" size={16} /> : <Activity size={16} />}{loading ? "Scoring…" : "Estimate delay risk"}</button></form>{error && <div className="inline-error"><AlertTriangle size={17} />{error}</div>}</Panel><Panel title="Prediction result" subtitle="Estimated probability · not a guarantee">{result ? <div className="prediction-result"><span className={`risk-orb ${result.risk_category}`}>{(result.delay_probability * 100).toFixed(0)}%</span><span className={`badge ${result.risk_category}`}>{result.risk_category} risk</span><p>Stage {result.stage} · alert threshold {(result.threshold * 100).toFixed(0)}%</p><small>Retrospective stage simulation. Per-order feature contributions are not computed; see aggregate permutation importance on Model Evaluation. Any actual historical outcome below is ground truth, not this prediction.</small></div> : <EmptyState text="Enter an order ID to view a model prediction." />}</Panel></div>{details && <Panel title="Historical order details" subtitle="Source-record information · actual outcome is ground truth, not the model prediction"><div className="details-grid"><div><small>ORDER STATUS</small><b>{details.order_status || "—"}</b></div><div><small>REGION</small><b>{details.customer_state || "—"}</b></div><div><small>ITEM COUNT</small><b>{details.item_count ?? "—"}</b></div><div><small>PURCHASE DATE</small><b>{new Date(details.order_purchase_timestamp).toLocaleString()}</b></div><div><small>PAYMENT APPROVAL</small><b>{details.order_approved_at ? new Date(details.order_approved_at).toLocaleString() : "Not available"}</b></div><div><small>CARRIER HANDOVER</small><b>{details.order_delivered_carrier_date ? new Date(details.order_delivered_carrier_date).toLocaleString() : "Not available"}</b></div><div><small>ACTUAL HISTORICAL OUTCOME</small><b className={details.late_delivery ? "outcome-late" : "outcome-ontime"}>{details.late_delivery ? "Delayed" : "On time"}</b></div></div></Panel>}<div className="notice"><ShieldCheck size={17} /><span><b>Feature availability matters.</b> Stage B requires a payment-approval timestamp; Stage C requires a carrier-handover timestamp. These historical estimates are not a live tracking feed. Actual historical outcome above is ground truth and is separate from predicted risk.</span></div></>;
}

function Evaluation() {
  const metrics = useApi("/metrics/stages");
  const comparisons = useApi("/metrics");
  const costs = useApi("/metrics/cost-analysis");
  const importance = useApi("/metrics/feature-importance");
  const calibration = useApi("/metrics/calibration");
  if (metrics.loading) return <Loading />;
  if (metrics.error) return <><PageTitle eyebrow="RESEARCH / MODEL QUALITY" title="Model evaluation" description="Test-period results, calibration, and cost-sensitive alert analysis." icon={ChartNoAxesCombined} /><ErrorState error={metrics.error} retry={metrics.retry} /></>;
  const stages = metrics.data || [];
  const calibrationSeries = calibration.data || [];
  const costMap = new Map();
  (costs.data || []).forEach((stage) => stage.thresholds.forEach((point) => {
    const row = costMap.get(point.threshold) || { threshold: point.threshold };
    row[`stage_${stage.stage}`] = point.total_cost;
    costMap.set(point.threshold, row);
  }));
  const costRows = [...costMap.values()].sort((a, b) => a.threshold - b.threshold);
  const importanceRows = (importance.data || []).flatMap((row) => row.features.slice(0, 5).map((item) => ({ name: `${row.stage}: ${item.feature}`, value: item.importance_mean })));
  return <><PageTitle eyebrow="RESEARCH / MODEL QUALITY" title="Model evaluation" description="Measured performance on the chronologically held-out final test period." icon={ChartNoAxesCombined} /><div className="metric-grid compact">{stages.map((item) => <MetricCard key={item.stage} title={`Stage ${item.stage} · PR-AUC`} value={item.pr_auc.toFixed(3)} hint={`Brier ${item.brier_score.toFixed(3)} · ${item.test_count.toLocaleString()} test orders`} icon={Gauge} />)}</div>
    <Panel title="Stage comparison" subtitle="Higher PR-AUC and lower Brier score are preferred"><StageChart stages={stages} /></Panel>
    <Panel title="Evaluation periods and sample counts" subtitle="Chronological partitions by stage"><div className="table-wrap"><table><thead><tr><th>STAGE</th><th>TRAIN N</th><th>VALIDATION N</th><th>FINAL TEST N</th><th>TRAIN PERIOD</th><th>VALIDATION PERIOD</th><th>TEST PERIOD</th></tr></thead><tbody>{stages.map((row) => <tr key={`period-${row.stage}`}><td>Stage {row.stage}</td><td>{row.train_count.toLocaleString()}</td><td>{row.validation_count.toLocaleString()}</td><td>{row.test_count.toLocaleString()}</td><td>{new Date(row.train_period.start).toLocaleDateString()} – {new Date(row.train_period.end).toLocaleDateString()}</td><td>{new Date(row.validation_period.start).toLocaleDateString()} – {new Date(row.validation_period.end).toLocaleDateString()}</td><td>{new Date(row.test_period.start).toLocaleDateString()} – {new Date(row.test_period.end).toLocaleDateString()}</td></tr>)}</tbody></table></div></Panel>
    <div className="content-grid"><Panel title="Calibration" subtitle="Final test set · quantile-bin probability vs. observed delay rate">{calibration.loading ? <Loading /> : calibration.error ? <ErrorState error={calibration.error} retry={calibration.retry} /> : calibrationSeries.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart><CartesianGrid strokeDasharray="3 3" stroke="#242c40" /><XAxis dataKey="mean_predicted_probability" type="number" domain={[0, 1]} name="Predicted probability" stroke="#8791a8" /><YAxis dataKey="observed_fraction" type="number" domain={[0, 1]} name="Observed delay rate" stroke="#8791a8" /><Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#121a2c", border: "1px solid #29334b" }} /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#748099" strokeDasharray="5 5" />{calibrationSeries.map((row, index) => <Scatter key={row.stage} name={`Stage ${row.stage}`} data={row.mean_predicted_probability.map((probability, bin) => ({ mean_predicted_probability: probability, observed_fraction: row.observed_fraction[bin] }))} fill={["#9580ff", "#35c9bb", "#ffbb68"][index % 3]} />)}</ScatterChart></ResponsiveContainer></div> : <EmptyState text="Calibration results are unavailable." />}</Panel><Panel title="Alert cost by threshold" subtitle="Final-test costs shown for thresholds selected on validation">{costs.loading ? <Loading /> : costs.error ? <ErrorState error={costs.error} retry={costs.retry} /> : costRows.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={costRows}><CartesianGrid strokeDasharray="3 3" stroke="#242c40" /><XAxis dataKey="threshold" stroke="#8791a8" /><YAxis stroke="#8791a8" /><Tooltip contentStyle={{ background: "#121a2c", border: "1px solid #29334b" }} /><Line dataKey="stage_A" name="Stage A" stroke="#9580ff" dot={false} /><Line dataKey="stage_B" name="Stage B" stroke="#35c9bb" dot={false} /><Line dataKey="stage_C" name="Stage C" stroke="#ffbb68" dot={false} /></LineChart></ResponsiveContainer></div> : <EmptyState text="Cost-analysis results are unavailable." />}</Panel></div>
    <div className="content-grid"><Panel title="Model comparison" subtitle="Validation-period candidate models">{comparisons.loading ? <Loading /> : comparisons.error ? <ErrorState error={comparisons.error} retry={comparisons.retry} /> : <div className="table-wrap"><table><thead><tr><th>STAGE</th><th>MODEL</th><th>VALIDATION PR-AUC</th><th>PRECISION</th><th>RECALL</th><th>BRIER</th></tr></thead><tbody>{(comparisons.data || []).map((row) => <tr key={`${row.stage}-${row.model}`}><td>Stage {row.stage}</td><td>{row.model}</td><td>{row.validation_pr_auc.toFixed(3)}</td><td>{row.precision.toFixed(3)}</td><td>{row.recall.toFixed(3)}</td><td>{row.brier_score.toFixed(3)}</td></tr>)}</tbody></table></div>}</Panel><Panel title="Feature importance" subtitle="Permutation importance · validation PR-AUC">{importance.loading ? <Loading /> : importance.error ? <ErrorState error={importance.error} retry={importance.retry} /> : importanceRows.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={importanceRows} layout="vertical" margin={{ left: 48 }}><CartesianGrid strokeDasharray="3 3" stroke="#242c40" horizontal={false} /><XAxis type="number" stroke="#8791a8" /><YAxis type="category" dataKey="name" width={120} stroke="#8791a8" tick={{ fontSize: 10 }} /><Tooltip contentStyle={{ background: "#121a2c", border: "1px solid #29334b" }} /><Bar dataKey="value" fill="#9580ff" radius={[0, 4, 4, 0]}>{importanceRows.map((row, index) => <Cell key={row.name} fill={["#9580ff", "#8171e6", "#6d63c9"][index % 3]} />)}</Bar></BarChart></ResponsiveContainer></div> : <EmptyState text="Feature-importance results are unavailable." />}</Panel></div>
    <Panel title="Final test confusion matrices" subtitle="Rows: actual outcome · columns: predicted outcome at the validation-selected threshold"><div className="confusion-grid">{stages.map((row) => <div className="confusion-card" key={row.stage}><b>Stage {row.stage}</b><div className="confusion-matrix"><span /><span>Predicted on time</span><span>Predicted delayed</span><b>Actual on time</b><span>{row.confusion_matrix[0][0].toLocaleString()} TN</span><span>{row.confusion_matrix[0][1].toLocaleString()} FP</span><b>Actual delayed</b><span>{row.confusion_matrix[1][0].toLocaleString()} FN</span><span>{row.confusion_matrix[1][1].toLocaleString()} TP</span></div></div>)}</div></Panel><div className="notice"><AlertTriangle size={17} /><span>Threshold costs are hypothetical unless operational costs are provided. Test-period plots evaluate a threshold chosen on validation data; they do not retune it. Feature importance does not establish causality.</span></div>
  </>;
}

export default function App() { return <Layout />; }
