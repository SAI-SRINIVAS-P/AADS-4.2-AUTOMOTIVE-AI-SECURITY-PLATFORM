from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, SIGNAL_COLUMNS, TARGET_COLUMN,
                      load_dataset, validate_frame, has_binary_classes)
from src.explain import explain_record
from src.models import evaluate_models, fit_detector, predict_frame
from src.features import TEMPORAL_FEATURE_COLUMNS
from src.monitoring import (get_alert_history, init_db, log_alert_batch,
                            monitoring_summary, save_feedback)
from src.quality import data_quality_report, drift_report
from src.registry import (ACTIVE_BASELINE, ACTIVE_META, ACTIVE_MODEL, ARTIFACTS, active_model_metadata,
                          list_registry, promote_candidate, save_experiment)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AADS | Automotive AI Security", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded", menu_items={
                       "About": "AADS is an automotive telemetry ML research platform. It is not a certified vehicle safety system."
                   })
init_db()

# Product-like design system: deep graphite, electric cyan, amber risk accents, restrained gradients.
st.markdown(r"""
<style>
:root { --aads-cyan:#5ce1e6; --aads-cyan-dim:#1f8f9b; --aads-amber:#ffbd69; --aads-red:#ff667d; }
.stApp { background: radial-gradient(ellipse at 88% 0%, rgba(35,132,152,.13), transparent 32%), #0b1018; }
[data-testid="stHeader"] { background: rgba(11,16,24,.78); }
.block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1540px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111b28 0%,#10141d 100%); border-right:1px solid rgba(113,224,235,.12); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height:1.45; }
[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(25,42,58,.78),rgba(15,22,33,.78)); border:1px solid rgba(119,200,220,.15); padding:16px 18px; border-radius:15px; min-height:108px; }
[data-testid="stMetricLabel"] { color:#9eb2c8; font-size:.86rem; }
[data-testid="stMetricValue"] { color:#edf7ff; font-weight:650; letter-spacing:-.035em; }
[data-testid="stMetricDelta"] { font-size:.8rem; }
[data-testid="stTabs"] button { font-weight:600; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:650; border:1px solid rgba(92,225,230,.32); transition:all .15s ease; }
.stButton button[kind="primary"] { background:linear-gradient(105deg,#36cad5,#5ce1e6); color:#07141b; border:none; }
.stButton button:hover, .stDownloadButton button:hover { border-color:#5ce1e6; transform:translateY(-1px); }
[data-testid="stDataFrame"] { border:1px solid rgba(141,173,198,.14); border-radius:12px; overflow:hidden; }
hr { border-color:rgba(141,173,198,.16); }
.aads-hero { position:relative; overflow:hidden; border:1px solid rgba(92,225,230,.24); border-radius:22px; padding:26px 30px; background:linear-gradient(112deg,rgba(21,45,63,.98),rgba(13,26,39,.93) 56%,rgba(22,45,56,.95)); box-shadow:0 20px 70px rgba(0,0,0,.17); margin-bottom:18px; }
.aads-hero:after { content:""; position:absolute; width:260px;height:260px;border:1px solid rgba(92,225,230,.13);border-radius:50%;right:-70px;top:-135px;box-shadow:0 0 0 25px rgba(92,225,230,.025),0 0 0 55px rgba(92,225,230,.02); }
.aads-kicker { color:#5ce1e6; text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:800; margin-bottom:8px; }
.aads-title { font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.1; color:#f1f8ff; font-weight:780; letter-spacing:-.045em; margin:0; }
.aads-subtitle { color:#a9c1d5; font-size:.96rem; margin-top:10px; max-width:820px; }
.aads-chip { display:inline-block; background:rgba(92,225,230,.1); border:1px solid rgba(92,225,230,.22); border-radius:999px; padding:5px 10px; margin:4px 6px 0 0; font-size:.76rem; color:#b5f7f8; }
.section-eyebrow { color:#5ce1e6; font-size:.7rem; text-transform:uppercase; font-weight:800; letter-spacing:.16em; margin-bottom:4px; }
.section-title { color:#eef7ff; font-size:1.35rem; font-weight:750; letter-spacing:-.025em; margin-bottom:.25rem; }
.section-note { color:#90a5ba; font-size:.88rem; margin-bottom:1rem; }
.status-pill { display:inline-block; border-radius:999px; border:1px solid rgba(92,225,230,.24); color:#8debf0; background:rgba(92,225,230,.07); font-size:.75rem; font-weight:700; padding:5px 10px; }
.sidebar-brand { padding:8px 4px 16px 4px; }
.sidebar-mark { font-size:1.6rem; font-weight:900; color:#f5fbff; letter-spacing:-.04em; }
.sidebar-mark span { color:#5ce1e6; }
.sidebar-small { color:#8ba3ba; font-size:.77rem; margin-top:2px; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, max_entries=3)
def load_bundled() -> tuple[pd.DataFrame, list[str]]:
    raw = load_dataset(ROOT / "compressed_data.csv.gz")
    return validate_frame(raw)

@st.cache_data(show_spinner=False, max_entries=2)
def load_uploaded(content: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    compression = "gzip" if filename.lower().endswith(".gz") else None
    raw = pd.read_csv(io.BytesIO(content), compression=compression, low_memory=False)
    return validate_frame(raw)


def section(kicker: str, title: str, note: str | None = None):
    st.markdown(f'<div class="section-eyebrow">{kicker}</div><div class="section-title">{title}</div>' +
                (f'<div class="section-note">{note}</div>' if note else ""), unsafe_allow_html=True)


def fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{float(v):.1%}"

# ---------- Sidebar / current dataset ----------
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="sidebar-mark">AADS<span> / </span>OPS</div><div class="sidebar-small">Automotive AI Security Platform</div></div>', unsafe_allow_html=True)
    st.markdown('<span class="status-pill">● LOCAL RESEARCH INSTANCE</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["Mission Control", "Data Lab", "Model Studio", "Alert Desk",
                                   "Model Registry", "Monitoring", "API & Ops"], label_visibility="visible")
    st.divider()
    st.markdown("**TELEMETRY SOURCE**")
    source = st.radio("Data source", ["Bundled dataset", "Upload CSV"], label_visibility="collapsed")
    uploaded = st.file_uploader("Choose CSV / CSV.GZ", type=["csv", "gz"]) if source == "Upload CSV" else None
    st.caption("Supports AADS fields, CAN ID + hexadecimal DATA payloads, and per-byte DATA[0..3] schemas.")
    st.divider()
    st.markdown("**EXPERIMENT SETTINGS**")
    contamination = st.slider("Expected outlier fraction", .01, .40, .10, .01,
                              help="Isolation Forest threshold assumption. This is not the true attack rate.")
    holdout = st.slider("Holdout fraction", .10, .35, .20, .05,
                        help="The final time-ordered portion reserved for supervised evaluation where possible.")
    max_train_rows = st.select_slider("Max model training rows", options=[10_000, 25_000, 50_000, 100_000], value=50_000)
    max_score_rows = st.select_slider("Max rows per scoring run", options=[10_000, 25_000, 50_000, 100_000, 250_000], value=50_000)
    st.caption("Limits protect laptop memory and keep the interface responsive.")

try:
    if source == "Upload CSV":
        if uploaded is None:
            st.markdown('<div class="aads-hero"><div class="aads-kicker">Telemetry intelligence · setup</div><div class="aads-title">Bring your vehicle data into focus.</div><div class="aads-subtitle">Upload a CSV to start data validation, feature profiling, model experiments, and alert review.</div></div>', unsafe_allow_html=True)
            st.info("Choose **Upload CSV** in the sidebar and select a supported file. Or switch to Bundled dataset to explore the included 1M-row sample.")
            st.stop()
        valid, schema_notes = load_uploaded(uploaded.getvalue(), uploaded.name)
        dataset_name = uploaded.name
    else:
        valid, schema_notes = load_bundled()
        dataset_name = "compressed_data.csv.gz"
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.markdown("Check that the file contains a CAN ID and payload byte fields or a hexadecimal DATA/Payload column.")
    st.stop()

dataset_key = f"{dataset_name}:{len(valid)}:{valid['Time'].iloc[0] if len(valid) else 'empty'}:{valid['Time'].iloc[-1] if len(valid) else 'empty'}"
if st.session_state.get("aads_dataset_key") != dataset_key:
    for _state_key in ["aads_results", "aads_run_id", "aads_scored", "aads_score_info", "aads_profile"]:
        st.session_state.pop(_state_key, None)
    st.session_state["aads_dataset_key"] = dataset_key

if st.session_state.get("aads_profile_key") != dataset_key:
    st.session_state["aads_profile"] = data_quality_report(valid)
    st.session_state["aads_profile_key"] = dataset_key
profile = st.session_state["aads_profile"]
labelled = has_binary_classes(valid)
valid_rows = valid[valid[TARGET_COLUMN].isin([0, 1])] if TARGET_COLUMN in valid else pd.DataFrame()
attack_rate = float(valid_rows[TARGET_COLUMN].mean()) if not valid_rows.empty else None
active_meta = active_model_metadata()
registry = list_registry()

# ---------- Mission Control ----------
if page == "Mission Control":
    st.markdown(f'''<div class="aads-hero"><div class="aads-kicker">Vehicle telemetry · anomaly intelligence · ml lifecycle</div>
    <div class="aads-title">See the signal.<br>Understand the risk.</div>
    <div class="aads-subtitle">A complete workbench for CAN telemetry quality, supervised attack detection, unsupervised anomaly scoring, explainability, model versioning, and operational monitoring.</div>
    <div style="margin-top:14px"><span class="aads-chip">DATA → FEATURES → MODELS</span><span class="aads-chip">EXPERIMENT REGISTRY</span><span class="aads-chip">ALERT FEEDBACK</span></div></div>''', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Telemetry records", f"{len(valid):,}")
    c2.metric("CAN message IDs", f"{profile['unique_ids']:,}")
    c3.metric("Signal fields", f"{len(SIGNAL_COLUMNS)}")
    c4.metric("Labelled attack rate", fmt_pct(attack_rate))
    c5.metric("Saved experiments", f"{len(registry):,}")
    for note in schema_notes:
        st.warning(note)
    st.markdown("")
    left,right = st.columns([1.4,1], gap="large")
    with left:
        section("Telemetry observability", "Signal pulse", "A deterministic sample of payload signals. Absolute timestamps are kept out of the classifier features.")
        plot_df = valid.sort_values("Time", kind="stable").iloc[np.linspace(0, len(valid)-1, min(2500,len(valid))).astype(int)].copy()
        plot_df["Record"] = np.arange(len(plot_df))
        long = plot_df.melt(id_vars="Record", value_vars=[c for c in SIGNAL_COLUMNS if c in plot_df],
                            var_name="Signal", value_name="Value").dropna(subset=["Value"])
        if not long.empty:
            fig = px.line(long, x="Record", y="Value", color="Signal", template="plotly_dark")
            fig.update_layout(height=345, margin=dict(l=8,r=8,t=10,b=8), legend_title_text="Payload", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="rgba(150,180,200,.12)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No usable numeric signal values to chart.")
    with right:
        section("Class context", "Label landscape", "Data labels are source ground truth only when their semantics are verified.")
        if not valid_rows.empty:
            counts = valid_rows[TARGET_COLUMN].value_counts().reindex([0,1], fill_value=0).rename_axis("Class").reset_index(name="Records")
            counts["Class"] = counts["Class"].map({0:"Normal",1:"Attack"})
            fig = px.pie(counts, names="Class", values="Records", hole=.66, template="plotly_dark", color="Class",
                         color_discrete_map={"Normal":"#51d7dc","Attack":"#ffbd69"})
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h",y=-.06))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No recognized binary labels. Use the unsupervised detector or inspect the schema.")
        colA,colB = st.columns(2)
        colA.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
        colB.metric("Missing cells", f"{profile['missing_cells']:,}")
        if active_meta:
            st.success(f"Active model: {active_meta.get('model_name')} · {active_meta.get('model_version')}")
        else:
            st.info("No active API model. Train through Model Studio or run `python -m scripts.train`.")
    st.divider()
    section("Lifecycle map", "All ten phases, one workspace")
    phases = [
        ("01", "Data quality", "Schema, missingness, duplicates"), ("02", "EDA", "Signal and class exploration"),
        ("03", "Feature engineering", "Stateless payload summaries"), ("04", "Experiments", "Random Forest / Extra Trees"),
        ("05", "Evaluation", "Temporal holdout, PR-AUC, FPR"), ("06", "Explainability", "Local feature ablation"),
        ("07", "Registry", "Versioned artifacts and promotion"), ("08", "Serving", "FastAPI prediction endpoints"),
        ("09", "Monitoring", "Alert history, feedback, drift"), ("10", "Delivery", "Tests, Docker, CI workflow")]
    cols = st.columns(5)
    for i,(number,name,desc) in enumerate(phases):
        with cols[i%5]:
            st.markdown(f'<div style="border:1px solid rgba(92,225,230,.16);border-radius:13px;background:rgba(18,31,44,.62);padding:12px;margin-bottom:12px;min-height:105px"><div style="color:#5ce1e6;font-size:.72rem;font-weight:800;letter-spacing:.1em">PHASE {number}</div><div style="font-weight:750;color:#edf7ff;margin:6px 0">{name}</div><div style="color:#8fa6bd;font-size:.78rem">{desc}</div></div>', unsafe_allow_html=True)

# ---------- Data Lab ----------
elif page == "Data Lab":
    section("Phase 01 / 02", "Data Lab", "Profile, audit and normalize data before asking a model to learn from it.")
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Rows", f"{profile['rows']:,}")
    q2.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    q3.metric("Total missing cells", f"{profile['missing_cells']:,}")
    q4.metric("Known CAN IDs", f"{profile['unique_ids']:,}")
    for note in schema_notes:
        st.info(note)
    for finding in profile["findings"]:
        st.warning(finding)
    tab_quality,tab_explore,tab_download = st.tabs(["Quality report", "EDA explorer", "Normalized export"])
    with tab_quality:
        miss = pd.DataFrame({"Feature": list(profile["signal_missing_fraction"].keys()),
                             "Missing %": [v*100 for v in profile["signal_missing_fraction"].values()]})
        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown("**Signal missingness**")
            if not miss.empty:
                fig=px.bar(miss,x="Feature",y="Missing %",template="plotly_dark",color="Missing %",color_continuous_scale=["#37cbd5","#ffbd69"])
                fig.update_layout(height=310,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,width="stretch")
        with c2:
            st.markdown("**Dataset integrity**")
            integrity = pd.DataFrame([
                {"Check":"Rows / columns", "Result":f"{profile['rows']:,} / {profile['columns']}"},
                {"Check":"Overall missing-cell rate", "Result":fmt_pct(profile['missing_cell_fraction'])},
                {"Check":"Time monotonic", "Result":"Unknown" if profile['time_ordered'] is None else ("Yes" if profile['time_ordered'] else "No")},
                {"Check":"Constant signals", "Result":", ".join(profile['constant_signals']) or "None detected"},
                {"Check":"Binary labels", "Result":"Available" if labelled else "Unavailable / not binary"},
            ])
            st.dataframe(integrity,hide_index=True,width="stretch")
        if profile["label_counts"]:
            st.markdown("**Label counts (normalized)**")
            st.json(profile["label_counts"])
    with tab_explore:
        left,right=st.columns(2)
        sample=valid.iloc[np.linspace(0,len(valid)-1,min(50000,len(valid))).astype(int)]
        with left:
            chosen_signal=st.selectbox("Signal to inspect",SIGNAL_COLUMNS)
            if chosen_signal in sample:
                fig=px.histogram(sample,x=chosen_signal,nbins=60,template="plotly_dark",title=f"Distribution · {chosen_signal}",color_discrete_sequence=["#5ce1e6"])
                fig.update_layout(height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,width="stretch")
        with right:
            ids=sample["ID"].value_counts().head(15).rename_axis("CAN ID").reset_index(name="Frames")
            fig=px.bar(ids,x="CAN ID",y="Frames",template="plotly_dark",title="Most frequent CAN IDs",color_discrete_sequence=["#5ce1e6"])
            fig.update_layout(height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,width="stretch")
        st.markdown("**Descriptive statistics**")
        st.dataframe(sample[SIGNAL_COLUMNS].describe().T,width="stretch")
    with tab_download:
        st.markdown("Export normalized data after schema mapping. This export preserves original columns plus canonical AADS columns where possible.")
        max_export = max(1, min(100000, len(valid)))
        if max_export <= 1000:
            rows_to_export = max_export
            st.caption(f"The dataset has {max_export:,} rows, so the full dataset can be exported.")
        else:
            step_export = max(1000, max_export // 100)
            rows_to_export=st.slider("Rows in normalized export", min_value=1000, max_value=max_export,
                                     value=min(max_export, 10000), step=step_export)
        st.download_button("Download normalized CSV", valid.head(rows_to_export).to_csv(index=False).encode("utf-8"),
                           file_name="aads_normalized_data.csv",mime="text/csv",type="primary")
        st.dataframe(valid.head(100),width="stretch",height=280)

# ---------- Model Studio ----------
elif page == "Model Studio":
    section("Phases 03 / 04 / 05", "Model Studio", "Train reproducible candidates, tune thresholds using validation data, and reserve a final temporal window for evaluation.")
    st.markdown("**Feature pipeline variants compared**")
    feature_cols = st.columns(4)
    feature_cols[0].metric("Raw signals", "4")
    feature_cols[1].metric("Payload summaries", "9")
    feature_cols[2].metric("Temporal features", str(len(TEMPORAL_FEATURE_COLUMNS)))
    feature_cols[3].metric("Absolute timestamp", "Excluded")
    st.caption("Temporal candidates use message-time gaps, per-CAN-ID signal deltas, recent ID frequency, and rolling statistics over up to five earlier same-ID messages inside a 100-message context window. The payload-only Extra Trees baseline omits temporal features to quantify their effect. Absolute timestamps and labels are not predictors. For single-record API requests, pass recent chronological history to populate temporal context.")
    if not labelled:
        st.warning("This dataset does not have two recognized classes. Supervised training/evaluation is disabled; open Alert Desk for unsupervised scoring.")
    if st.button("▶  Run model experiment", type="primary", disabled=not labelled, width="content"):
        with st.spinner(f"Training three candidates on up to {max_train_rows:,} rows with causal sequence features; creating separate validation and final holdout windows..."):
            try:
                results=evaluate_models(valid,test_size=holdout,contamination=contamination,max_train_rows=max_train_rows)
                st.session_state["aads_results"]=results
                st.session_state["aads_run_dataset"]=dataset_name
                run_id=save_experiment(results,dataset_name,len(valid))
                st.session_state["aads_run_id"]=run_id
                st.success(f"Experiment saved to registry. Run ID: {run_id}")
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
    results=st.session_state.get("aads_results")
    if results:
        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Train rows", f"{results['train_rows']:,}")
        c2.metric("Validation rows", f"{results['validation_rows']:,}")
        c3.metric("Final test rows", f"{results['test_rows']:,}")
        c4.metric("Best observed F1", f"{results['metrics']['F1'].max():.3f}")
        st.markdown(f"**Final test split:** {results['split_name']} · **Threshold validation:** {results['validation_split_name']}")
        for note in results["notes"]:
            # Use explicit branches here. A ternary expression returning Streamlit's
            # DeltaGenerator can be rendered by Streamlit magic as an unwanted object.
            if any(word in note.lower() for word in ("shift", "fallback", "prevalence")):
                st.warning(note)
            else:
                st.caption("• " + note)
        metrics=results["metrics"].reset_index()
        st.markdown("### Held-out evaluation")
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),hide_index=True,width="stretch")
        measure=st.selectbox("Metric for comparison chart",["F1","Precision","Recall","PR-AUC","ROC-AUC","False-positive rate"],index=0)
        fig=px.bar(metrics,x="Model",y=measure,color="Model",template="plotly_dark",text_auto=".3f",
                   color_discrete_sequence=["#5ce1e6","#ffbd69"],title=f"Candidate comparison · {measure}")
        fig.update_layout(showlegend=False,height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")
        selected=st.selectbox("Confusion matrix candidate",list(results["confusion"].keys()))
        cm=results["confusion"][selected]
        fig=px.imshow(cm,text_auto=True,x=["Predicted normal","Predicted attack"],y=["Actual normal","Actual attack"],
                      color_continuous_scale=[[0,"#132c44"],[1,"#5ce1e6"]],template="plotly_dark",title=f"{selected} · held-out confusion matrix")
        fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")
        st.caption("Thresholds are tuned on the validation slice, not on the final test window. Repeated model selection using one holdout can still overfit that benchmark; keep a final untouched benchmark for formal reporting.")
        st.markdown("### Global feature importance")
        importance = results.get("importance")
        if importance is not None and not importance.empty:
            importance_chart = importance.sort_values("importance", ascending=True).tail(20)
            imp_fig = px.bar(importance_chart, x="importance", y="feature", orientation="h",
                             template="plotly_dark", title=f"Top global drivers · {results['best_model_name']}",
                             color_discrete_sequence=["#5ce1e6"])
            imp_fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(imp_fig, width="stretch")
            st.caption("Global tree importance summarizes overall model behavior, not a causal or per-alert explanation.")
        else:
            st.info("Global feature importance is unavailable for the current selected model.")
        st.markdown("### Local prediction sensitivity")
        candidates=results["fitted"]
        explain_model_name=st.selectbox("Model to explain",list(candidates.keys()),key="explain_model_select")
        explain_pool=st.session_state.get("aads_scored")
        reference=valid
        if explain_pool is not None and len(explain_pool):
            pool=explain_pool.sort_values("anomaly_score",ascending=False).head(100)
            opt_labels=[f"row {i} · ID {r['ID']} · anomaly score {r['anomaly_score']:.4f}" for i,(_,r) in enumerate(pool.iterrows())]
            chosen_idx=st.selectbox("Select an unusual record (from recent scoring run)",range(len(pool)),format_func=lambda i:opt_labels[i])
            record=pool.iloc[[chosen_idx]][MODEL_INPUT_COLUMNS]
        else:
            position=st.number_input("Record row index",min_value=0,max_value=len(valid)-1,value=0,step=1)
            record=valid.iloc[[int(position)]][MODEL_INPUT_COLUMNS]
        if st.button("Explain selected prediction"):
            try:
                explanation=explain_record(candidates[explain_model_name],record,reference)
                if explanation.empty:
                    st.info("This record has insufficient signal values for local sensitivity analysis.")
                else:
                    st.dataframe(explanation,hide_index=True,width="stretch")
                    st.caption("One-feature-at-a-time ablation diagnostic. It is not SHAP, not causal evidence, and can miss feature interactions.")
            except Exception as exc:
                st.error(f"Could not create local sensitivity explanation: {exc}")
    else:
        st.info("Run an experiment to compare models and save its candidates and metrics into the local model registry.")

# ---------- Alert Desk ----------
elif page == "Alert Desk":
    section("Phases 06 / 09", "Alert Desk", "Score unusual telemetry, inspect the evidence, export findings, and send selected alerts to analyst review.")
    st.info("Isolation Forest scores are relative outlier rankings. When labels exist, the dashboard also compares flags to labels for diagnostic metrics.")
    b1,b2=st.columns([1,2])
    with b1:
        score_take = min(max_score_rows, max(1, int(len(valid) * 0.25)))
        run_score=st.button(f"⚡ Score latest {score_take:,} records",type="primary")
    with b2:
        st.caption(f"Isolation Forest fit cap: 80,000 normal-labelled records from the earlier window · score window: latest {score_take:,} records · expected outlier fraction: {contamination:.0%}.")
    if run_score:
        with st.spinner("Fitting a normal baseline on earlier telemetry, then scoring the later time window..."):
            try:
                ordered=valid.sort_values("Time",kind="stable").copy()
                take=min(max_score_rows,max(1,int(len(ordered)*0.25)))
                score_source=ordered.tail(take).copy()
                fit_source=ordered.iloc[:-take].copy()
                if len(fit_source) < 20:
                    raise ValueError("At least 20 earlier records are needed before a later window can be scored. Upload a larger dataset or reduce the score window.")
                # Train only on the earlier window, then score the latest contiguous window.
                detector=fit_detector(fit_source,contamination=contamination)
                scored=predict_frame(detector,score_source,batch_size=10_000,context_frame=ordered)
                ARTIFACTS.mkdir(exist_ok=True)
                joblib.dump(detector,ARTIFACTS/"isolation_forest.joblib",compress=3)
                score_id=log_alert_batch(scored,model_version=f"isolation-forest:{contamination:.2f}",max_alerts=500)
                st.session_state["aads_scored"]=scored
                st.session_state["aads_score_info"]={"count":len(scored),"sampled_from":len(valid),"stored_alerts":score_id,"contamination":contamination}
                st.success(f"Scoring complete. Saved {score_id} highest-scoring alert candidates to the review database.")
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
    scored=st.session_state.get("aads_scored")
    if scored is not None:
        info=st.session_state.get("aads_score_info",{})
        flagged=scored[scored["anomaly_prediction"]==1]
        x1,x2,x3,x4=st.columns(4)
        x1.metric("Records scored",f"{len(scored):,}")
        x2.metric("Flagged records",f"{len(flagged):,}")
        x3.metric("Flagged share",fmt_pct(len(flagged)/max(len(scored),1)))
        x4.metric("Review queue saved",f"{info.get('stored_alerts',0):,}")
        if has_binary_classes(scored):
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            y=scored[TARGET_COLUMN].to_numpy(dtype=int); pred=scored["anomaly_prediction"].to_numpy(dtype=int)
            c1,c2,c3=st.columns(3)
            c1.metric("Label-based precision",f"{precision_score(y,pred,zero_division=0):.3f}")
            c2.metric("Label-based recall",f"{recall_score(y,pred,zero_division=0):.3f}")
            c3.metric("Label-based F1",f"{f1_score(y,pred,zero_division=0):.3f}")
            st.caption("These are diagnostics against this dataset's labels and selected sampling strategy. The label quality and sample representativeness still matter.")
        display_cols=[c for c in [TARGET_COLUMN,"Time","ID",*SIGNAL_COLUMNS,"anomaly_prediction","anomaly_score"] if c in scored]
        top=scored.sort_values("anomaly_score",ascending=False).head(500)
        st.markdown("### Highest-scoring telemetry")
        st.dataframe(top[display_cols],hide_index=False,width="stretch",height=370)
        st.download_button("⬇ Download scored sample (CSV)",scored.to_csv(index=False).encode("utf-8"),
                           file_name="aads_scored_telemetry.csv",mime="text/csv",type="primary")
        st.markdown("### Local sensitivity on top flagged row")
        registry_now=list_registry()
        active=active_model_metadata()
        if active and ACTIVE_MODEL.exists():
            ids=top.index.tolist()
            chosen=st.selectbox("Choose top-scoring row to inspect",ids,format_func=lambda i:f"Row {i} · CAN ID {top.loc[i,'ID']} · score {top.loc[i,'anomaly_score']:.4f}")
            if st.button("Explain with active supervised model"):
                model=joblib.load(ACTIVE_MODEL)
                record=valid.loc[[chosen]][MODEL_INPUT_COLUMNS] if chosen in valid.index else top.loc[[chosen]][MODEL_INPUT_COLUMNS]
                expl=explain_record(model,record,valid)
                st.dataframe(expl,hide_index=True,width="stretch")
                st.caption("Explanation shows local model sensitivity, not proof of malicious intent.")
        else:
            st.caption("Promote a supervised candidate in Model Registry to enable classifier-based local sensitivity on these records.")
        st.markdown("### Human review queue")
        st.caption("The top 500 rows by anomaly score are stored locally. Review labels are saved in SQLite, not used for automatic retraining.")
        history=get_alert_history(limit=500)
        if not history.empty:
            unreviewed=history[history["reviewed_label"].isna()]
            options=unreviewed["id"].astype(int).tolist() if not unreviewed.empty else history["id"].astype(int).tolist()
            if options:
                alert_id=st.selectbox("Alert ID",options,format_func=lambda x:f"#{x} · " + str(history.loc[history['id']==x,'can_id'].iloc[0]))
                fc1,fc2=st.columns(2)
                with fc1:
                    reviewed_label=st.radio("Reviewer conclusion",[0,1],format_func=lambda v:"Normal / false alarm" if v==0 else "Attack / suspicious",horizontal=True)
                    reviewer=st.text_input("Reviewer",value="analyst")
                with fc2:
                    comment=st.text_area("Review note",placeholder="Why did you classify this alert this way?",max_chars=2000)
                if st.button("Save analyst feedback"):
                    try:
                        save_feedback(int(alert_id),int(reviewed_label),comment,reviewer)
                        st.success("Feedback saved. It is available in Monitoring.")
                    except Exception as exc:
                        st.error(str(exc))
    else:
        st.info("Click **Score sampled records** to fit Isolation Forest and create a review queue.")

# ---------- Model Registry ----------
elif page == "Model Registry":
    section("Phase 07", "Model Registry", "Versioned experiments, candidate artifacts, recorded metrics, and explicit model promotion.")
    active=active_model_metadata()
    if active:
        st.success(f"Active serving model: {active.get('model_name')} · {active.get('model_version')} · threshold {active.get('threshold',0.5):.3f}")
    else:
        st.warning("No model has been promoted for API inference yet. Promote a candidate below, or train one with scripts/train.py.")
    entries=list_registry()
    if not entries:
        st.info("No registered experiments yet. Run Model Studio or use `python -m scripts.train`.")
    else:
        flat=[]
        for run in entries:
            for candidate in run.get("candidates",[]):
                flat.append({"Run ID":run.get("run_id"),"Created":run.get("created_at"),"Dataset":run.get("dataset_name"),
                             "Model":candidate.get("model_name"),"Promoted":candidate.get("promoted",False),
                             "Threshold":candidate.get("threshold",0.5),"F1":candidate.get("metrics",{}).get("F1"),
                             "PR-AUC":candidate.get("metrics",{}).get("PR-AUC"),"Artifact":candidate.get("artifact")})
        st.dataframe(pd.DataFrame(flat),hide_index=True,width="stretch")
        runs_by_id={r["run_id"]:r for r in entries}
        selected_run=st.selectbox("Experiment run",list(runs_by_id.keys()),format_func=lambda rid:f"{rid} · {runs_by_id[rid].get('dataset_name','dataset')}")
        selected_entry=runs_by_id[selected_run]
        selected_names=[c["model_name"] for c in selected_entry.get("candidates",[])]
        selected_model=st.selectbox("Candidate model",selected_names)
        candidate=next(c for c in selected_entry["candidates"] if c["model_name"]==selected_model)
        st.json({"run_id":selected_run,"model":selected_model,"threshold":candidate.get("threshold",.5),"metrics":candidate.get("metrics",{}),"validation":candidate.get("validation",{})})
        st.warning("Promotion switches the local API artifact to this candidate. It does not certify the model for vehicle deployment. Check evaluation metrics and dataset limitations first.")
        if st.button("Promote selected candidate for local API",type="primary"):
            try:
                promoted=promote_candidate(selected_run,selected_model)
                st.session_state.pop("aads_api_model_cache",None)
                st.success(f"Promoted {promoted['model_version']}. The API checks the artifact modification time; restart it if the health endpoint still reports an old version.")
            except Exception as exc:
                st.error(f"Promotion failed: {exc}")
        st.markdown("### Experiment notes")
        for note in selected_entry.get("notes",[]): st.caption("• "+note)

# ---------- Monitoring ----------
elif page == "Monitoring":
    section("Phase 09", "Monitoring & feedback", "Observe prediction activity, review quality, and compare recent telemetry distributions with a reference window.")
    summary=monitoring_summary()
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Logged events",f"{summary['predictions_logged']:,}")
    m2.metric("Flagged events",f"{summary['flagged_predictions']:,}")
    m3.metric("Reviewed alerts",f"{summary['reviewed_alerts']:,}")
    m4.metric("Available supervised labels",f"{summary['predictions_with_ground_truth']:,}")
    if summary["review_agreement_with_ground_truth"] is not None:
        st.caption(f"Reviewer agreement with available source ground truth: {summary['review_agreement_with_ground_truth']:.1%}. This is not a model-accuracy estimate unless those records form a representative labelled evaluation sample.")
    st.markdown("### Data drift screening")
    if len(valid)<100:
        st.info("At least 100 rows are needed for a useful reference/current comparison.")
    else:
        ordered=valid.sort_values("Time",kind="stable") if "Time" in valid else valid
        if ACTIVE_BASELINE.is_file():
            reference_raw=load_dataset(ACTIVE_BASELINE)
            ref,_=validate_frame(reference_raw)
            current=ordered.iloc[int(len(ordered)*.75):]
            ref_source="promoted model training reference"
        else:
            split=max(20,int(len(ordered)*.75))
            ref=ordered.iloc[:split]
            current=ordered.iloc[split:]
            ref_source="first 75% of current dataset (fallback; promote/train a model to save a model-specific baseline)"
        ref=ref.iloc[np.linspace(0,len(ref)-1,min(50000,len(ref))).astype(int)]
        current=current.iloc[np.linspace(0,len(current)-1,min(50000,len(current))).astype(int)]
        report=drift_report(ref,current)
        st.caption(f"Reference: {ref_source} · {len(ref):,} reference rows · {len(current):,} current rows. PSI is a screening heuristic; investigate shifts and do not automatically retrain based on it alone.")
        if not report.empty:
            st.dataframe(report.style.format({"PSI":"{:.3f}","Reference missing %":"{:.2f}","Current missing %":"{:.2f}","Missing-rate change (pp)":"{:.2f}"},na_rep="—"),hide_index=True,width="stretch")
            numeric_report=report.dropna(subset=["PSI"])
            if not numeric_report.empty:
                fig=px.bar(numeric_report,x="Feature",y="PSI",color="Drift signal",template="plotly_dark",
                           color_discrete_map={"Low":"#5ce1e6","Moderate":"#ffbd69","High":"#ff667d"},title="Population Stability Index (PSI)")
                fig.add_hline(y=.1,line_dash="dash",line_color="#ffbd69",annotation_text="investigate")
                fig.add_hline(y=.25,line_dash="dash",line_color="#ff667d",annotation_text="stronger drift signal")
                fig.update_layout(height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,width="stretch")
    st.markdown("### Alert feedback history")
    history=get_alert_history(limit=500)
    if history.empty:
        st.info("No alerts or predictions have been logged yet. Use Alert Desk or the API to create events.")
    else:
        show_cols=[c for c in ["id","created_at","model_version","source","can_id","prediction","probability","anomaly_score","ground_truth","reviewed_label","reviewer","comment"] if c in history]
        st.dataframe(history[show_cols],hide_index=True,width="stretch",height=380)
        st.download_button("Export monitoring history",history[show_cols].to_csv(index=False).encode("utf-8"),
                           file_name="aads_monitoring_history.csv",mime="text/csv")
    st.caption("Feedback is stored for inspection. It does not automatically retrain or change the serving model.")

# ---------- API & Ops ----------
elif page == "API & Ops":
    section("Phases 08 / 10", "API & operations", "Serve predictions locally, verify runtime health, and reproduce the environment with one documented setup.")
    active=active_model_metadata()
    left,right=st.columns([1.1,1],gap="large")
    with left:
        st.markdown("### Local service status")
        st.markdown(f"**Model artifact:** {'Configured' if ACTIVE_MODEL.exists() else 'Not configured'}")
        st.markdown(f"**Active metadata:** {'Available' if active else 'Missing'}")
        st.markdown("**Expected local URLs**")
        st.code("Dashboard: http://127.0.0.1:8501\nAPI docs:  http://127.0.0.1:8000/docs\nHealth:    http://127.0.0.1:8000/health",language="text")
        st.markdown("**Start API in a second terminal**")
        st.code("python -m uvicorn api:app --reload",language="powershell")
        st.markdown("**Train/register and activate a local model**")
        st.code("python -m scripts.train --input compressed_data.csv.gz --max-rows 50000",language="powershell")
    with right:
        st.markdown("### Test request")
        st.code(json.dumps({"Time":81008449.45,"ID":"id3","Signal1_of_ID":0.2,"Signal2_of_ID":1.0,"Signal3_of_ID":None,"Signal4_of_ID":None},indent=2),language="json")
        st.markdown("### Health check response")
        st.code(json.dumps({"status":"ok","model":"loaded","model_version":active.get("model_version") if active else None},indent=2),language="json")
        st.caption("The actual health endpoint reports runtime status. This panel does not call the API automatically.")
    st.divider()
    st.markdown("### Reproducible developer workflow")
    st.code("""python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py""",language="powershell")
    st.markdown("### Docker workflow")
    st.code("docker build -t aads:4.2 .\ndocker run --rm -p 8501:8501 aads:4.2",language="powershell")
    st.markdown("### Operational boundary")
    st.warning("This prototype is designed for local research and portfolio demonstration. Add authentication, rate limits, secrets management, artifact integrity checks, and deployment monitoring before exposing the API outside a trusted development environment. Do not connect it to vehicle control or safety-critical functions.")

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#6e859a;font-size:.76rem'><span>AADS · Automotive AI Security Platform · v4.2</span><span>Research only · no safety-critical use · trained outputs require validation</span></div>",unsafe_allow_html=True)
