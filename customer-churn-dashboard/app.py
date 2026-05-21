from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


APP_TITLE = "Customer Churn Intelligence Dashboard"
DATA_PATH = Path(__file__).parent / "data" / "telecom_churn.csv"
RANDOM_STATE = 42
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
THEME = {
    "background": "#080B12",
    "surface": "#101522",
    "surface_2": "#151D2E",
    "border": "#263149",
    "text": "#F4F7FB",
    "muted": "#A9B4C6",
    "cyan": "#35D0FF",
    "green": "#56F0A6",
    "amber": "#FFD166",
    "red": "#FF5C7A",
    "violet": "#A88CFF",
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    target_column: str
    metrics: dict[str, float]
    confusion: np.ndarray
    roc_points: tuple[np.ndarray, np.ndarray]
    report: pd.DataFrame
    predictions: pd.DataFrame
    train_rows: int
    test_rows: int
    model_name: str


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(build_css(), unsafe_allow_html=True)


def build_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --bg: {THEME["background"]};
        --surface: {THEME["surface"]};
        --surface-2: {THEME["surface_2"]};
        --border: {THEME["border"]};
        --text: {THEME["text"]};
        --muted: {THEME["muted"]};
        --cyan: {THEME["cyan"]};
        --green: {THEME["green"]};
        --amber: {THEME["amber"]};
        --red: {THEME["red"]};
        --violet: {THEME["violet"]};
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(180deg, #080B12 0%, #0A0F19 48%, #080B12 100%);
        color: var(--text);
    }}

    .block-container {{
        max-width: 1440px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(12, 16, 27, .92);
        border-right: 1px solid rgba(255, 255, 255, .08);
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    div[data-testid="stFileUploader"] section {{
        background: rgba(21, 29, 46, .82);
        border: 1px dashed rgba(53, 208, 255, .42);
        border-radius: 14px;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 2.0rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.10);
        background: linear-gradient(135deg, rgba(17, 23, 36, .98), rgba(10, 15, 26, .94));
        box-shadow: 0 18px 56px rgba(0,0,0,.30);
    }}

    .brand-pill {{
        display: inline-flex;
        gap: .5rem;
        align-items: center;
        padding: .45rem .75rem;
        border: 1px solid rgba(53, 208, 255, .34);
        border-radius: 999px;
        color: #DDE6F3;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, .04);
    }}

    .hero h1 {{
        margin: .9rem 0 .7rem 0;
        max-width: 980px;
        color: var(--text);
        font-size: clamp(2.1rem, 4.1vw, 4.7rem);
        line-height: .98;
        font-weight: 800;
        letter-spacing: 0;
    }}

    .hero p {{
        max-width: 760px;
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.65;
    }}

    .section-title {{
        margin: 2.1rem 0 .85rem 0;
    }}

    .section-title h2 {{
        margin: 0;
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0;
    }}

    .section-title p {{
        margin: .3rem 0 0 0;
        color: var(--muted);
        font-size: .93rem;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.4rem;
    }}

    .metric-card, .insight-card, .summary-card {{
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(19,26,40,.94), rgba(12,17,29,.94));
        box-shadow: 0 14px 38px rgba(0,0,0,.20);
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }}

    .metric-card:hover, .insight-card:hover, .summary-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(53, 208, 255, .35);
        box-shadow: 0 22px 58px rgba(0,0,0,.30);
    }}

    .metric-card {{
        padding: 1rem;
    }}

    .metric-label {{
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }}

    .metric-value {{
        margin-top: .55rem;
        color: var(--text);
        font-size: 1.9rem;
        line-height: 1;
        font-weight: 800;
    }}

    .metric-trend {{
        margin-top: .7rem;
        color: var(--muted);
        font-size: .84rem;
    }}

    .trend-good {{ color: var(--green); }}
    .trend-warn {{ color: var(--amber); }}
    .trend-bad {{ color: var(--red); }}

    .panel {{
        padding: 1rem;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 12px;
        background: rgba(14, 20, 34, .72);
    }}

    .insight-card, .summary-card {{
        min-height: 126px;
        padding: 1.05rem;
    }}

    .insight-card h3, .summary-card h3 {{
        margin: 0 0 .45rem 0;
        color: var(--text);
        font-size: .98rem;
        font-weight: 800;
    }}

    .insight-card p, .summary-card p {{
        margin: 0;
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.55;
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        padding: .27rem .56rem;
        border-radius: 999px;
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .02em;
    }}

    .badge-low {{ background: rgba(86, 240, 166, .12); color: var(--green); border: 1px solid rgba(86, 240, 166, .28); }}
    .badge-medium {{ background: rgba(255, 209, 102, .13); color: var(--amber); border: 1px solid rgba(255, 209, 102, .28); }}
    .badge-high {{ background: rgba(255, 92, 122, .13); color: var(--red); border: 1px solid rgba(255, 92, 122, .28); }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 11px;
        border: 1px solid rgba(53, 208, 255, .32);
        background: rgba(30, 41, 59, .92);
        color: var(--text);
        font-weight: 800;
        min-height: 2.7rem;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: rgba(148, 163, 184, .78);
        color: white;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        overflow: hidden;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(21,29,46,.78);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: .9rem;
    }}

    .js-plotly-plot .plotly .modebar {{
        opacity: .22;
    }}

    @media (max-width: 1000px) {{
        .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .hero {{ padding: 1.35rem; }}
    }}

    @media (max-width: 640px) {{
        .metric-grid {{ grid-template-columns: 1fr; }}
        .metric-value {{ font-size: 1.65rem; }}
    }}
    </style>
    """


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return generate_demo_data()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def generate_demo_data(rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    tenure = rng.integers(1, 73, rows)
    monthly_charge = rng.normal(72, 22, rows).clip(18, 145)
    support_calls = rng.poisson(1.4, rows).clip(0, 8)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55, .25, .20])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], rows)
    logit = (
        -2.1
        + (contract == "Month-to-month") * 1.35
        + (payment == "Electronic check") * .55
        + (monthly_charge > 88) * .55
        + (support_calls >= 4) * .9
        - tenure * .025
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)
    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(rows)],
            "tenure": tenure,
            "Contract": contract,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charge.round(2),
            "TotalCharges": (monthly_charge * tenure + rng.normal(0, 90, rows)).clip(0).round(2),
            "SupportTickets": support_calls,
            "Churn": churn,
        }
    )


def infer_target_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Churn",
        "churn",
        "Exited",
        "Attrition",
        "is_churn",
        "Customer_Status",
        "target",
        "label",
    ]
    for column in candidates:
        if column in df.columns:
            return column

    binary_columns = []
    for column in df.columns:
        series = df[column].dropna()
        if 1 < series.nunique() <= 2:
            binary_columns.append(column)
    return binary_columns[-1] if binary_columns else None


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    positive_values = {"yes", "true", "1", "churn", "churned", "left", "exited", "lost", "cancelled"}
    normalized = series.astype(str).str.strip().str.lower()
    if normalized.nunique() == 2:
        positive = normalized.map(lambda value: 1 if value in positive_values else 0)
        if positive.nunique() == 1:
            return pd.factorize(normalized)[0]
        return positive.astype(int)
    return pd.factorize(normalized)[0]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip().replace(" ", "_") for column in cleaned.columns]
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            numeric_version = pd.to_numeric(cleaned[column], errors="coerce")
            if numeric_version.notna().mean() > .86:
                cleaned[column] = numeric_version
    return cleaned


def detect_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    drop_like = {"customerid", "customer_id", "id", "name", "email", "phone", target_column.lower()}
    feature_columns = []
    for column in df.columns:
        if column.lower() in drop_like:
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        feature_columns.append(column)
    return feature_columns


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier(model_choice: str, class_weight: str | None):
    if model_choice == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=260,
            max_depth=4,
            learning_rate=.055,
            subsample=.88,
            colsample_bytree=.88,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=340,
        max_depth=12,
        min_samples_leaf=3,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


@st.cache_resource(show_spinner=False)
def train_model_cached(
    df_json: str,
    target_column: str,
    model_choice: str,
    test_size: float,
    class_weight: str | None,
) -> ModelBundle:
    df = pd.read_json(io.StringIO(df_json), orient="split")
    df[target_column] = normalize_target(df[target_column])
    feature_columns = detect_feature_columns(df, target_column)
    X = df[feature_columns]
    y = df[target_column].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    classifier = build_classifier(model_choice, class_weight)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(df, feature_columns)),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_probability = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_probability = y_pred.astype(float)

    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    scored = score_customers(df, pipeline, feature_columns, target_column)

    model_name = "XGBoost Classifier" if model_choice == "XGBoost" and XGBOOST_AVAILABLE else "Random Forest Classifier"
    return ModelBundle(
        pipeline=pipeline,
        feature_names=feature_names,
        target_column=target_column,
        metrics={
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_probability),
        },
        confusion=confusion_matrix(y_test, y_pred),
        roc_points=(fpr, tpr),
        report=report,
        predictions=scored,
        train_rows=len(X_train),
        test_rows=len(X_test),
        model_name=model_name,
    )


def score_customers(df: pd.DataFrame, pipeline: Pipeline, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    scored = df.copy()
    probabilities = pipeline.predict_proba(scored[feature_columns])[:, 1]
    scored["Churn_Probability"] = probabilities
    scored["Predicted_Churn"] = np.where(probabilities >= .5, 1, 0)
    scored["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.001, .34, .67, 1.001],
        labels=RISK_ORDER,
    ).astype(str)

    leading_columns = [column for column in ["customerID", "CustomerID", "customer_id", target_column] if column in scored.columns]
    prediction_columns = ["Churn_Probability", "Predicted_Churn", "Risk_Level"]
    remaining = [column for column in scored.columns if column not in leading_columns + prediction_columns]
    return scored[leading_columns + prediction_columns + remaining]


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(label: str, value: str, trend: str, status: str = "good") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-trend trend-{status}">{trend}</div>
    </div>
    """


def render_hero(bundle: ModelBundle) -> None:
    predictions = bundle.predictions
    churn_rate = predictions["Predicted_Churn"].mean()
    high_risk = (predictions["Risk_Level"] == "High Risk").sum()
    avg_probability = predictions["Churn_Probability"].mean()
    churn_status = "bad" if churn_rate > .28 else "warn" if churn_rate > .16 else "good"
    risk_status = "bad" if high_risk / max(len(predictions), 1) > .25 else "warn"

    st.markdown(
        f"""
        <div class="hero">
            <div class="brand-pill">Retention Analytics Platform</div>
            <h1>{APP_TITLE}</h1>
            <p>
                A premium churn analytics command center for retention teams.
                Upload customer data, train a predictive model, segment risk, and convert scoring signals
                into executive-ready retention actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    kpi_columns = st.columns(4)
    cards = [
        metric_card("Total Customers", f"{len(predictions):,}", "Live portfolio coverage", "good"),
        metric_card("Predicted Churn Rate", format_percent(churn_rate), "Model-scored customer base", churn_status),
        metric_card("High-Risk Customers", f"{high_risk:,}", "Immediate retention queue", risk_status),
        metric_card("Avg. Churn Probability", format_percent(avg_probability), "Portfolio risk intensity", churn_status),
    ]
    for column, card in zip(kpi_columns, cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)


def plot_template() -> str:
    px.defaults.template = "plotly_dark"
    return "plotly_dark"


def update_figure_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=52, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 11, 18, .25)",
        font=dict(color=THEME["text"], family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#101522", bordercolor=THEME["border"], font_size=13),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.10)")
    return fig


def render_data_section(df: pd.DataFrame, target_column: str) -> None:
    section_title("Data Upload & Preview", "Inspect schema, completeness, data quality, and customer-level records.")
    upload_col, quality_col = st.columns([1.2, .8], gap="large")

    with upload_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.dataframe(df.head(250), width="stretch", height=330)
        st.markdown("</div>", unsafe_allow_html=True)

    with quality_col:
        missing_pct = df.isna().mean().sort_values(ascending=False)
        duplicate_rows = df.duplicated().sum()
        numeric_count = len(df.select_dtypes(include=np.number).columns)
        categorical_count = df.shape[1] - numeric_count

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c1.metric("Numeric Fields", numeric_count)
        c2.metric("Categorical Fields", categorical_count)
        c1.metric("Duplicate Rows", f"{duplicate_rows:,}")
        c2.metric("Target", target_column)
        st.markdown("</div>", unsafe_allow_html=True)

        missing_frame = (
            missing_pct[missing_pct > 0]
            .mul(100)
            .round(2)
            .rename("Missing %")
            .reset_index()
            .rename(columns={"index": "Column"})
        )
        if missing_frame.empty:
            st.success("Data quality check passed: no missing values detected.")
        else:
            st.dataframe(missing_frame, width="stretch", height=170)


def render_model_section(bundle: ModelBundle) -> None:
    section_title("Model Training", "Automated preprocessing, train/test validation, and model performance diagnostics.")
    st.caption(f"Active model: {bundle.model_name} | Train rows: {bundle.train_rows:,} | Test rows: {bundle.test_rows:,}")

    metric_cols = st.columns(5)
    for col, (metric_name, value) in zip(metric_cols, bundle.metrics.items()):
        col.metric(metric_name, format_percent(value))

    confusion_col, roc_col = st.columns(2, gap="large")
    with confusion_col:
        cm = bundle.confusion
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale=["#101522", THEME["cyan"], THEME["violet"]],
            labels=dict(x="Predicted", y="Actual", color="Customers"),
            x=["Retained", "Churn"],
            y=["Retained", "Churn"],
            title="Confusion Matrix",
        )
        st.plotly_chart(update_figure_layout(fig), width="stretch")

    with roc_col:
        fpr, tpr = bundle.roc_points
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=THEME["cyan"], width=4), name="Model ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=THEME["muted"], dash="dash"), name="Baseline"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(update_figure_layout(fig), width="stretch")

    report = bundle.report.copy()
    st.dataframe(report.style.format("{:.3f}"), width="stretch", height=230)


def risk_badge(level: str) -> str:
    css = {"Low Risk": "badge-low", "Medium Risk": "badge-medium", "High Risk": "badge-high"}.get(level, "badge-medium")
    return f'<span class="badge {css}">{level}</span>'


def render_predictions_section(bundle: ModelBundle) -> None:
    section_title("Churn Predictions", "Prioritized customer table with probability scores, churn flags, and risk badges.")
    predictions = bundle.predictions.copy()
    risk_filter = st.multiselect("Risk segment", RISK_ORDER, default=RISK_ORDER)
    probability_floor = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.01)
    filtered = predictions[
        predictions["Risk_Level"].isin(risk_filter)
        & (predictions["Churn_Probability"] >= probability_floor)
    ].sort_values("Churn_Probability", ascending=False)

    chart_col, table_col = st.columns([.85, 1.15], gap="large")
    with chart_col:
        risk_distribution = (
            filtered.groupby("Risk_Level", observed=False)
            .size()
            .reindex(RISK_ORDER, fill_value=0)
            .reset_index(name="Customers")
        )
        fig = px.bar(
            risk_distribution,
            x="Risk_Level",
            y="Customers",
            color="Risk_Level",
            category_orders={"Risk_Level": RISK_ORDER},
            color_discrete_map={"Low Risk": THEME["green"], "Medium Risk": THEME["amber"], "High Risk": THEME["red"]},
            title="Filtered Risk Distribution",
        )
        st.plotly_chart(update_figure_layout(fig, height=360), width="stretch")

    with table_col:
        table = filtered.head(12).copy()
        display_columns = [
            column
            for column in table.columns
            if column in ["customerID", "CustomerID", "customer_id", bundle.target_column, "Churn_Probability", "Predicted_Churn", "Risk_Level"]
        ]
        if len(display_columns) < 5:
            display_columns.extend([column for column in table.columns if column not in display_columns][: 5 - len(display_columns)])
        table = table[display_columns]
        st.caption(f"Showing top {len(table):,} customers from {len(filtered):,} filtered records.")
        st.dataframe(
            table.style.format({"Churn_Probability": "{:.1%}"}).map(
                lambda value: (
                    f"color: {THEME['red']}; font-weight: 700;"
                    if value == "High Risk"
                    else f"color: {THEME['amber']}; font-weight: 700;"
                    if value == "Medium Risk"
                    else f"color: {THEME['green']}; font-weight: 700;"
                    if value == "Low Risk"
                    else ""
                )
            ),
            width="stretch",
            height=360,
        )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction CSV",
        csv,
        "customer_churn_predictions.csv",
        "text/csv",
        width="stretch",
    )


def get_importance_frame(bundle: ModelBundle) -> pd.DataFrame:
    classifier = bundle.pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = np.zeros(len(bundle.feature_names))
    return (
        pd.DataFrame({"Feature": bundle.feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(20)
    )


def render_feature_importance_section(bundle: ModelBundle) -> None:
    section_title("Feature Importance", "Identify the behavioral and financial drivers behind churn risk.")
    importance = get_importance_frame(bundle)
    col_a, col_b = st.columns([1.08, .92], gap="large")

    with col_a:
        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#35D0FF", "#A88CFF", "#FF5C7A"],
            title="Top Churn Drivers",
        )
        st.plotly_chart(update_figure_layout(fig, height=520), width="stretch")

    with col_b:
        shap_frame = compute_shap_values(bundle)
        if shap_frame is not None:
            fig = px.bar(
                shap_frame.sort_values("Mean |SHAP|"),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                color="Mean |SHAP|",
                color_continuous_scale=["#56F0A6", "#35D0FF", "#A88CFF"],
                title="SHAP Contribution Analysis",
            )
            st.plotly_chart(update_figure_layout(fig, height=520), width="stretch")
        else:
            st.info("SHAP explanation is unavailable for this dataset/model combination. Feature importance is still shown.")

    top_features = importance.head(3)["Feature"].tolist()
    interpretation = build_feature_interpretation(top_features)
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>Business Interpretation</h3>
            <p>{interpretation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def compute_shap_values_cached(df_json: str, feature_names: tuple[str, ...], target_column: str, model_choice: str) -> pd.DataFrame | None:
    return None


def compute_shap_values(bundle: ModelBundle) -> pd.DataFrame | None:
    try:
        preprocessor = bundle.pipeline.named_steps["preprocessor"]
        classifier = bundle.pipeline.named_steps["classifier"]
        source = bundle.predictions.drop(columns=["Churn_Probability", "Predicted_Churn", "Risk_Level"], errors="ignore")
        feature_columns = detect_feature_columns(source, bundle.target_column)
        sample = source[feature_columns].head(350)
        transformed = preprocessor.transform(sample)
        transformed = np.asarray(transformed)
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        if values.ndim == 3:
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)
        return (
            pd.DataFrame({"Feature": bundle.feature_names, "Mean |SHAP|": mean_abs})
            .sort_values("Mean |SHAP|", ascending=False)
            .head(14)
        )
    except Exception:
        return None


def build_feature_interpretation(features: Iterable[str]) -> str:
    readable = ", ".join(feature.replace("_", " ") for feature in features)
    if not readable:
        return "The model did not expose ranked drivers, but the churn scores remain available for operational prioritization."
    return (
        f"The strongest churn signals in this portfolio are {readable}. "
        "Retention teams should use these drivers to shape targeted outreach, pricing interventions, onboarding support, and contract renewal campaigns."
    )


def find_column(df: pd.DataFrame, keywords: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in df.columns}
    for keyword in keywords:
        for lowered_column, original in lowered.items():
            if keyword in lowered_column:
                return original
    return None


def enrich_segments(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    tenure_col = find_column(enriched, ["tenure", "accountweeks", "account_week"])
    charge_col = find_column(enriched, ["monthlycharge", "monthly_charge", "monthlycharges", "charge"])
    contract_col = find_column(enriched, ["contract"])
    payment_col = find_column(enriched, ["payment"])

    if tenure_col:
        enriched["Tenure_Segment"] = pd.cut(
            pd.to_numeric(enriched[tenure_col], errors="coerce"),
            bins=[-np.inf, 12, 36, 72, np.inf],
            labels=["0-12", "13-36", "37-72", "73+"],
        ).astype(str)
    else:
        enriched["Tenure_Segment"] = "Unknown"

    if charge_col:
        enriched["Charge_Segment"] = pd.qcut(
            pd.to_numeric(enriched[charge_col], errors="coerce").rank(method="first"),
            q=4,
            labels=["Budget", "Core", "Premium", "Enterprise"],
        ).astype(str)
    else:
        enriched["Charge_Segment"] = "Unknown"

    if contract_col:
        enriched["Contract_Segment"] = enriched[contract_col].astype(str)
    elif "ContractRenewal" in enriched.columns:
        enriched["Contract_Segment"] = np.where(enriched["ContractRenewal"].astype(float) > 0, "Renewed Contract", "No Renewal")
    else:
        enriched["Contract_Segment"] = "Standard Plan"

    if payment_col:
        enriched["Payment_Segment"] = enriched[payment_col].astype(str)
    elif "OverageFee" in enriched.columns:
        enriched["Payment_Segment"] = pd.qcut(
            pd.to_numeric(enriched["OverageFee"], errors="coerce").rank(method="first"),
            q=3,
            labels=["Low overage", "Moderate overage", "High overage"],
        ).astype(str)
    else:
        enriched["Payment_Segment"] = "Digital payment"

    return enriched


def render_segmentation_section(bundle: ModelBundle) -> None:
    section_title("Customer Segmentation", "Analyze churn concentration across tenure, pricing, contract, and payment dimensions.")
    data = enrich_segments(bundle.predictions)

    row_1_col_1, row_1_col_2 = st.columns(2, gap="large")
    with row_1_col_1:
        tenure_chart = (
            data.groupby("Tenure_Segment", observed=False)["Churn_Probability"]
            .mean()
            .reset_index()
            .sort_values("Tenure_Segment")
        )
        fig = px.bar(
            tenure_chart,
            x="Tenure_Segment",
            y="Churn_Probability",
            color="Churn_Probability",
            color_continuous_scale=["#56F0A6", "#FFD166", "#FF5C7A"],
            title="Churn Risk by Tenure",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(update_figure_layout(fig), width="stretch")

    with row_1_col_2:
        contract = (
            data.groupby(["Contract_Segment", "Risk_Level"], observed=False)
            .size()
            .reset_index(name="Customers")
        )
        fig = px.bar(
            contract,
            x="Contract_Segment",
            y="Customers",
            color="Risk_Level",
            category_orders={"Risk_Level": RISK_ORDER},
            color_discrete_map={"Low Risk": THEME["green"], "Medium Risk": THEME["amber"], "High Risk": THEME["red"]},
            title="Risk Mix by Contract",
        )
        st.plotly_chart(update_figure_layout(fig), width="stretch")

    row_2_col_1, row_2_col_2 = st.columns(2, gap="large")
    with row_2_col_1:
        charge_col = find_column(data, ["monthlycharge", "monthly_charge", "monthlycharges", "charge"])
        if charge_col:
            fig = px.box(
                data,
                x="Risk_Level",
                y=charge_col,
                color="Risk_Level",
                category_orders={"Risk_Level": RISK_ORDER},
                color_discrete_map={"Low Risk": THEME["green"], "Medium Risk": THEME["amber"], "High Risk": THEME["red"]},
                title="Monthly Charges by Risk Segment",
            )
            st.plotly_chart(update_figure_layout(fig), width="stretch")
        else:
            st.info("Monthly charge analysis will appear when a charge column is available.")

    with row_2_col_2:
        payment = data.groupby("Payment_Segment", observed=False)["Churn_Probability"].mean().reset_index()
        fig = px.pie(
            payment,
            values="Churn_Probability",
            names="Payment_Segment",
            hole=.58,
            color_discrete_sequence=[THEME["cyan"], THEME["violet"], THEME["green"], THEME["amber"], THEME["red"]],
            title="Relative Churn Exposure by Payment Segment",
        )
        st.plotly_chart(update_figure_layout(fig), width="stretch")

    heatmap_data = (
        data.pivot_table(
            index="Tenure_Segment",
            columns="Charge_Segment",
            values="Churn_Probability",
            aggfunc="mean",
            observed=False,
        )
        .fillna(0)
    )
    fig = px.imshow(
        heatmap_data,
        text_auto=".0%",
        color_continuous_scale=["#101522", "#35D0FF", "#FFD166", "#FF5C7A"],
        title="Churn Probability Heatmap: Tenure x Charge Segment",
    )
    st.plotly_chart(update_figure_layout(fig, height=440), width="stretch")


def build_business_insights(bundle: ModelBundle) -> list[tuple[str, str]]:
    data = enrich_segments(bundle.predictions)
    insights = []

    high_risk_rate = (data["Risk_Level"] == "High Risk").mean()
    insights.append(
        (
            "Retention Queue",
            f"{format_percent(high_risk_rate)} of customers are high risk. Prioritize this segment for proactive save offers and customer success outreach.",
        )
    )

    tenure_risk = data.groupby("Tenure_Segment", observed=False)["Churn_Probability"].mean().sort_values(ascending=False)
    if not tenure_risk.empty:
        insights.append(
            (
                "Tenure Signal",
                f"Customers in the {tenure_risk.index[0]} tenure band show the highest predicted churn probability at {format_percent(tenure_risk.iloc[0])}.",
            )
        )

    contract_risk = data.groupby("Contract_Segment", observed=False)["Churn_Probability"].mean().sort_values(ascending=False)
    if not contract_risk.empty:
        insights.append(
            (
                "Contract Exposure",
                f"{contract_risk.index[0]} customers carry the highest churn exposure. Renewal campaigns should focus there first.",
            )
        )

    payment_risk = data.groupby("Payment_Segment", observed=False)["Churn_Probability"].mean().sort_values(ascending=False)
    if not payment_risk.empty:
        insights.append(
            (
                "Payment Pattern",
                f"The {payment_risk.index[0]} payment segment has elevated risk. Review billing friction, failed payments, and plan-fit messaging.",
            )
        )

    return insights


def render_insights_section(bundle: ModelBundle) -> None:
    section_title("Business Insights", "Automatically generated retention signals and actions for operators.")
    insights = build_business_insights(bundle)
    columns = st.columns(2)
    for index, (title, body) in enumerate(insights):
        with columns[index % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h3>{title}</h3>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_executive_summary(bundle: ModelBundle) -> None:
    section_title("Executive Summary", "Consulting-style synthesis of churn trends, operational risk, and retention strategy.")
    predictions = bundle.predictions
    churn_rate = predictions["Predicted_Churn"].mean()
    high_risk = predictions[predictions["Risk_Level"] == "High Risk"]
    avg_probability = predictions["Churn_Probability"].mean()
    revenue_col = find_column(predictions, ["monthlycharge", "monthly_charge", "monthlycharges", "charge"])
    estimated_revenue_at_risk = None
    if revenue_col:
        estimated_revenue_at_risk = pd.to_numeric(high_risk[revenue_col], errors="coerce").sum()

    cards = [
        (
            "Churn Trend",
            f"The current model estimates a {format_percent(churn_rate)} churn rate with an average customer churn probability of {format_percent(avg_probability)}.",
        ),
        (
            "Risk Observation",
            f"{len(high_risk):,} customers are classified as high risk and should be routed into retention workflows within the next campaign cycle.",
        ),
        (
            "Retention Recommendation",
            "Launch tiered interventions: fast-response outreach for high-risk accounts, education journeys for medium-risk users, and loyalty reinforcement for low-risk customers.",
        ),
        (
            "Business Impact",
            f"Estimated monthly revenue at risk is ${estimated_revenue_at_risk:,.0f}." if estimated_revenue_at_risk is not None else "Revenue-at-risk estimation will activate when a monthly charge column is present.",
        ),
    ]

    columns = st.columns(4)
    for col, (title, body) in zip(columns, cards):
        with col:
            st.markdown(
                f"""
                <div class="summary-card">
                    <h3>{title}</h3>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar(default_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, float, str | None]:
    st.sidebar.markdown("## Control Center")
    st.sidebar.caption("Upload a customer CSV or use the included telecom churn sample.")
    uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = clean_dataframe(read_uploaded_csv(uploaded_file))
            source_name = uploaded_file.name
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")
            df = clean_dataframe(default_df)
            source_name = "Included telecom sample"
    else:
        df = clean_dataframe(default_df)
        source_name = "Included telecom sample"

    detected_target = infer_target_column(df)
    target_options = list(df.columns)
    target_index = target_options.index(detected_target) if detected_target in target_options else max(len(target_options) - 1, 0)
    target_column = st.sidebar.selectbox("Target column", target_options, index=target_index)

    available_models = ["XGBoost", "RandomForest"] if XGBOOST_AVAILABLE else ["RandomForest"]
    model_choice = st.sidebar.selectbox("Model", available_models, index=0)
    test_size = st.sidebar.slider("Test size", .15, .40, .25, .05)
    balance_classes = st.sidebar.toggle("Balance classes", value=True)
    class_weight = "balanced" if balance_classes and model_choice == "RandomForest" else None

    retrain = st.sidebar.button("Retrain model", width="stretch")
    if retrain:
        st.cache_resource.clear()
        st.toast("Model cache cleared. Retraining with current settings.")

    st.sidebar.markdown("---")
    st.sidebar.metric("Dataset", source_name)
    st.sidebar.metric("Rows loaded", f"{len(df):,}")
    st.sidebar.metric("Columns loaded", f"{df.shape[1]:,}")
    return df, target_column, model_choice, test_size, class_weight


def validate_dataset(df: pd.DataFrame, target_column: str) -> list[str]:
    errors = []
    if target_column not in df.columns:
        errors.append("The selected target column is missing from the dataset.")
    elif normalize_target(df[target_column]).nunique() != 2:
        errors.append("The target column must contain exactly two classes for churn classification.")
    if len(df) < 50:
        errors.append("At least 50 rows are recommended for stable model training.")
    feature_columns = detect_feature_columns(df, target_column) if target_column in df.columns else []
    if len(feature_columns) < 2:
        errors.append("The dataset needs at least two usable feature columns.")
    return errors


def main() -> None:
    configure_page()
    plot_template()
    default_df = load_default_data()
    df, target_column, model_choice, test_size, class_weight = render_sidebar(default_df)

    errors = validate_dataset(df, target_column)
    if errors:
        st.error("Dataset validation failed.")
        for error in errors:
            st.warning(error)
        st.stop()

    df_json = df.to_json(orient="split", date_format="iso")
    with st.spinner("Training churn model and generating customer intelligence..."):
        bundle = train_model_cached(df_json, target_column, model_choice, test_size, class_weight)

    pages = [
        "Overview",
        "Data & Model",
        "Predictions",
        "Drivers & Segments",
        "Executive Summary",
    ]
    selected_page = st.sidebar.radio("Pages", pages, index=0)

    if selected_page == "Overview":
        render_hero(bundle)
        render_insights_section(bundle)
        render_executive_summary(bundle)
    elif selected_page == "Data & Model":
        render_hero(bundle)
        render_data_section(df, target_column)
        render_model_section(bundle)
    elif selected_page == "Predictions":
        render_hero(bundle)
        render_predictions_section(bundle)
    elif selected_page == "Drivers & Segments":
        render_hero(bundle)
        render_feature_importance_section(bundle)
        render_segmentation_section(bundle)
    else:
        render_hero(bundle)
        render_executive_summary(bundle)
        render_insights_section(bundle)

    st.caption(
        textwrap.dedent(
            """
            Built as a client-facing churn intelligence platform. Predictions are model-generated and should be paired with business review before operational deployment.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
