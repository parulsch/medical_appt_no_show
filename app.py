# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap

# Optional: only needed if you want to include MLP inference in the app.
# If tensorflow is not installed, the app will still work (MLP plots are images).
try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Appointment No-Show Risk Predictor", layout="wide")

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ARTIFACTS / "figures"
METRICS_DIR = ARTIFACTS / "metrics"
MODEL_DIR = ARTIFACTS / "models"
SHAP_DIR = ARTIFACTS / "shap"


# -----------------------------
# Small utilities
# -----------------------------
def must_exist(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"Missing {label}: {path}")
        st.stop()


def show_img(img_path: Path, caption_2sent: str) -> None:
    must_exist(img_path, "image")
    st.image(str(img_path), use_container_width=True)
    st.caption(caption_2sent)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    p = METRICS_DIR / "model_comparison.csv"
    must_exist(p, "model_comparison.csv")
    dfm = pd.read_csv(p)
    if "model" in dfm.columns:
        dfm = dfm.set_index("model")
    return dfm


@st.cache_resource
def load_models() -> Tuple[Dict[str, Any], Optional[Any]]:
    """
    Loads sklearn pipeline models from joblib.
    Returns:
      models: dict of {display_name: pipeline_model}
      mlp_model: tf model if available and present; else None
    """
    must_exist(MODEL_DIR, "artifacts/models")

    models: Dict[str, Any] = {}

    mapping = [
        ("RandomForest", MODEL_DIR / "rf.joblib"),
        ("XGBoost", MODEL_DIR / "xgb.joblib"),
        ("LogisticRegression", MODEL_DIR / "logreg.joblib"),
        ("DecisionTree", MODEL_DIR / "tree.joblib"),
    ]
    for name, path in mapping:
        if path.exists():
            models[name] = joblib.load(path)

    mlp_model = None
    mlp_path = MODEL_DIR / "mlp.keras"
    if TF_AVAILABLE and mlp_path.exists():
        try:
            mlp_model = tf.keras.models.load_model(mlp_path)
        except Exception:
            mlp_model = None

    return models, mlp_model


def extract_key_hyperparams(model_name: str, pipeline_obj: Any) -> Dict[str, Any]:
    """
    Extracts hyperparameters from the loaded estimator used in the saved pipeline.
    """
    params: Dict[str, Any] = {}

    clf = pipeline_obj.named_steps["model"] if hasattr(pipeline_obj, "named_steps") else pipeline_obj
    p = clf.get_params()

    if model_name == "DecisionTree":
        keys = ["max_depth", "min_samples_leaf", "min_samples_split", "criterion", "class_weight", "random_state"]
    elif model_name == "RandomForest":
        keys = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features",
                "bootstrap", "class_weight", "random_state"]
    elif model_name == "XGBoost":
        keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "reg_alpha", "reg_lambda", "gamma", "min_child_weight"]
    elif model_name == "LogisticRegression":
        keys = ["C", "penalty", "solver", "class_weight", "max_iter", "l1_ratio"]
    else:
        keys = list(p.keys())

    for k in keys:
        if k in p:
            params[k] = p[k]
    return params


def safe_default_inputs() -> Dict[str, Any]:
    """
    Defaults for interactive prediction.
    Note: Neighbourhood is high-cardinality; we set a safe default.
    """
    return {
        "Gender": "F",
        "Age": 35,
        "Neighbourhood": "Other",
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 0,
        "lead_time_days": 7,
        "appt_weekday": "Monday",
    }


def build_input_row(vals: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([vals])


def predict_proba_selected(
    model_name: str,
    models: Dict[str, Any],
    mlp_model: Optional[Any],
    X_row: pd.DataFrame
) -> float:
    """
    Returns predicted probability of no-show (class 1).
    """
    if model_name == "MLP":
        if mlp_model is None:
            raise RuntimeError("MLP model not available (tensorflow not installed or model missing).")

        preprocess_path = MODEL_DIR / "preprocess.joblib"
        must_exist(preprocess_path, "preprocess.joblib")
        pre = joblib.load(preprocess_path)

        X_one = pre.transform(X_row)
        X_one_dense = X_one.toarray() if hasattr(X_one, "toarray") else np.asarray(X_one)
        X_one_dense = np.asarray(X_one_dense, dtype=np.float64)  # ensure numeric

        proba = float(mlp_model.predict(X_one_dense, verbose=0).reshape(-1)[0])
        return proba

    pipeline = models[model_name]
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    return proba


def rf_shap_waterfall(rf_pipeline: Any, X_row: pd.DataFrame, max_display: int = 15) -> plt.Figure:
    """
    Robust SHAP waterfall for one custom input using RandomForest.
    Handles SHAP output differences across versions (list vs ndarray; scalar vs array base values).
    """
    pre = rf_pipeline.named_steps["preprocess"]
    rf_model = rf_pipeline.named_steps["model"]

    # Transform to model space
    X_one = pre.transform(X_row)
    X_one_dense = X_one.toarray() if hasattr(X_one, "toarray") else np.asarray(X_one)
    X_one_dense = np.asarray(X_one_dense, dtype=np.float64)

    # Feature names
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(X_one_dense.shape[1])])
    feature_names = [str(x) for x in feature_names]

    # Compute SHAP
    explainer = shap.TreeExplainer(rf_model)
    sv = explainer.shap_values(X_one_dense)

    class_idx = 1  # no-show class

    # Select SHAP values for this row, robustly
    if isinstance(sv, list):
        # [class0_matrix, class1_matrix]
        shap_row = np.asarray(sv[class_idx][0], dtype=np.float64)
    else:
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            shap_row = np.asarray(sv_arr[0, :, class_idx], dtype=np.float64)
        elif sv_arr.ndim == 2:
            # (n_samples, n_features)
            shap_row = np.asarray(sv_arr[0], dtype=np.float64)
        else:
            shap_row = np.asarray(sv_arr).reshape(-1).astype(np.float64)

    # Select base value robustly
    ev = explainer.expected_value
    if isinstance(ev, list):
        base = ev[class_idx]
    else:
        ev_arr = np.asarray(ev)
        if ev_arr.ndim == 0:
            base = ev_arr
        else:
            flat = ev_arr.reshape(-1)
            base = flat[class_idx] if flat.size > class_idx else flat[0]

    base_value = float(np.asarray(base).reshape(-1)[0])

    exp = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=X_one_dense[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


# -----------------------------
# Validate folders
# -----------------------------
must_exist(ARTIFACTS, "artifacts folder")
must_exist(FIG_DIR, "artifacts/figures")
must_exist(METRICS_DIR, "artifacts/metrics")
must_exist(MODEL_DIR, "artifacts/models")
must_exist(SHAP_DIR, "artifacts/shap")

df_results = load_model_comparison()
models, mlp_model = load_models()

model_choices = list(models.keys())
if mlp_model is not None:
    model_choices.append("MLP")


# -----------------------------
# Page header
# -----------------------------
st.title("Appointment No-Show Risk Predictor (MSIS 522 HW1)")
st.write("End-to-end workflow: Descriptive analytics → Multiple models → Explainability (SHAP) → Deployed Streamlit app.")

tabs = st.tabs([
    "Tab 1 — Executive Summary",
    "Tab 2 — Descriptive Analytics",
    "Tab 3 — Model Performance",
    "Tab 4 — Explainability & Interactive Prediction"
])


# =========================================================
# TAB 1 — Executive Summary
# =========================================================
with tabs[0]:
    st.header("Executive Summary")

    st.subheader("Dataset & prediction task")
    st.write(
        "This project uses a tabular dataset of medical appointments, where each row represents a scheduled appointment. "
        "The dataset contains patient and appointment attributes such as age, gender, whether an SMS reminder was received, "
        "lead time between scheduling and the appointment date, weekday, neighborhood, and several binary health indicators "
        "(e.g., hypertension, diabetes). The prediction target is **No-show** (binary): whether the patient missed the appointment."
    )

    st.subheader("Why this matters (the “so what”)")
    st.write(
        "No-shows create operational and financial costs for clinics and other appointment-based services: unused time slots, "
        "inefficient staffing, longer wait times for other patients, and lost revenue. A reliable no-show risk predictor enables "
        "targeted interventions (reminders, confirmations, deposits for long lead-time bookings) so the organization can reduce "
        "missed appointments while maintaining a good customer/patient experience."
    )

    st.subheader("Approach & key findings")
    st.write(
        "We followed the complete data science workflow: exploratory visual analysis of the dataset, model training and comparison "
        "using a held-out test set, and explainability using SHAP for a tree-based model. Because the no-show class is imbalanced "
        "(~20.4%), we focus on **F1** and **AUC** rather than accuracy alone to measure how well the model identifies no-shows."
    )
    st.write(
        "In our results, **Random Forest** achieved the strongest overall performance by **F1**, indicating the best precision–recall "
        "balance for catching no-shows. XGBoost performed similarly and achieved a slightly higher AUC, indicating strong risk ranking "
        "across thresholds. SHAP explains which features most influence predictions and supports operational decisions like prioritizing "
        "outreach for high-risk appointments."
    )


# =========================================================
# TAB 2 — Descriptive Analytics
# =========================================================
with tabs[1]:
    st.header("Descriptive Analytics (Part 1)")

    col1, col2 = st.columns(2)

    with col1:
        show_img(
            FIG_DIR / "target_distribution.png",
            "The target distribution shows that no-shows are the minority class (~20.4%). "
            "Because the dataset is imbalanced, we rely on F1 and AUC to evaluate models rather than accuracy alone."
        )
        show_img(
            FIG_DIR / "rate_by_sms.png",
            "This plot compares no-show rates by whether an SMS reminder was received. "
            "If the rates differ meaningfully, reminder workflows become a concrete operational lever to reduce missed appointments."
        )
        show_img(
            FIG_DIR / "rate_by_weekday.png",
            "No-show rates vary by weekday, indicating behavioral or operational patterns across the week. "
            "This can guide staffing decisions and targeted reminder strategies for higher-risk days."
        )

    with col2:
        show_img(
            FIG_DIR / "rate_by_leadtime_bucket.png",
            "Lead time (days between scheduling and the appointment) is strongly associated with no-show behavior. "
            "Long lead times can increase uncertainty and forgetting, motivating extra confirmations closer to the appointment date."
        )
        show_img(
            FIG_DIR / "age_by_outcome.png",
            "Comparing age distributions between show vs. no-show appointments helps identify whether risk differs by age group. "
            "If differences exist, age becomes a useful predictive feature, while interventions should remain fair and supportive."
        )
        show_img(
            FIG_DIR / "correlation_heatmap.png",
            "The correlation heatmap highlights relationships among numeric features and possible redundancy in signals. "
            "Strong correlations can affect interpretation for linear models, while tree models are more robust; SHAP helps clarify drivers."
        )


# =========================================================
# TAB 3 — Model Performance
# =========================================================
with tabs[2]:
    st.header("Model Performance (Part 2)")

    st.subheader("Model comparison table (test set)")
    st.dataframe(df_results.style.format("{:.3f}"), use_container_width=True)

    show_img(
        FIG_DIR / "model_f1_comparison.png",
        "This chart compares F1 scores across models on the held-out test set. "
        "F1 is emphasized because identifying no-shows is an imbalanced classification problem and we care about precision–recall balance."
    )

    st.subheader("ROC curves (test set)")
    roc_cols = st.columns(3)
    roc_files = [
        ("Statsmodels Logit", FIG_DIR / "roc_logit_statsmodels.png"),
        ("Sklearn LogisticRegression", FIG_DIR / "roc_logreg.png"),
        ("Decision Tree", FIG_DIR / "roc_tree.png"),
        ("Random Forest", FIG_DIR / "roc_rf.png"),
        ("XGBoost", FIG_DIR / "roc_xgb.png"),
        ("MLP", FIG_DIR / "roc_mlp.png"),
    ]
    for i, (label, p) in enumerate(roc_files):
        with roc_cols[i % 3]:
            show_img(
                p,
                f"The ROC curve for {label} summarizes performance across decision thresholds. "
                "Higher AUC indicates better ranking of no-show risk even when the classification cutoff changes."
            )

    st.subheader("Best hyperparameters (from the saved best models)")
    st.write(
        "The values below are extracted directly from the saved final models in `artifacts/models/`. "
        "These correspond to the best hyperparameters selected during cross-validation/tuning (or the final settings used)."
    )

    rows = []
    for mname in ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"]:
        if mname in models:
            rows.append({"model": mname, **extract_key_hyperparams(mname, models[mname])})

    params_df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    if not params_df.empty:
        st.dataframe(params_df, use_container_width=True)
    else:
        st.warning("No tunable models found to report hyperparameters.")

    show_img(
        FIG_DIR / "decision_tree_visual.png",
        "This visualization shows the learned structure of the best decision tree (top levels for readability). "
        "It illustrates how rule-based splits separate higher- vs. lower-risk appointments, although deeper trees can overfit."
    )

    rules_path = METRICS_DIR / "decision_tree_rules.txt"
    if rules_path.exists():
        with st.expander("Decision tree full rules (text export)"):
            st.code(rules_path.read_text()[:20000])


# =========================================================
# TAB 4 — Explainability & Interactive Prediction
# =========================================================
with tabs[3]:
    st.header("Explainability & Interactive Prediction")

    st.subheader("Global explainability (SHAP)")
    cA, cB = st.columns(2)
    with cA:
        show_img(
            SHAP_DIR / "shap_summary_beeswarm_rf.png",
            "The SHAP beeswarm plot shows which features matter most and how they push predictions up or down across many examples. "
            "Points to the right increase predicted no-show risk; points to the left decrease it."
        )
    with cB:
        show_img(
            SHAP_DIR / "shap_bar_mean_abs_rf.png",
            "The SHAP bar plot ranks features by mean absolute SHAP value (average contribution magnitude). "
            "This provides a stable feature-importance ranking without showing direction for each individual example."
        )

    st.divider()
    st.subheader("Interactive prediction")

    if not model_choices:
        st.error("No models found in artifacts/models.")
        st.stop()

    chosen_model = st.selectbox("Select a model for prediction", options=model_choices, index=0)
    defaults = safe_default_inputs()

    col1, col2, col3 = st.columns(3)
    with col1:
        lead_time_days = st.slider("Lead time (days)", 0, 180, int(defaults["lead_time_days"]))
        sms_received = st.selectbox("SMS received", [0, 1], index=int(defaults["SMS_received"]))
        age = st.slider("Age", 0, 110, int(defaults["Age"]))
    with col2:
        gender = st.selectbox("Gender", ["F", "M"], index=0 if defaults["Gender"] == "F" else 1)
        appt_weekday = st.selectbox(
            "Appointment weekday",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0
        )
        scholarship = st.selectbox("Scholarship", [0, 1], index=int(defaults["Scholarship"]))
    with col3:
        hipertension = st.selectbox("Hipertension", [0, 1], index=int(defaults["Hipertension"]))
        diabetes = st.selectbox("Diabetes", [0, 1], index=int(defaults["Diabetes"]))
        alcoholism = st.selectbox("Alcoholism", [0, 1], index=int(defaults["Alcoholism"]))
        handcap = st.selectbox("Handcap", [0, 1], index=int(defaults["Handcap"]))

    user_vals = {
        "Gender": gender,
        "Age": age,
        "Neighbourhood": "Other",
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "lead_time_days": lead_time_days,
        "appt_weekday": appt_weekday,
    }

    X_row = build_input_row(user_vals)

    try:
        proba = predict_proba_selected(chosen_model, models, mlp_model, X_row)
        pred_class = 1 if proba >= 0.5 else 0
        st.markdown("### Predicted outcome")
        st.write(f"**Model:** {chosen_model}")
        st.write(f"**Predicted class:** {'No-show (1)' if pred_class == 1 else 'Show (0)'}")
        st.write(f"**Predicted probability of no-show:** {proba:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    st.subheader("SHAP waterfall for your custom input")

    if "RandomForest" not in models:
        st.warning("RandomForest model not found; SHAP waterfall is shown only for RandomForest.")
    else:
        try:
            fig = rf_shap_waterfall(models["RandomForest"], X_row, max_display=15)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Could not compute SHAP waterfall: {e}")

    st.caption(
        "Deployment requirement: models are pre-trained and loaded from `artifacts/models/`. "
        "The app does not retrain models on the fly."
    )