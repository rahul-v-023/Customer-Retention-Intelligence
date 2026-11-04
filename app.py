"""
Customer Retention Intelligence — Streamlit Dashboard
======================================================
Interactive churn prediction interface with risk classification
and personalized retention recommendations.

Author : Rahul V
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ──────────────────────────────────────────────────────────────
# Constants & Configuration
# ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "Customer_Retention_100_Records.csv"

RISK_THRESHOLDS = {"high": 0.60, "medium": 0.30}
CHURN_THRESHOLD = 0.50

PAGE_CONFIG = {
    "page_title": "Customer Retention Intelligence",
    "page_icon": "📊",
    "layout": "wide",
}


# ──────────────────────────────────────────────────────────────
# Data Loading & Preparation
# ──────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Load and clean the customer dataset."""
    df = pd.read_csv(path)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate features and binary churn target."""
    X = df.drop(columns=["Customer_ID", "Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y


def get_feature_groups(X: pd.DataFrame):
    """Identify numeric and categorical feature columns."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    return numeric, categorical


# ──────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────

def build_preprocessor(numeric_features: list, categorical_features: list):
    """Build a sklearn ColumnTransformer for mixed feature types."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


@st.cache_resource
def train_model(_X: pd.DataFrame, _y: pd.Series,
                numeric_features: list, categorical_features: list):
    """Train a Logistic Regression pipeline and return the fitted model."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(_X, _y)
    return pipeline


# ──────────────────────────────────────────────────────────────
# Risk Classification & Retention Recommendations
# ──────────────────────────────────────────────────────────────

def classify_risk(probability: float) -> str:
    """Assign a risk tier based on churn probability."""
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def generate_retention_offer(customer: dict, risk: str,
                             median_charges: float) -> str:
    """
    Build a personalized retention recommendation string
    based on risk tier and customer profile attributes.
    """
    offers: list[str] = []

    # --- Risk-based core offers ---
    risk_offers = {
        "High": [
            "15% monthly bill discount for 3 months",
            "Personal retention call within 24 hours",
        ],
        "Medium": [
            "10% monthly bill discount for 2 months",
            "Offer a suitable plan review",
        ],
        "Low": [
            "5% loyalty discount on next bill",
            "Send a loyalty appreciation message",
        ],
    }
    offers.extend(risk_offers.get(risk, []))

    # --- Context-aware add-on offers ---
    if customer.get("Contract") == "Month-to-month":
        offers.append("Special discount for switching to a 1-year contract")

    if customer.get("TechSupport") == "No":
        offers.append("One month of free technical support")

    if customer.get("OnlineSecurity") == "No":
        offers.append("One month of free online security")

    if float(customer.get("MonthlyCharges", 0)) >= median_charges:
        offers.append("Review plan for a lower-cost package")

    # De-duplicate while preserving order
    return " | ".join(dict.fromkeys(offers))


# ──────────────────────────────────────────────────────────────
# Report Builder
# ──────────────────────────────────────────────────────────────

def build_retention_report(df: pd.DataFrame, X: pd.DataFrame,
                           model, median_charges: float) -> pd.DataFrame:
    """Generate the full customer-wise retention report."""
    probabilities = model.predict_proba(X)[:, 1]

    report = df[["Customer_ID", "Churn"]].copy()
    report.rename(columns={"Churn": "Actual_Churn"}, inplace=True)

    report["Churn_Probability"] = probabilities.round(4)
    report["Predicted_Churn"] = np.where(
        probabilities >= CHURN_THRESHOLD, "Yes", "No"
    )
    report["Risk_Level"] = [classify_risk(p) for p in probabilities]
    report["Recommended_Offer"] = [
        generate_retention_offer(row.to_dict(), classify_risk(p), median_charges)
        for (_, row), p in zip(X.iterrows(), probabilities)
    ]

    return report


# ──────────────────────────────────────────────────────────────
# Streamlit UI Components
# ──────────────────────────────────────────────────────────────

def render_report_section(report: pd.DataFrame):
    """Display the full retention report with download button."""
    st.subheader("📋 Customer-Wise Retention Report")
    st.dataframe(report, use_container_width=True, hide_index=True)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Retention Report (CSV)",
        data=csv,
        file_name="Customer_Retention_Report.csv",
        mime="text/csv",
    )


def render_prediction_form(X: pd.DataFrame, numeric_features: list,
                           model, median_charges: float):
    """Render the individual customer prediction form and results."""
    st.subheader("🔍 Individual Customer Churn Prediction")
    st.write(
        "Enter customer details to estimate churn probability "
        "and receive a suggested retention action."
    )

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        values: dict = {}

        for idx, feature in enumerate(X.columns):
            target_col = columns[idx % 3]

            if feature in numeric_features:
                default = (
                    float(X[feature].median())
                    if X[feature].notna().any()
                    else 0.0
                )
                if feature == "SeniorCitizen":
                    values[feature] = target_col.selectbox(
                        feature, options=[0, 1], index=0
                    )
                elif feature == "tenure":
                    values[feature] = target_col.number_input(
                        feature, min_value=0, max_value=100,
                        value=int(default),
                    )
                else:
                    values[feature] = target_col.number_input(
                        feature, min_value=0.0, value=default, step=1.0,
                    )
            else:
                options = sorted(
                    X[feature].dropna().astype(str).unique().tolist()
                )
                values[feature] = target_col.selectbox(feature, options)

        submitted = st.form_submit_button("🚀 Predict Churn Risk")

    if submitted:
        _display_prediction(values, X.columns, model, median_charges)


def _display_prediction(values: dict, feature_columns,
                        model, median_charges: float):
    """Compute and display prediction results for one customer."""
    input_df = pd.DataFrame([values], columns=feature_columns)

    probability = float(model.predict_proba(input_df)[0, 1])
    risk = classify_risk(probability)
    offer = generate_retention_offer(values, risk, median_charges)

    # --- Results display ---
    st.divider()
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Estimated Churn Probability", f"{probability:.1%}")
    res_col2.metric("Risk Level", risk)

    st.write("**Recommended Retention Action**")
    st.info(offer)

    # --- Individual report download ---
    individual = input_df.copy()
    individual.insert(0, "Customer_ID", "New_Customer")
    individual["Churn_Probability"] = round(probability, 4)
    individual["Predicted_Churn"] = "Yes" if probability >= CHURN_THRESHOLD else "No"
    individual["Risk_Level"] = risk
    individual["Recommended_Offer"] = offer

    csv = individual.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Individual Prediction (CSV)",
        data=csv,
        file_name="Individual_Customer_Retention_Prediction.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────

def main():
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(**PAGE_CONFIG)

    # --- Header ---
    st.title("📊 Customer Retention Intelligence")
    st.caption(
        "Churn Probability  •  Risk Classification  •  Retention Recommendations"
    )

    # --- Data & Model ---
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    numeric_features, categorical_features = get_feature_groups(X)
    model = train_model(X, y, numeric_features, categorical_features)
    median_charges = float(df["MonthlyCharges"].median())

    # --- Full Report ---
    report = build_retention_report(df, X, model, median_charges)
    render_report_section(report)

    st.divider()

    # --- Individual Prediction ---
    render_prediction_form(X, numeric_features, model, median_charges)

    # --- Footer ---
    st.divider()
    st.caption(
        "⚠️ Educational prototype trained on the included 100-record sample. "
        "Predictions and offers are illustrative, not production-validated "
        "or automatically applied."
    )


if __name__ == "__main__":
    main()
