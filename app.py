import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path

st.set_page_config(page_title="Customer Retention Intelligence", page_icon="📊", layout="wide")
st.title("Customer Retention Intelligence")
st.caption("Churn probability • Customer risk • Retention recommendations")
DATA_PATH = Path(__file__).parent / "Customer_Retention_100_Records.csv"
df = pd.read_csv(DATA_PATH)
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
X = df.drop(columns=["Customer_ID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])
@st.cache_resource
def train_model():
    pipeline = Pipeline([("preprocessing", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipeline.fit(X, y)
    return pipeline
model = train_model()

def risk_level(prob):
    return "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")

def recommend(customer, risk):
    offers = []
    if risk == "High":
        offers += ["15% monthly bill discount for 3 months", "Personal retention call within 24 hours"]
    elif risk == "Medium":
        offers += ["10% monthly bill discount for 2 months", "Offer a suitable plan review"]
    else:
        offers += ["5% loyalty discount on next bill", "Send a loyalty appreciation message"]
    if customer.get("Contract") == "Month-to-month": offers.append("Special discount for switching to a 1-year contract")
    if customer.get("TechSupport") == "No": offers.append("One month of free technical support")
    if customer.get("OnlineSecurity") == "No": offers.append("One month of free online security")
    if float(customer.get("MonthlyCharges", 0)) >= df["MonthlyCharges"].median(): offers.append("Review plan for a lower-cost package")
    return " | ".join(dict.fromkeys(offers))

# Build complete dataset report with model predictions and retention actions.
all_probs = model.predict_proba(X)[:, 1]
report = df[["Customer_ID", "Churn"]].copy().rename(columns={"Churn": "Actual_Churn"})
report["Churn_Probability"] = all_probs.round(4)
report["Predicted_Churn"] = np.where(all_probs >= 0.50, "Yes", "No")
report["Risk_Level"] = [risk_level(p) for p in all_probs]
report["Recommended_Offer"] = [recommend(row, risk_level(p)) for (_, row), p in zip(X.iterrows(), all_probs)]

st.subheader("Customer-wise retention report")
st.dataframe(report, use_container_width=True, hide_index=True)
st.download_button("Download full retention report (CSV)", report.to_csv(index=False).encode("utf-8"), "Customer_Retention_Report.csv", "text/csv")
st.divider()
st.subheader("Individual customer churn prediction")
st.write("Enter customer details to estimate churn probability and receive a suggested retention action.")
with st.form("customer_form"):
    c1, c2, c3 = st.columns(3)
    values = {}
    for j, col in enumerate(X.columns):
        target = [c1, c2, c3][j % 3]
        if col in numeric_features:
            default = float(X[col].median()) if X[col].notna().any() else 0.0
            if col == "SeniorCitizen": values[col] = target.selectbox(col, [0, 1], index=0)
            elif col == "tenure": values[col] = target.number_input(col, min_value=0, max_value=100, value=int(default))
            else: values[col] = target.number_input(col, min_value=0.0, value=default, step=1.0)
        else:
            opts = sorted(X[col].dropna().astype(str).unique().tolist())
            values[col] = target.selectbox(col, opts)
    submitted = st.form_submit_button("Predict churn risk")
if submitted:
    input_df = pd.DataFrame([values], columns=X.columns)
    probability = float(model.predict_proba(input_df)[0, 1])
    risk = risk_level(probability)
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.metric("Risk level", risk)
    st.write("**Recommended retention action**")
    st.info(recommend(values, risk))
    individual = input_df.copy()
    individual.insert(0, "Customer_ID", "New_Customer")
    individual["Churn_Probability"] = probability
    individual["Predicted_Churn"] = "Yes" if probability >= 0.50 else "No"
    individual["Risk_Level"] = risk
    individual["Recommended_Offer"] = recommend(values, risk)
    st.download_button("Download individual prediction (CSV)", individual.to_csv(index=False).encode("utf-8"), "Individual_Customer_Retention_Prediction.csv", "text/csv")

st.caption("Educational prototype trained on the included 100-record sample. Predictions and offers are illustrative, not production-validated or automatically applied.")
