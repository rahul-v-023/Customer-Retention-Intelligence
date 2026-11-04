<div align="center">

# 📊 Customer Retention Intelligence

**Predict churn. Quantify risk. Retain customers.**

An end-to-end Machine Learning system that predicts customer churn probability, classifies risk levels, and generates personalized retention strategies — complete with an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

<img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="status"/>
<img src="https://img.shields.io/badge/Type-Mini%20Project-blue" alt="type"/>

</div>

---

## 🎯 Problem Statement

Customer churn is one of the most critical challenges for subscription-based businesses. Losing a customer costs **5–25× more** than retaining one. This project builds a machine learning pipeline that:

- **Predicts** the probability of a customer churning
- **Classifies** customers into **High / Medium / Low** risk tiers
- **Recommends** targeted, personalized retention offers
- **Visualizes** insights through an interactive dashboard

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│              Customer_Retention_100_Records.csv                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │ Missing Value │  │  Standard     │  │  One-Hot Encoding    │ │
│  │ Imputation    │  │  Scaling      │  │  (Categoricals)      │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                         │
│  ┌──────────────────┐ ┌──────────────┐ ┌─────────────────────┐ │
│  │ Logistic         │ │ Decision     │ │ Random Forest       │ │
│  │ Regression       │ │ Tree         │ │ Classifier          │ │
│  └──────────────────┘ └──────────────┘ └─────────────────────┘ │
│          5-Fold Stratified Cross-Validation                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTELLIGENCE ENGINE                           │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │ Churn         │  │  Risk Level   │  │  Retention Offer     │ │
│  │ Probability   │  │  Assignment   │  │  Recommendation      │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│    ┌───────────────────┐        ┌────────────────────────────┐ │
│    │  Jupyter Notebook │        │  Streamlit Dashboard       │ │
│    │  (Analysis + EDA) │        │  (Interactive Predictions) │ │
│    └───────────────────┘        └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Customer-Retention-Intelligence/
│
├── 📓 Customer_Retention_Intelligence.ipynb   # Complete ML workflow & analysis
├── 🚀 app.py                                  # Streamlit interactive dashboard
├── 📄 Customer_Retention_100_Records.csv       # Input dataset (100 records)
├── 📊 Customer_Retention_Final_Results.csv     # Pre-generated model results
├── 📋 requirements.txt                         # Python dependencies
├── 📝 README.md                                # Project documentation
└── 🔒 .gitignore                               # Git ignore rules
```

---

## 🔬 Features & Methodology

### Machine Learning Pipeline

| Component | Details |
|-----------|---------|
| **Algorithms** | Logistic Regression, Decision Tree, Random Forest |
| **Validation** | 5-Fold Stratified Cross-Validation |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Preprocessing** | Median imputation, Standard scaling, One-hot encoding |
| **Diagnostics** | Confusion matrices, ROC curves, Feature importances |

### Risk Classification Engine

| Risk Tier | Churn Probability | Action Priority |
|-----------|-------------------|-----------------|
| 🔴 **High** | ≥ 60% | Immediate intervention |
| 🟡 **Medium** | 30% – 59% | Proactive engagement |
| 🟢 **Low** | < 30% | Loyalty reinforcement |

### Intelligent Retention Recommendations

The system generates **context-aware offers** based on:

- Customer risk tier (discount intensity scales with risk)
- Contract type (month-to-month customers get contract migration offers)
- Service gaps (missing Tech Support / Online Security → free trials)
- Spending pattern (high spenders get plan review suggestions)

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/rahul-v-023/Customer-Retention-Intelligence.git
cd Customer-Retention-Intelligence

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Jupyter Notebook

For the full analysis pipeline — EDA, model training, evaluation, and diagnostics:

```bash
jupyter notebook Customer_Retention_Intelligence.ipynb
```

Run all cells top-to-bottom to reproduce the complete workflow.

### Launch the Streamlit Dashboard

For the interactive prediction interface:

```bash
streamlit run app.py
```

Then open the local URL printed in the terminal (typically `http://localhost:8501`).

**Dashboard capabilities:**
- 📋 View the complete customer retention report with predictions
- 🔍 Enter individual customer profiles for real-time churn prediction
- 📈 See churn probability, risk level, and recommended actions
- 📥 Download reports as CSV files

---

## 📊 Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| `Customer_ID` | ID | Unique customer identifier |
| `SeniorCitizen` | Binary | Whether the customer is a senior citizen (0/1) |
| `Partner` | Categorical | Whether the customer has a partner |
| `Dependents` | Categorical | Whether the customer has dependents |
| `tenure` | Numeric | Number of months with the company |
| `InternetService` | Categorical | Internet service type (DSL / Fiber optic / No) |
| `OnlineSecurity` | Categorical | Online security add-on subscription |
| `OnlineBackup` | Categorical | Online backup add-on subscription |
| `TechSupport` | Categorical | Technical support add-on subscription |
| `Contract` | Categorical | Contract type (Month-to-month / One year / Two year) |
| `PaperlessBilling` | Categorical | Whether the customer uses paperless billing |
| `PaymentMethod` | Categorical | Payment method used |
| `MonthlyCharges` | Numeric | Monthly charge amount (\$) |
| `TotalCharges` | Numeric | Total charges to date (\$) |
| `Churn` | Target | Whether the customer churned (Yes/No) |

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:----------:|:-------:|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core Language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data Manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical Computing |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) | Machine Learning |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white) | Data Visualization |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | Interactive Dashboard |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) | Notebook Environment |

</div>

---

## ⚠️ Limitations & Disclaimer

> **Note:** This is an educational prototype built as a BTech CSE mini-project.

- The included dataset contains only **100 records** — evaluation metrics are illustrative
- Model performance should be validated on a larger, representative dataset before any production use
- Feature coefficients and importances indicate **model associations**, not causal relationships
- Retention offers are **suggestions only** and are not automatically applied

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author

**Rahul V**

[![GitHub](https://img.shields.io/badge/GitHub-rahul--v--023-181717?logo=github)](https://github.com/rahul-v-023)

---

<div align="center">

⭐ **If you found this project useful, consider giving it a star!** ⭐

</div>
