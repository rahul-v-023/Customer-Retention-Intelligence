# Customer Retention Intelligence

A BTech CSE mini-project that predicts customer churn, assigns risk levels, and suggests retention actions.

## Included files
- `Customer_Retention_100_Records.csv` — 100-record sample dataset.
- `Customer_Retention_Intelligence.ipynb` — complete ML workflow, visualizations, evaluation, and model diagnostics.
- `app.py` — interactive Streamlit dashboard for individual customer predictions and downloadable reports.
- `Customer_Retention_Final_Results.csv` — notebook-generated sample results (regenerate by running the notebook).
- `requirements.txt` — Python dependencies.

## Run the notebook
1. Install dependencies: `pip install -r requirements.txt`
2. Open `Customer_Retention_Intelligence.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom.

## Run the interactive dashboard
1. Open a terminal in this project folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Open the local URL printed in the terminal.

The dashboard shows the full sample retention report, supports an individual customer profile form, predicts churn probability and risk, suggests retention actions, and allows CSV downloads for both the full report and an individual prediction.

## Model evaluation and diagnostics
The notebook compares Logistic Regression, Decision Tree, and Random Forest using 5-fold stratified cross-validation. It reports Accuracy, Precision, Recall, F1-score, and ROC-AUC; displays confusion matrices and ROC curves for all three models; and analyzes Logistic Regression coefficients and tree-based feature importances.

The main customer-wise notebook workflow uses Logistic Regression with out-of-fold probabilities. The Streamlit demo fits Logistic Regression on the complete included sample to support interactive predictions.

## Important limitations
The included dataset is a small 100-record sample. Evaluation results are illustrative and may vary; they are not evidence of production-level performance. Validate on a larger, representative dataset before business deployment. Feature coefficients/importances indicate model associations, not causation. Retention offers are suggestions only and are not automatically applied.
