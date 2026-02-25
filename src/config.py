"""Project configuration."""

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_PATH = "data/telco_churn.csv"

ARTIFACT_DIR = "outputs"
MODEL_PATH = "outputs/model.joblib"
METRICS_PATH = "outputs/metrics.json"
ROC_PATH = "outputs/roc_curve.png"
PR_PATH = "outputs/pr_curve.png"
CM_PATH = "outputs/confusion_matrix.png"
COEF_PATH = "outputs/top_coefficients.csv"
