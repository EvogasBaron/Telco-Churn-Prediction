
import os

# ====== ĐƯỜNG DẪN GỐC PROJECT ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ====== ĐƯỜNG DẪN DỮ LIỆU ======
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
FEATURED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "featured_data.csv")

# ====== ĐƯỜNG DẪN MODEL ======
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_final.pkl")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_final.pkl")
STACKING_MODEL_PATH = os.path.join(MODEL_DIR, "stacking_final.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_columns.pkl")  # lưu danh sách cột sau one-hot

# ====== ĐƯỜNG DẪN BÁO CÁO ======
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
COMPARISON_REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison_report.md")

# ====== CỘT DỮ LIỆU ======
TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

ADDON_SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Các cột feature engineering sẽ tạo thêm (xem feature_engineering.py)
ENGINEERED_NUMERIC_COLS = [
    "charge_velocity", "engagement_score", "cltv_proxy",
    "contract_risk_score", "price_sensitivity_index",
]

# ====== CẤU HÌNH HUẤN LUYỆN ======
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Trọng số dùng để ưu tiên Recall khi cần (tham khảo cho scoring tùy biến)
SCORING_METRIC = "recall"