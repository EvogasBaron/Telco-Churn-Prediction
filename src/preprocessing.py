
import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURED_DATA_PATH, SCALER_PATH, ENCODER_PATH,
    TARGET_COL, ID_COL,
    NUMERIC_COLS, ENGINEERED_NUMERIC_COLS,
    TEST_SIZE, RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Các cột sẽ loại bỏ trước khi đưa vào mô hình
DROP_COLS = [
    ID_COL,
    # Các cột gốc đã được mã hóa/biến đổi thành feature mới
    "Contract",         # -> contract_risk_score
    "PaymentMethod",    # -> payment_risk_flag
    # tenure_group là dạng string categorical -> sẽ one-hot encode bên dưới,
    # nhưng các cột gốc tạo ra nó (tenure, MonthlyCharges) vẫn giữ lại vì mang
    # thêm thông tin liên tục mà binning bị mất.
]

# Tất cả cột numeric cần scale (gốc + engineered)
ALL_NUMERIC_COLS = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS


def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các cột không cần thiết, bỏ qua nếu cột không tồn tại."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    logger.info(f"Loại bỏ các cột: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:

    # tenure_group là Categorical dtype (từ pd.cut), cần convert về string trước
    if "tenure_group" in df.columns:
        df = df.copy()
        df["tenure_group"] = df["tenure_group"].astype(str)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"One-Hot Encoding {len(categorical_cols)} cột categorical: {categorical_cols}")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df


def split_and_prepare(
    df: pd.DataFrame = None,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    if df is None:
        logger.info(f"Đọc dữ liệu từ: {FEATURED_DATA_PATH}")
        df = pd.read_csv(FEATURED_DATA_PATH)

    logger.info(f"Shape đầu vào: {df.shape}")

    # ── 1. Tách target ──────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}' trong DataFrame!")

    y = (df[TARGET_COL] == 'Yes').astype(int)
    X = df.drop(columns=[TARGET_COL])

    # ── 2. Loại bỏ cột không dùng ───────────────────────────────────────────
    X = _drop_unused_columns(X)

    # ── 3. Encode categorical ────────────────────────────────────────────────
    X = _encode_categoricals(X)

    # ── 4. Train-test split (TRƯỚC khi scale để tránh leakage) ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,   # giữ tỷ lệ churn trong cả 2 tập
    )
    logger.info(
        f"Train: {X_train.shape}, Test: {X_test.shape} | "
        f"Churn rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}"
    )

    # ── 5. Scale numeric (fit ONLY trên train) ───────────────────────────────
    numeric_cols_present = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    logger.info(f"StandardScaler cho {len(numeric_cols_present)} cột numeric.")

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols_present] = scaler.fit_transform(X_train[numeric_cols_present])
    X_test[numeric_cols_present] = scaler.transform(X_test[numeric_cols_present])

    # ── 6. Lưu artifacts ────────────────────────────────────────────────────
    if save_artifacts:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(list(X_train.columns), ENCODER_PATH)
        logger.info(f"Đã lưu scaler tại: {SCALER_PATH}")
        logger.info(f"Đã lưu danh sách {len(X_train.columns)} cột tại: {ENCODER_PATH}")

    logger.info("Preprocessing hoàn tất.")
    return X_train, X_test, y_train, y_test


def load_scaler_and_columns():
    """
    Tiện ích: load lại scaler & danh sách cột đã lưu để dùng trong production/inference.
    """
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(ENCODER_PATH)
    return scaler, columns


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_and_prepare()
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"Churn rate    : train={y_train.mean():.3f} | test={y_test.mean():.3f}")
    print(f"\nDanh sách cột ({len(X_train.columns)}):")
    print(list(X_train.columns))