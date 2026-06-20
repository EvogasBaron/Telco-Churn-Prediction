
import pandas as pd
import logging

from config import RAW_DATA_PATH, CLEANED_DATA_PATH, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc file CSV thô."""
    logger.info(f"Đang đọc dữ liệu thô từ: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dữ liệu thô có shape: {df.shape}")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu TotalCharges về số, xử lý các giá trị lỗi (khoảng trắng)."""
    df = df.copy()
    n_before = df["TotalCharges"].isna().sum()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isna().sum()
    logger.info(f"Số dòng TotalCharges bị lỗi/thiếu sau ép kiểu: {n_missing} (trước: {n_before})")

    # Kiểm chứng giả thuyết: các dòng thiếu đều có tenure = 0
    missing_rows = df[df["TotalCharges"].isna()]
    if len(missing_rows) > 0:
        assert (missing_rows["tenure"] == 0).all(), \
            "Phát hiện dòng thiếu TotalCharges nhưng tenure != 0 -> cần kiểm tra lại giả định!"
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        logger.info("Đã điền 0 cho các khách hàng mới (tenure=0) có TotalCharges thiếu.")

    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột dạng chuỗi: loại bỏ khoảng trắng dư, đồng nhất nhãn."""
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển nhãn target Yes/No -> 1/0 để thuận tiện cho mô hình & EDA."""
    df = df.copy()
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def drop_duplicates_and_check(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ trùng lặp (nếu có) và log số liệu kiểm tra chất lượng dữ liệu."""
    df = df.copy()
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.warning(f"Phát hiện {n_dup} dòng trùng lặp -> đã loại bỏ.")
        df = df.drop_duplicates()

    null_summary = df.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    if len(null_summary) > 0:
        logger.warning(f"Các cột còn giá trị null sau làm sạch:\n{null_summary}")
    else:
        logger.info("Không còn giá trị null nào sau bước làm sạch.")

    return df


def run_cleaning_pipeline(save: bool = True) -> pd.DataFrame:
    """Hàm tổng hợp toàn bộ bước làm sạch, gọi từ pipeline.py hoặc main.py."""
    df = load_raw_data()
    df = clean_categorical_columns(df)
    df = fix_total_charges(df)
    df = encode_target(df)
    df = drop_duplicates_and_check(df)

    if save:
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Đã lưu dữ liệu đã làm sạch tại: {CLEANED_DATA_PATH}")

    return df


if __name__ == "__main__":
    cleaned_df = run_cleaning_pipeline()
    print(cleaned_df.info())
    print(cleaned_df.head())