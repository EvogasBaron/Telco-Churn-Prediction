
import pandas as pd
import numpy as np
import logging

from config import FEATURED_DATA_PATH, CLEANED_DATA_PATH, ADDON_SERVICE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_charge_velocity(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    safe_tenure = df["tenure"].replace(0, 1)
    avg_historical_charge = df["TotalCharges"] / safe_tenure

    # Tránh chia cho 0 nếu avg_historical_charge = 0 (trường hợp TotalCharges=0)
    avg_historical_charge = avg_historical_charge.replace(0, np.nan)
    df["charge_velocity"] = df["MonthlyCharges"] / avg_historical_charge

    # Khách mới (tenure=0) chưa có lịch sử -> mặc định velocity = 1 (trung tính)
    df.loc[df["tenure"] == 0, "charge_velocity"] = 1.0
    df["charge_velocity"] = df["charge_velocity"].fillna(1.0)

    # Giới hạn outlier cực đoan (winsorize) để tránh ảnh hưởng mô hình tuyến tính
    upper_bound = df["charge_velocity"].quantile(0.99)
    df["charge_velocity"] = df["charge_velocity"].clip(upper=upper_bound)

    return df


def add_tenure_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Phân nhóm khách hàng theo cột mốc vòng đời (lifecycle)."""
    df = df.copy()
    bins = [-1, 12, 24, 48, 72]
    labels = ["New_0_12m", "Growing_12_24m", "Mature_24_48m", "Loyal_48_72m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement Score = tổng số dịch vụ giá trị gia tăng khách hàng đang đăng ký.
    Đo lường mức độ "gắn bó/trung thành" về mặt sử dụng dịch vụ.
    """
    df = df.copy()
    available_cols = [c for c in ADDON_SERVICE_COLS if c in df.columns]
    df["engagement_score"] = (df[available_cols] == "Yes").sum(axis=1)
    return df


def add_cltv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLTV proxy (ước lượng giá trị khách hàng trọn đời, đơn giản hóa):
    CLTV ~ tenure * MonthlyCharges
    Dùng để đưa business context vào mô hình & dashboard (ai đáng giữ chân nhất).
    """
    df = df.copy()
    df["cltv_proxy"] = df["tenure"] * df["MonthlyCharges"]
    return df


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contract Risk Score: hợp đồng theo tháng có rủi ro rời bỏ cao nhất.
    Month-to-month = 2 (rủi ro cao) ; One year = 1 ; Two year = 0 (rủi ro thấp)
    """
    df = df.copy()
    mapping = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["Contract"].map(mapping)
    return df


def add_payment_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment Risk Flag: theo EDA trên dataset gốc, khách thanh toán bằng
    'Electronic check' có tỷ lệ churn cao bất thường so với các phương thức khác.
    """
    df = df.copy()
    df["payment_risk_flag"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def add_price_sensitivity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price Sensitivity Index = MonthlyCharges / (engagement_score + 1)
    Khách trả nhiều tiền nhưng dùng ít dịch vụ giá trị gia tăng -> nhạy cảm về giá,
    dễ cảm thấy "không đáng tiền" -> nguy cơ rời bỏ cao hơn.
    (+1 để tránh chia cho 0 khi engagement_score = 0)
    """
    df = df.copy()
    df["price_sensitivity_index"] = df["MonthlyCharges"] / (df["engagement_score"] + 1)
    return df


def run_feature_engineering(df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    """Hàm tổng hợp toàn bộ bước feature engineering."""
    if df is None:
        df = pd.read_csv(CLEANED_DATA_PATH)

    logger.info("Bắt đầu tạo các đặc trưng mới...")
    df = add_engagement_score(df)          # phải chạy trước price_sensitivity_index
    df = add_charge_velocity(df)
    df = add_tenure_binning(df)
    df = add_cltv_proxy(df)
    df = add_contract_risk_score(df)
    df = add_payment_risk_flag(df)
    df = add_price_sensitivity_index(df)

    new_cols = [
        "charge_velocity", "tenure_group", "engagement_score", "cltv_proxy",
        "contract_risk_score", "payment_risk_flag", "price_sensitivity_index",
    ]
    logger.info(f"Đã tạo thành công {len(new_cols)} đặc trưng mới: {new_cols}")

    if save:
        df.to_csv(FEATURED_DATA_PATH, index=False)
        logger.info(f"Đã lưu dữ liệu đã feature engineering tại: {FEATURED_DATA_PATH}")

    return df


if __name__ == "__main__":
    featured_df = run_feature_engineering()
    print(featured_df[[
        "tenure", "MonthlyCharges", "TotalCharges", "charge_velocity",
        "tenure_group", "engagement_score", "cltv_proxy",
        "contract_risk_score", "payment_risk_flag", "price_sensitivity_index",
    ]].head(10))