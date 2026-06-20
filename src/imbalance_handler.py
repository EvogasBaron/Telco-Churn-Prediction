
import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Áp dụng SMOTE để cân bằng lớp trên tập huấn luyện.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features của tập train (SAU khi đã encode và scale).
    y_train : pd.Series
        Labels của tập train (0/1).
    sampling_strategy : float hoặc "auto"
        Tỷ lệ minority/majority sau khi oversample.
        "auto" = cân bằng hoàn toàn 50/50 (mặc định).
        Ví dụ 0.5 = minority sẽ bằng 50% majority.
    k_neighbors : int
        Số láng giềng gần nhất dùng để tạo sample synthetic (mặc định 5).

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.Series
        Dữ liệu đã được cân bằng, giữ nguyên tên cột.
    """
    logger.info(
        f"Trước SMOTE: total={len(y_train)}, "
        f"churn={y_train.sum()} ({y_train.mean():.1%}), "
        f"no-churn={len(y_train) - y_train.sum()}"
    )

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )

    X_resampled_np, y_resampled_np = smote.fit_resample(X_train, y_train)

    # Giữ lại tên cột và index để tương thích với các bước sau
    X_resampled = pd.DataFrame(X_resampled_np, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled_np, name=y_train.name)

    logger.info(
        f"Sau SMOTE:  total={len(y_resampled)}, "
        f"churn={y_resampled.sum()} ({y_resampled.mean():.1%}), "
        f"no-churn={len(y_resampled) - y_resampled.sum()}"
    )

    return X_resampled, y_resampled


def get_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Tính scale_pos_weight cho XGBoost / LightGBM.

    Công thức: count(negative class) / count(positive class)
    Ví dụ: 5000 no-churn / 1869 churn ≈ 2.68
    -> Model sẽ phạt lỗi trên lớp churn nặng gấp 2.68 lần lớp no-churn.

    Parameters
    ----------
    y_train : pd.Series
        Labels tập train (0 = no-churn, 1 = churn).

    Returns
    -------
    float
        Giá trị scale_pos_weight để truyền vào XGBClassifier / LGBMClassifier.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    if n_positive == 0:
        raise ValueError("Tập train không có mẫu positive (churn)!")

    spw = n_negative / n_positive
    logger.info(
        f"scale_pos_weight = {n_negative}/{n_positive} = {spw:.4f} "
        f"(no-churn/churn)"
    )
    return spw


def get_class_distribution(y: pd.Series, label: str = "dataset") -> dict:
    """
    Tiện ích: in phân phối lớp để kiểm tra nhanh.

    Returns
    -------
    dict với keys 'total', 'churn', 'no_churn', 'churn_rate'
    """
    n_total = len(y)
    n_churn = y.sum()
    n_no_churn = n_total - n_churn
    churn_rate = n_churn / n_total

    info = {
        "total": n_total,
        "churn": int(n_churn),
        "no_churn": int(n_no_churn),
        "churn_rate": round(float(churn_rate), 4),
    }
    logger.info(
        f"[{label}] Total: {n_total} | "
        f"Churn: {n_churn} ({churn_rate:.1%}) | "
        f"No-Churn: {n_no_churn} ({1 - churn_rate:.1%})"
    )
    return info


if __name__ == "__main__":
    # Smoke test: kiểm tra nhanh không cần dữ liệu thật
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    # Giả lập dataset mất cân bằng: 730 no-churn, 270 churn
    y_mock = pd.Series([0] * 730 + [1] * 270, name="Churn")
    X_mock = pd.DataFrame(
        np.random.randn(1000, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )

    print("=== Test get_class_distribution ===")
    get_class_distribution(y_mock, label="mock dataset")

    print("\n=== Test get_scale_pos_weight ===")
    spw = get_scale_pos_weight(y_mock)
    print(f"scale_pos_weight = {spw:.4f}")

    print("\n=== Test apply_smote ===")
    X_res, y_res = apply_smote(X_mock, y_mock)
    get_class_distribution(y_res, label="after SMOTE")