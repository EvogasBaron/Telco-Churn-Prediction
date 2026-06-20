
import os
import shap
import matplotlib.pyplot as plt

from config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

import shap
from sklearn.linear_model import LogisticRegression


def explain_global(model, X, model_name=""):
    # Kiểm tra xem mô hình hiện tại có phải là Logistic Regression không
    if isinstance(model, LogisticRegression):
        # Nếu là Logistic Regression, dùng LinearExplainer kết hợp với dữ liệu mẫu (X)
        explainer = shap.LinearExplainer(model, X)
    else:
        # Nếu là các mô hình còn lại (Random Forest, XGBoost, LightGBM...), dùng TreeExplainer
        explainer = shap.TreeExplainer(model)

    # Tính toán shap_values
    shap_values = explainer.shap_values(X)

    return explainer, shap_values


def explain_single_customer(explainer, shap_values, X_test, customer_index: int, model_name: str = "model"):
    """
    Giải thích lý do dự đoán cho 1 khách hàng cụ thể (theo index trong X_test).
    Dùng cho dashboard: khi nhân viên CSKH muốn biết "vì sao khách này bị gắn cờ rủi ro cao".
    """
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    shap.force_plot(
        expected_value,
        values_to_plot[customer_index],
        X_test.iloc[customer_index],
        matplotlib=True,
        show=False,
    )

    path = os.path.join(FIGURES_DIR, f"shap_force_{model_name}_customer_{customer_index}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu SHAP force plot cho khách hàng #{customer_index} tại: {path}")
    plt.close()


def get_top_global_features(explainer, shap_values, X_test, top_n: int = 10):
    """Trả về danh sách top N feature ảnh hưởng nhiều nhất (dùng để in vào báo cáo)."""
    import numpy as np
    import pandas as pd

    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(values_to_plot).mean(axis=0)
    importance_df = pd.Series(mean_abs_shap, index=X_test.columns)
    importance_df = importance_df.sort_values(ascending=False).head(top_n)
    return importance_df