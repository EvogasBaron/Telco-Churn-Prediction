

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import run_cleaning_pipeline
from feature_engineering import run_feature_engineering
from preprocessing import split_and_prepare
from imbalance_handler import apply_smote

from models.train_baseline import train_logistic_regression
from models.train_random_forest import train_random_forest, plot_feature_importance
from models.train_boosting import train_xgboost, train_lightgbm
from models.train_stacking import build_stacking_model

from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, build_comparison_table, save_comparison_report,
)
from explainability import explain_global, get_top_global_features

import joblib
from config import (
    LOGISTIC_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    LGBM_MODEL_PATH, STACKING_MODEL_PATH,
)


def run_full_pipeline(use_smote: bool = True, tune_models: bool = True):
    """Chạy toàn bộ pipeline end-to-end, in kết quả so sánh ra console + file."""

    print("\n========== BƯỚC 1: LÀM SẠCH DỮ LIỆU ==========")
    cleaned_df = run_cleaning_pipeline()

    print("\n========== BƯỚC 2: KỸ NGHỆ ĐẶC TRƯNG ==========")
    featured_df = run_feature_engineering(cleaned_df)

    print("\n========== BƯỚC 3: TIỀN XỬ LÝ (ENCODE + SCALE + SPLIT) ==========")
    X_train, X_test, y_train, y_test = split_and_prepare(featured_df)

    if use_smote:
        print("\n========== BƯỚC 4: ÁP DỤNG SMOTE TRÊN TRAIN SET ==========")
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    print("\n========== BƯỚC 5: HUẤN LUYỆN CÁC MÔ HÌNH ==========")

    print("\n--- 5.1 Logistic Regression (Baseline) ---")
    lr_model = train_logistic_regression(X_train_resampled, y_train_resampled, tune=tune_models)
    joblib.dump(lr_model, LOGISTIC_MODEL_PATH)

    print("\n--- 5.2 Random Forest ---")
    rf_model = train_random_forest(X_train_resampled, y_train_resampled, tune=tune_models)
    joblib.dump(rf_model, RF_MODEL_PATH)
    plot_feature_importance(rf_model, X_train.columns)

    print("\n--- 5.3 XGBoost ---")
    xgb_model = train_xgboost(X_train_resampled, y_train_resampled, tune=tune_models)
    joblib.dump(xgb_model, XGB_MODEL_PATH)

    print("\n--- 5.4 LightGBM ---")
    lgbm_model = train_lightgbm(X_train_resampled, y_train_resampled, tune=tune_models)
    joblib.dump(lgbm_model, LGBM_MODEL_PATH)

    print("\n--- 5.5 Stacking Ensemble (nâng cao) ---")
    stacking_model = build_stacking_model(y_train_resampled)
    stacking_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(stacking_model, STACKING_MODEL_PATH)

    print("\n========== BƯỚC 6: ĐÁNH GIÁ & SO SÁNH MÔ HÌNH ==========")
    models_dict = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
        "Stacking Ensemble": stacking_model,
    }

    metrics_list = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics_list.append(metrics)
        plot_confusion_matrix(model, X_test, y_test, name)

    comparison_df = build_comparison_table(metrics_list)
    print("\n===== BẢNG SO SÁNH CUỐI CÙNG (sắp xếp theo Recall) =====")
    print(comparison_df.round(4))
    save_comparison_report(comparison_df)

    plot_roc_curve(models_dict, X_test, y_test)
    plot_precision_recall_curve(models_dict, X_test, y_test)

    print("\n========== BƯỚC 7: GIẢI THÍCH MÔ HÌNH TỐT NHẤT (SHAP) ==========")
    best_model_name = comparison_df.index[0]
    best_model = models_dict[best_model_name]
    print(f"Mô hình có Recall cao nhất: {best_model_name}")

    if best_model_name != "Stacking Ensemble":  # SHAP TreeExplainer không hỗ trợ trực tiếp Stacking
        explainer, shap_values = explain_global(best_model, X_test, model_name=best_model_name)
        top_features = get_top_global_features(explainer, shap_values, X_test)
        print(f"\nTop 10 yếu tố ảnh hưởng đến churn nhiều nhất:\n{top_features}")

    print("\n========== HOÀN TẤT PIPELINE ==========")
    return comparison_df, models_dict


if __name__ == "__main__":
    run_full_pipeline(use_smote=True, tune_models=True)