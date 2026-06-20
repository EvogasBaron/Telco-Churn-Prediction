
import sys
import os
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGB_MODEL_PATH, LGBM_MODEL_PATH, CV_FOLDS, RANDOM_STATE, SCORING_METRIC
from preprocessing import split_and_prepare
from imbalance_handler import get_scale_pos_weight


def train_xgboost(X_train, y_train, tune: bool = True):
    """Huấn luyện XGBoost với scale_pos_weight để xử lý mất cân bằng."""
    spw = get_scale_pos_weight(y_train)

    base_model = XGBClassifier(
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=30,
        scoring=SCORING_METRIC,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best params (XGBoost): {search.best_params_}")
    print(f"Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_


def train_lightgbm(X_train, y_train, tune: bool = True):
    """Huấn luyện LightGBM với scale_pos_weight."""
    verbose = -1,
    spw = get_scale_pos_weight(y_train)

    base_model = LGBMClassifier(
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "subsample": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=30,
        scoring=SCORING_METRIC,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best params (LightGBM): {search.best_params_}")
    print(f"Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_


def run():
    X_train, X_test, y_train, y_test = split_and_prepare()

    xgb_model = train_xgboost(X_train, y_train, tune=True)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    print(f"Đã lưu mô hình XGBoost tại: {XGB_MODEL_PATH}")

    lgbm_model = train_lightgbm(X_train, y_train, tune=True)
    joblib.dump(lgbm_model, LGBM_MODEL_PATH)
    print(f"Đã lưu mô hình LightGBM tại: {LGBM_MODEL_PATH}")

    return xgb_model, lgbm_model, X_test, y_test


if __name__ == "__main__":
    run()