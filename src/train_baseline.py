

import sys
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGISTIC_MODEL_PATH, CV_FOLDS, RANDOM_STATE, SCORING_METRIC
from preprocessing import split_and_prepare


def train_logistic_regression(X_train, y_train, tune: bool = True):
    """
    Huấn luyện Logistic Regression.
    tune=True: chạy GridSearchCV để tìm hyperparameter C (độ regularization) tốt nhất,
    chấm theo Recall (SCORING_METRIC) vì đây là chỉ số quan trọng nhất của đồ án.
    """
    base_model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    }

    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring=SCORING_METRIC,
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params (Logistic Regression): {grid_search.best_params_}")
    print(f"Best CV {SCORING_METRIC}: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def run():
    X_train, X_test, y_train, y_test = split_and_prepare()
    model = train_logistic_regression(X_train, y_train, tune=True)

    joblib.dump(model, LOGISTIC_MODEL_PATH)
    print(f"Đã lưu mô hình Logistic Regression tại: {LOGISTIC_MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    run()