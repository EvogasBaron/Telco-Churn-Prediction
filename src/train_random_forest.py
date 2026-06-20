
import sys
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RF_MODEL_PATH, CV_FOLDS, RANDOM_STATE, SCORING_METRIC, FIGURES_DIR
from preprocessing import split_and_prepare

import matplotlib.pyplot as plt


def train_random_forest(X_train, y_train, tune: bool = True):
    """Huấn luyện Random Forest với class_weight='balanced' + RandomizedSearchCV."""
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if not tune:
        base_model.fit(X_train, y_train)
        return base_model

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
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

    print(f"Best params (Random Forest): {search.best_params_}")
    print(f"Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_


def plot_feature_importance(model, feature_names, top_n: int = 15, save: bool = True):
    """Vẽ biểu đồ Top N feature quan trọng nhất theo Random Forest."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", color="teal")
    plt.title(f"Top {top_n} Feature Importance - Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "feature_importance_rf.png")
        plt.savefig(path, dpi=150)
        print(f"Đã lưu biểu đồ feature importance tại: {path}")
    plt.close()

    return importances


def run():
    X_train, X_test, y_train, y_test = split_and_prepare()
    model = train_random_forest(X_train, y_train, tune=True)

    joblib.dump(model, RF_MODEL_PATH)
    print(f"Đã lưu mô hình Random Forest tại: {RF_MODEL_PATH}")

    plot_feature_importance(model, X_train.columns)

    return model, X_test, y_test


if __name__ == "__main__":
    run()