
import sys
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STACKING_MODEL_PATH, RANDOM_STATE, CV_FOLDS
from preprocessing import split_and_prepare
from imbalance_handler import get_scale_pos_weight


def build_stacking_model(y_train):
    """Xây dựng kiến trúc Stacking Ensemble."""
    spw = get_scale_pos_weight(y_train)

    base_learners = [
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(scale_pos_weight=spw, n_estimators=200, max_depth=4, random_state=RANDOM_STATE,
                               eval_metric="logloss", use_label_encoder=False)),
    ]

    meta_learner = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        n_jobs=-1,
        passthrough=False,  # meta-learner chỉ học từ output của base learners
    )

    return stacking_model


def run():
    X_train, X_test, y_train, y_test = split_and_prepare()

    model = build_stacking_model(y_train)
    print("Đang huấn luyện Stacking Ensemble (có thể mất vài phút)...")
    model.fit(X_train, y_train)

    joblib.dump(model, STACKING_MODEL_PATH)
    print(f"Đã lưu mô hình Stacking Ensemble tại: {STACKING_MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    run()