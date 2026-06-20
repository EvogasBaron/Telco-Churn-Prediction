
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

from config import FIGURES_DIR, COMPARISON_REPORT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số màu sắc nhất quán cho tất cả biểu đồ
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#DD8452",
    "XGBoost":             "#55A868",
    "LightGBM":            "#C44E52",
    "Stacking Ensemble":   "#8172B3",
}
DEFAULT_COLOR = "#7F7F7F"


def _get_color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


def _get_y_scores(model, X_test) -> np.ndarray:
    """
    Lấy probability score cho lớp positive (churn=1).
    Tương thích với cả sklearn models và stacking ensemble.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Normalize về [0, 1] để dùng chung
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        raise ValueError(f"Model {type(model).__name__} không hỗ trợ predict_proba hay decision_function!")


# ─────────────────────────────────────────────────────────────────────────────
# 1. EVALUATE MODEL - tính toàn bộ metrics cho 1 model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Tính đầy đủ các chỉ số đánh giá cho 1 mô hình trên tập test.

    Parameters
    ----------
    model : sklearn-compatible model
    X_test, y_test : dữ liệu test
    model_name : tên hiển thị trong báo cáo
    threshold : ngưỡng quyết định (mặc định 0.5, có thể tùy chỉnh cho Recall cao hơn)
    verbose : in kết quả chi tiết ra console

    Returns
    -------
    dict chứa tất cả các metrics
    """
    y_scores = _get_y_scores(model, X_test)
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "Model":              model_name,
        "Accuracy":           accuracy_score(y_test, y_pred),
        "Precision":          precision_score(y_test, y_pred, zero_division=0),
        "Recall":             recall_score(y_test, y_pred, zero_division=0),
        "F1-Score":           f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":            roc_auc_score(y_test, y_scores),
        "PR-AUC":             average_precision_score(y_test, y_scores),
        "Threshold":          threshold,
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  KẾT QUẢ: {model_name}")
        print(f"{'=' * 50}")
        print(f"  Threshold  : {threshold}")
        print(f"  Accuracy   : {metrics['Accuracy']:.4f}")
        print(f"  Precision  : {metrics['Precision']:.4f}")
        print(f"  Recall     : {metrics['Recall']:.4f}  ← chỉ số ưu tiên")
        print(f"  F1-Score   : {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC    : {metrics['ROC-AUC']:.4f}")
        print(f"  PR-AUC     : {metrics['PR-AUC']:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    logger.info(f"[{model_name}] Recall={metrics['Recall']:.4f} | ROC-AUC={metrics['ROC-AUC']:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
    save: bool = True,
) -> None:
    """
    Vẽ Confusion Matrix dạng heatmap với số liệu tuyệt đối + tỷ lệ phần trăm.
    Hiển thị rõ False Negatives (bỏ sót churn) để nhấn mạnh bài toán business.
    """
    y_scores = _get_y_scores(model, X_test)
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ["No Churn\n(0)", "Churn\n(1)"]
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    # Ghi số liệu vào ô: "count\n(xx%)"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color=color, fontsize=12, fontweight="bold",
            )

    ax.set_xlabel("Dự đoán", fontsize=12)
    ax.set_ylabel("Thực tế", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_name = model_name.replace(" ", "_").lower()
        path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu Confusion Matrix tại: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC CURVE - so sánh nhiều model
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """
    Vẽ đường cong ROC cho nhiều mô hình trên cùng 1 biểu đồ.

    Parameters
    ----------
    models_dict : dict {model_name: model_object}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        y_scores = _get_y_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc = roc_auc_score(y_test, y_scores)
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc:.3f})",
            color=_get_color(name),
            linewidth=2,
        )

    # Đường baseline random classifier
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — So sánh các mô hình", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "roc_curve_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu ROC Curve tại: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRECISION-RECALL CURVE - phù hợp hơn với imbalanced data
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curve(
    models_dict: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> None:
    """
    Vẽ đường cong Precision-Recall.
    Đây là biểu đồ quan trọng hơn ROC khi data mất cân bằng vì:
    - ROC có thể "đẹp giả" do TN nhiều làm FPR nhỏ
    - PR curve phản ánh thực tế hơn với lớp thiểu số (churn)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline_pr = y_test.mean()
    ax.axhline(y=baseline_pr, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (P = {baseline_pr:.3f})")

    for name, model in models_dict.items():
        y_scores = _get_y_scores(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        ap = average_precision_score(y_test, y_scores)
        ax.plot(
            recall, precision,
            label=f"{name} (AP = {ap:.3f})",
            color=_get_color(name),
            linewidth=2,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — So sánh các mô hình", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "pr_curve_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu PR Curve tại: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Gộp danh sách metrics từ nhiều mô hình thành bảng so sánh.
    Sắp xếp theo Recall giảm dần (mục tiêu chính của đồ án).

    Parameters
    ----------
    metrics_list : list of dict, mỗi dict là output của evaluate_model()

    Returns
    -------
    pd.DataFrame với index là tên model, sắp xếp theo Recall
    """
    df = pd.DataFrame(metrics_list)
    df = df.set_index("Model")
    df = df.sort_values("Recall", ascending=False)

    # Đánh dấu mô hình tốt nhất theo từng chỉ số để dễ đọc báo cáo
    logger.info("\n" + "=" * 60)
    logger.info("BẢNG SO SÁNH MÔ HÌNH (sắp xếp theo Recall ↓)")
    logger.info("=" * 60)
    logger.info(f"\n{df.round(4).to_string()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE COMPARISON REPORT (Markdown)
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_report(comparison_df: pd.DataFrame) -> None:
    """
    Xuất bảng so sánh thành file Markdown để đưa vào báo cáo đồ án.
    Bao gồm: timestamp, bảng số liệu, phân tích ngắn, nhận xét business.
    """
    os.makedirs(os.path.dirname(COMPARISON_REPORT_PATH), exist_ok=True)

    best_model = comparison_df.index[0]
    best_recall = comparison_df.loc[best_model, "Recall"]
    best_auc = comparison_df.loc[best_model, "ROC-AUC"]

    lines = [
        "# Báo cáo So sánh Mô hình — Churn Prediction",
        "",
        f"> **Thời gian tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Mục tiêu đánh giá",
        "",
        "Đồ án ưu tiên **Recall** (tỷ lệ phát hiện đúng khách churn) vì:",
        "- Chi phí bỏ sót khách hàng rời bỏ (False Negative) > chi phí",
        "  tiếp cận nhầm khách trung thành (False Positive).",
        "- Mỗi khách hàng rời bỏ tương đương mất toàn bộ CLTV tương lai.",
        "",
        "## 2. Kết quả so sánh",
        "",
        comparison_df.round(4).to_markdown(),
        "",
        "## 3. Nhận xét",
        "",
        f"- **Mô hình tốt nhất** (theo Recall): **{best_model}**",
        f"  - Recall = {best_recall:.4f} ({best_recall:.1%} khách churn được phát hiện đúng)",
        f"  - ROC-AUC = {best_auc:.4f}",
        "",
        "### Phân tích trade-off Precision ↔ Recall",
        "",
    ]

    for model_name in comparison_df.index:
        row = comparison_df.loc[model_name]
        lines.append(
            f"- **{model_name}**: Precision={row['Precision']:.3f}, "
            f"Recall={row['Recall']:.3f}, F1={row['F1-Score']:.3f}, "
            f"AUC={row['ROC-AUC']:.3f}"
        )

    lines += [
        "",
        "## 4. Khuyến nghị triển khai",
        "",
        f"Sử dụng **{best_model}** làm mô hình chính cho hệ thống cảnh báo sớm.",
        "Kết hợp với phân tích SHAP để giải thích từng trường hợp cụ thể.",
        "Xem xét điều chỉnh threshold dự đoán (mặc định 0.5) tùy ngưỡng chấp nhận",
        "rủi ro của bộ phận kinh doanh.",
        "",
        "---",
        "*Báo cáo được tạo tự động bởi evaluation.py*",
    ]

    report_content = "\n".join(lines)
    with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Đã lưu báo cáo so sánh tại: {COMPARISON_REPORT_PATH}")
    print(f"\n✓ Báo cáo markdown đã lưu tại: {COMPARISON_REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BONUS: THRESHOLD TUNING (tối ưu Recall theo ngưỡng)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    min_precision: float = 0.4,
    save: bool = True,
) -> float:

    y_scores = _get_y_scores(model, X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    # precision_recall_curve trả về len(thresholds) = len(precisions) - 1
    # vì điểm cuối cùng là (precision=1, recall=0) không có threshold tương ứng
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Lọc những threshold có precision >= min_precision
    valid_mask = precisions >= min_precision
    if not valid_mask.any():
        logger.warning(
            f"Không có threshold nào đạt Precision >= {min_precision}. "
            f"Trả về threshold mặc định 0.5"
        )
        return 0.5

    best_idx = recalls[valid_mask].argmax()
    optimal_threshold = thresholds[valid_mask][best_idx]
    best_recall = recalls[valid_mask][best_idx]
    best_precision = precisions[valid_mask][best_idx]

    print(
        f"\n[{model_name}] Threshold tối ưu = {optimal_threshold:.3f} "
        f"| Precision = {best_precision:.3f} | Recall = {best_recall:.3f}"
    )

    # Vẽ biểu đồ threshold vs metrics
    if save:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", linewidth=2)
        ax.plot(thresholds, recalls, label="Recall", color="#DD8452", linewidth=2)
        ax.axvline(
            x=optimal_threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"Optimal threshold = {optimal_threshold:.3f}"
        )
        ax.axhline(y=min_precision, color="gray", linestyle=":", linewidth=1,
                   label=f"Min Precision = {min_precision}")
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Threshold Tuning — {model_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        safe_name = model_name.replace(" ", "_").lower()
        path = os.path.join(FIGURES_DIR, f"threshold_tuning_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Đã lưu biểu đồ threshold tuning tại: {path}")
        plt.close()

    return float(optimal_threshold)


if __name__ == "__main__":
    # Smoke test với DummyClassifier để kiểm tra không crash
    from sklearn.dummy import DummyClassifier
    import numpy as np

    np.random.seed(42)
    n = 500
    X_mock = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y_mock = pd.Series(np.random.choice([0, 1], size=n, p=[0.73, 0.27]), name="Churn")

    X_tr, X_te = X_mock[:400], X_mock[400:]
    y_tr, y_te = y_mock[:400], y_mock[400:]

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_tr, y_tr)

    metrics = evaluate_model(dummy, X_te, y_te, model_name="DummyClassifier")
    table = build_comparison_table([metrics])
    print(table)

    plot_confusion_matrix(dummy, X_te, y_te, "DummyClassifier", save=False)
    plot_roc_curve({"DummyClassifier": dummy}, X_te, y_te, save=False)
    plot_precision_recall_curve({"DummyClassifier": dummy}, X_te, y_te, save=False)
    opt_thresh = find_optimal_threshold(dummy, X_te, y_te, "DummyClassifier", save=False)
    print(f"Optimal threshold: {opt_thresh:.3f}")
    print("Smoke test PASSED ✓")