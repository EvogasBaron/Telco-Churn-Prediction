# Báo cáo So sánh Mô hình — Churn Prediction

> **Thời gian tạo:** 2026-06-19 23:49:40

## 1. Mục tiêu đánh giá

Đồ án ưu tiên **Recall** (tỷ lệ phát hiện đúng khách churn) vì:
- Chi phí bỏ sót khách hàng rời bỏ (False Negative) > chi phí
  tiếp cận nhầm khách trung thành (False Positive).
- Mỗi khách hàng rời bỏ tương đương mất toàn bộ CLTV tương lai.

## 2. Kết quả so sánh

| Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC |   Threshold |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|------------:|
| Logistic Regression |     0.7601 |      0.5359 |   0.7193 |     0.6142 |    0.834  |   0.6296 |         0.5 |
| LightGBM            |     0.7658 |      0.5464 |   0.6925 |     0.6108 |    0.8308 |   0.6407 |         0.5 |
| Stacking Ensemble   |     0.7679 |      0.5533 |   0.6524 |     0.5988 |    0.8292 |   0.6347 |         0.5 |
| XGBoost             |     0.7665 |      0.551  |   0.6497 |     0.5963 |    0.8145 |   0.6182 |         0.5 |
| Random Forest       |     0.7686 |      0.5563 |   0.6337 |     0.5925 |    0.8223 |   0.6061 |         0.5 |

## 3. Nhận xét

- **Mô hình tốt nhất** (theo Recall): **Logistic Regression**
  - Recall = 0.7193 (71.9% khách churn được phát hiện đúng)
  - ROC-AUC = 0.8340

### Phân tích trade-off Precision ↔ Recall

- **Logistic Regression**: Precision=0.536, Recall=0.719, F1=0.614, AUC=0.834
- **LightGBM**: Precision=0.546, Recall=0.693, F1=0.611, AUC=0.831
- **Stacking Ensemble**: Precision=0.553, Recall=0.652, F1=0.599, AUC=0.829
- **XGBoost**: Precision=0.551, Recall=0.650, F1=0.596, AUC=0.815
- **Random Forest**: Precision=0.556, Recall=0.634, F1=0.593, AUC=0.822

## 4. Khuyến nghị triển khai

Sử dụng **Logistic Regression** làm mô hình chính cho hệ thống cảnh báo sớm.
Kết hợp với phân tích SHAP để giải thích từng trường hợp cụ thể.
Xem xét điều chỉnh threshold dự đoán (mặc định 0.5) tùy ngưỡng chấp nhận
rủi ro của bộ phận kinh doanh.

---
*Báo cáo được tạo tự động bởi evaluation.py*