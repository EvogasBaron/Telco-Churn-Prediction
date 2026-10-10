# Telco Customer Churn Prediction

Dự án dự đoán khách hàng rời bỏ dịch vụ (churn) của một công ty viễn thông, dựa trên bộ dữ liệu Telco Customer Churn. Dự án xây dựng pipeline end-to-end: làm sạch dữ liệu, tạo đặc trưng mới, xử lý mất cân bằng lớp, huấn luyện và so sánh 5 mô hình, sau đó giải thích mô hình bằng SHAP.

Chỉ số được ưu tiên là Recall (tỷ lệ phát hiện đúng khách sắp rời bỏ), vì bỏ sót một khách hàng churn gây thiệt hại lớn hơn việc tiếp cận nhầm một khách hàng trung thành.

## Dữ liệu

- File: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Quy mô: 7.043 khách hàng, 20 đặc trưng cùng cột nhãn `Churn` (Yes/No).
- Nhóm đặc trưng gốc: nhân khẩu học (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), dịch vụ đăng ký (`PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`, ...), hợp đồng và thanh toán (`Contract`, `PaymentMethod`, `PaperlessBilling`), chi phí (`tenure`, `MonthlyCharges`, `TotalCharges`).

## Quy trình xử lý

1. Làm sạch (`src/data_cleaning.py`): ép kiểu `TotalCharges` về số, điền 0 cho các khách mới có `tenure = 0`, chuẩn hóa chuỗi, loại bản ghi trùng.
2. Tạo đặc trưng (`src/feature_engineering.py`): thêm 7 đặc trưng.
   - `charge_velocity`: tỷ lệ giữa cước tháng hiện tại và cước trung bình lịch sử.
   - `tenure_group`: nhóm vòng đời khách hàng (0-12, 12-24, 24-48, 48-72 tháng).
   - `engagement_score`: số dịch vụ giá trị gia tăng đang đăng ký.
   - `cltv_proxy`: `tenure × MonthlyCharges`, ước lượng giá trị khách hàng.
   - `contract_risk_score`: mức rủi ro theo loại hợp đồng (theo tháng = 2, một năm = 1, hai năm = 0).
   - `payment_risk_flag`: cờ cho khách thanh toán bằng Electronic check.
   - `price_sensitivity_index`: `MonthlyCharges / (engagement_score + 1)`.
3. Tiền xử lý (`src/preprocessing.py`): bỏ cột không dùng, One-Hot Encoding, chia train/test 80/20 có phân tầng, chuẩn hóa số liệu bằng `StandardScaler` (chỉ fit trên tập train để tránh rò rỉ dữ liệu). Scaler và danh sách cột được lưu vào `models/`.
4. Xử lý mất cân bằng (`src/imbalance_handler.py`): SMOTE trên tập train, kèm `class_weight` hoặc `scale_pos_weight` cho các mô hình hỗ trợ.
5. Huấn luyện: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM và Stacking Ensemble. Siêu tham số được tinh chỉnh bằng GridSearchCV hoặc RandomizedSearchCV với 5-fold CV, chấm điểm theo Recall.
6. Đánh giá (`src/evaluation.py`): Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, ma trận nhầm lẫn, đường cong ROC và Precision-Recall.
7. Giải thích (`src/explainability.py`): SHAP cho mô hình có Recall cao nhất (bỏ qua nếu đó là Stacking Ensemble).

## Kết quả

Kết quả trên tập test (ngưỡng dự đoán 0,5), sắp xếp theo Recall. Chi tiết xem tại `reports/model_comparison_report.md`.

| Mô hình             | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:--------------------|---------:|----------:|-------:|---------:|--------:|-------:|
| Logistic Regression |   0,7601 |    0,5359 | 0,7193 |   0,6142 |  0,8340 | 0,6296 |
| LightGBM            |   0,7658 |    0,5464 | 0,6925 |   0,6108 |  0,8308 | 0,6407 |
| Stacking Ensemble   |   0,7679 |    0,5533 | 0,6524 |   0,5988 |  0,8292 | 0,6347 |
| XGBoost             |   0,7665 |    0,5510 | 0,6497 |   0,5963 |  0,8145 | 0,6182 |
| Random Forest       |   0,7686 |    0,5563 | 0,6337 |   0,5925 |  0,8223 | 0,6061 |

Logistic Regression có Recall cao nhất (71,9%) và ROC-AUC cao nhất (0,834), đồng thời dễ giải thích, nên được khuyến nghị làm mô hình chính. Các biểu đồ nằm trong `reports/figures/`.


## Cài đặt

Dự án được phát triển trên Python 3.14. Các phiên bản trong `requirements.txt` lấy từ môi trường ảo của dự án; chưa kiểm tra trên phiên bản Python khác.

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Cài thư viện
pip install -r requirements.txt
```

Nên giữ đúng phiên bản `pandas` (3.x) trong `requirements.txt`. Mã nguồn hiện giữ nhãn `Churn` ở dạng chuỗi "Yes"/"No" qua các bước làm sạch và tiền xử lý, và hành vi này phụ thuộc vào cách pandas 3 xử lý kiểu dữ liệu chuỗi.

## Cách chạy

Chạy từ thư mục gốc của dự án.

Chạy toàn bộ pipeline (làm sạch, tạo đặc trưng, huấn luyện, đánh giá, SHAP):

```bash
python -m src.pipeline
```

Lệnh `python src/pipeline.py` sẽ báo lỗi `No module named 'models'`, vì `pipeline.py` import `models.train_*` và cần thư mục gốc nằm trong đường dẫn tìm module. Hãy dùng `python -m src.pipeline` như trên.

Bước tinh chỉnh siêu tham số (RandomizedSearchCV, 30 tổ hợp × 5 fold cho mỗi mô hình) tốn nhiều thời gian. Để chạy nhanh hơn, tắt tinh chỉnh:

```bash
python -c "from src.pipeline import run_full_pipeline; run_full_pipeline(use_smote=True, tune_models=False)"
```

Chạy riêng từng bước:

```bash
python src/data_cleaning.py         # tạo data/processed/cleaned_data.csv
python src/feature_engineering.py   # tạo data/processed/featured_data.csv
python src/preprocessing.py         # chia dữ liệu, lưu scaler và danh sách cột
python src/train_baseline.py        # Logistic Regression
python src/train_random_forest.py   # Random Forest
python src/train_boosting.py        # XGBoost và LightGBM
python src/train_stacking.py        # Stacking Ensemble
```

Các script `train_*.py` chạy riêng huấn luyện trên dữ liệu chưa qua SMOTE, còn `pipeline.py` áp dụng SMOTE trên tập train, nên kết quả của hai cách chạy có thể khác nhau.

## Đầu ra

- `data/processed/`: dữ liệu sau làm sạch và sau tạo đặc trưng.
- `models/`: các mô hình `.pkl`, `scaler.pkl`, `encoder_columns.pkl`.
- `reports/figures/`: biểu đồ đánh giá.
- `reports/model_comparison_report.md`: bảng so sánh mô hình và khuyến nghị.

## Tham số chính

Các tham số nằm trong `src/config.py`: `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `CV_FOLDS = 5`, `SCORING_METRIC = "recall"`.
