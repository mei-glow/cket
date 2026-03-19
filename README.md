# Phân Tích Hành Vi Người Dùng - Vòng Chung Kết

## Tổng Quan

Dự án này triển khai các mô hình học máy cho cuộc thi DATAFLOW 2026, tập trung vào phân tích hành vi người dùng để hỗ trợ quyết định chuỗi cung ứng. Mục tiêu là dự đoán 6 thuộc tính phân loại (`attr_1` đến `attr_6`) từ các chuỗi hành vi người dùng.

## Cấu Trúc Dự Án

```
├── app.py                 # Ứng dụng web demo Streamlit
├── app_2.py              # Ứng dụng Streamlit thay thế
├── eda.ipynb             # Notebook Phân tích Khám phá Dữ liệu
├── requirements.txt      # Các thư viện Python cần thiết
├── transformer_final.py  # Triển khai mô hình transformer cuối cùng
├── transformer_raw.py    # Triển khai transformer thô
├── data/                 # Các tệp dữ liệu
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── X_test.csv
│   ├── Y_train.csv
│   └── Y_val.csv
├── eda/                  # Đầu ra và trực quan hóa EDA
├── src/                  # Mã nguồn cho các mô hình
│   ├── gru_weighted_l2_model.py
│   └── new_TCN_finetuned.py
├── t_max/               # Đầu ra và artifacts mô hình cuối cùng
│   ├── model_config.json
│   ├── submission_A.csv
│   ├── attention_maps/
│   └── visualizations/
└── transformer_raw/     # Đầu ra transformer thô
    ├── submission_A.csv
    ├── submission_B.csv
    └── attention_maps/
```

## Các Mô Hình Được Triển Khai

1. **GRU với Attention**: Bộ mã hóa BiGRU với pooling attention có mặt nạ và các đầu phân loại rời rạc
2. **Mạng Tích Chập Thời Gian (TCN)**: Kiến trúc tích chập cho mô hình hóa chuỗi
3. **Transformer**: Mô hình dựa trên multi-head self-attention với huấn luyện ensemble

## Tính Năng Chính

- **Hàm Mất MSE Chuẩn hóa Có Trọng số**: Hàm mất tùy chỉnh tính đến các thang đo thuộc tính khác nhau
- **Ensemble Đa Seed**: Cải thiện độ robust thông qua dự đoán ensemble
- **Trực quan hóa Attention**: Tạo bản đồ attention để giải thích mô hình
- **Demo Streamlit**: Ứng dụng web tương tác để demo mô hình

## Cài Đặt

1. Clone repository:
```bash
git clone <repository-url>
cd User-Behavior-Analysis-Final-Round
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cách Sử Dụng

### Phân Tích Khám Phá Dữ Liệu

Mở `eda.ipynb` trong Jupyter notebook để khám phá tập dữ liệu và insights mô hình.
### Huấn Luyện Mô Hình

Chạy mô hình transformer:
```bash
python transformer_final.py
```

Chạy mô hình GRU:
```bash
python src/gru_weighted_l2_model.py
```

### Chạy Ứng Dụng Demo

```bash
streamlit run app_2.py
```

**Lưu ý về Model**: Vì kích thước mô hình > 100MB, model đã được upload lên Hugging Face Hub. Ứng dụng sẽ tự động tải model từ repository Hugging Face khi khởi chạy lần đầu. Đảm bảo có kết nối internet ổn định để tải model.


## Mô Tả Dữ Liệu

- **Đầu vào**: Các chuỗi hành vi người dùng (ID token phân loại)
- **Đầu ra**: 6 thuộc tính phân loại với số lượng lớp khác nhau:
  - attr_1, attr_4: 12 lớp (1-12) - tháng bắt đầu và hoàn thành giao dịch
  - attr_2, attr_5: 31 lớp (1-31) - ngày bắt đầu và hoàn thành giao dịch
  - attr_3, attr_6: 100 lớp (0-99) - chỉ số hoạt động nhà máy

## Cấu Hình Mô Hình

Các siêu tham số chính được định nghĩa trong `t_max/model_config.json`:
- Kích thước từ vựng: 953
- Độ dài chuỗi tối đa: 66
- Kích thước đặc trưng phụ trợ: 54
