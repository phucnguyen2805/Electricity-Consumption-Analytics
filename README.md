# Electricity Consumption Analytics

Đồ án môn **Nhập môn Big Data**.

Project thực hiện phân tích dữ liệu tiêu thụ điện năng theo thời gian, theo khu vực/site và xây dựng mô hình dự báo nhu cầu điện bằng **Apache Spark / Spark ML**. Kết quả dự báo được lưu vào **InfluxDB** để phục vụ truy vấn và trực quan hóa trên **Streamlit Dashboard**.

---

## 1. Mục tiêu

- Phân tích dữ liệu tiêu thụ điện năng theo thời gian.
- Làm sạch và chuẩn hóa dữ liệu điện năng, metadata và weather.
- Tạo các đặc trưng phục vụ bài toán dự báo.
- Xây dựng và so sánh nhiều mô hình hồi quy.
- Đánh giá mô hình bằng **RMSE, MAE và R²**.
- Lưu kết quả dự báo vào **InfluxDB**.
- Xây dựng dashboard để theo dõi tiêu thụ thực tế và dự báo.

---

## 2. Công nghệ sử dụng

- **Python**
- **Apache Spark / PySpark 4.2.0**
- **Spark ML**
- **InfluxDB 3**
- **Streamlit**
- **Plotly**
- **Parquet**
- **PowerShell** cho script chạy demo

---

## 3. Dataset

Project sử dụng **Building Data Genome Project 2 (BDG2)**.

Dataset gồm 3 nhóm dữ liệu chính:

- **Electricity**: dữ liệu tiêu thụ điện theo giờ.
- **Metadata**: thông tin về building và site.
- **Weather**: dữ liệu thời tiết theo site và timestamp.

Dữ liệu điện ban đầu ở dạng wide và được chuyển sang dạng long để thuận tiện cho việc xử lý bằng Spark.

### Dữ liệu chính sau preprocessing

| Thành phần | Số lượng |
|---|---:|
| Building ban đầu | 1,578 |
| Timestamp | 17,544 |
| Electricity long records | 27,684,432 |
| Building giữ lại sau xử lý | 1,514 |
| Electricity records sau cleaning | 25,949,330 |
| Site sau cleaning | 18 |
| Site-hour records | 313,370 |
| Dòng dữ liệu model-ready | 304,670 |
| Test records năm 2017 | 153,016 |

---

## 4. Quy trình xử lý dữ liệu

```text
BDG2 Raw Data
      |
      v
Wide -> Long
      |
      v
Data Cleaning
  - Missing values
  - Duplicate check
  - Data validation
      |
      v
Join Metadata + Weather
      |
      v
Feature Engineering
  - Lag features
  - Rolling features
  - Time features
  - Weather features
      |
      v
Train / Validation / Test
      |
      v
Spark ML Regression
      |
      v
Prediction + Evaluation
      |
      v
InfluxDB
      |
      v
Streamlit + Plotly Dashboard
```

---

## 5. Tiền xử lý dữ liệu

### Electricity

Dữ liệu điện được chuyển từ dạng **wide** sang **long** bằng Spark.

Các bước chính:

- Kiểm tra missing values.
- Kiểm tra timestamp.
- Kiểm tra duplicate building và timestamp.
- Loại các building có tỷ lệ thiếu dữ liệu lớn hơn **50%**.
- Không thay missing consumption bằng `0`.
- Các giá trị consumption bị thiếu được giữ dưới dạng missing và loại khỏi tập dùng để huấn luyện.

### Metadata

Electricity được join với metadata theo `building_id` để bổ sung các thuộc tính của building và site.

### Weather

Giữ lại các trường thời tiết có tỷ lệ thiếu phù hợp cho mô hình:

- `airTemperature`
- `dewTemperature`
- `seaLvlPressure`
- `windDirection`
- `windSpeed`

Các trường weather có tỷ lệ thiếu rất cao được loại khỏi bộ đặc trưng. Dữ liệu thiếu của các trường weather được giữ lại được điền theo median của site, có fallback về median toàn cục.

### Outlier

Outlier được phát hiện bằng phương pháp **IQR**. Các outlier không bị xóa khỏi dữ liệu; thay vào đó project tạo cột `is_outlier` để đánh dấu và phục vụ phân tích sai số của mô hình.

---

## 6. Feature Engineering

Các đặc trưng chính được sử dụng:

### Lag features

- `lag_1h`
- `lag_24h`
- `lag_168h`

### Rolling features

- `rolling_mean_24h`
- `rolling_mean_168h`
- `rolling_count_24h`
- `rolling_count_168h`

Các rolling window được tính từ các thời điểm trước đó và không sử dụng giá trị của chính timestamp đang cần dự báo.

### Time features

- `hour`
- `day_of_week`
- `is_weekend`
- `month`
- `quarter`
- `hour_sin`, `hour_cos`
- `day_of_week_sin`, `day_of_week_cos`

### Weather features

- `airTemperature`
- `dewTemperature`
- `seaLvlPressure`
- `windSpeed`
- `windDirection`
- `wind_dir_sin`, `wind_dir_cos`

---

## 7. Chia dữ liệu train / test

Project sử dụng cách chia theo thời gian để tránh đưa dữ liệu tương lai vào quá trình huấn luyện.

- **Train:** năm 2016
- **Test:** năm 2017

Kết quả:

- Train: **151,654** records
- Test: **153,016** records
- Tập model-ready sau feature engineering: **304,670** records

---

## 8. Mô hình

### 8.1 Naive Baseline

Dùng giá trị tiêu thụ của **24 giờ trước** làm dự báo.

### 8.2 Linear Regression

Mô hình hồi quy tuyến tính được sử dụng làm baseline bằng Spark ML.

### 8.3 Random Forest Regression

Mô hình hồi quy phi tuyến được sử dụng để so sánh với các mô hình tuyến tính.

### 8.4 Ridge Regression

Ridge Regression được tuning trên tập validation và sử dụng:

```text
regParam = 10
```

Đây là cấu hình được sử dụng cho mô hình cuối trong quy trình hiện tại.

---

## 9. Kết quả mô hình

Kết quả đánh giá trên tập test năm 2017:

| Mô hình | RMSE | MAE | R² |
|---|---:|---:|---:|
| Naive | 2171.919 | 768.596 | 0.958002 |
| Linear Regression | 1296.932 | 377.430 | 0.985025 |
| Random Forest | 1357.076 | 537.138 | 0.983604 |
| Ridge Regression | **1295.987** | 383.172 | **0.985047** |

### Mô hình cuối

```text
Ridge Regression
regParam = 10

RMSE = 1295.99
MAE  = 383.17
R²   = 0.98505
```

Ngoài các chỉ số tổng thể, project còn phân tích sai số theo site và theo trạng thái outlier.

---

## 10. InfluxDB

Kết quả dự báo cuối được lưu vào **InfluxDB 3** trong database:

```text
electricity
```

Measurement sử dụng:

```text
electricity_forecast
```

Dữ liệu được lưu theo timestamp và site, bao gồm các thông tin chính như:

- Actual consumption
- Prediction
- Error
- Absolute error
- Absolute percentage error
- `is_outlier`

Tổng số prediction được lưu vào InfluxDB:

```text
153,016 points
```

---

## 11. Dashboard

Dashboard được xây dựng bằng **Streamlit** và truy vấn dữ liệu trực tiếp từ InfluxDB.

Các nội dung chính:

- **Actual vs Predicted**
- **Consumption by Hour**
- **Daily Trend / Consumption by Day**
- **Consumption by Site**
- **Model Performance**
- **Outlier Analysis**
- **Top 20 Prediction Errors**
- **Daily Summary Table**

Dashboard có bộ lọc theo:

- Site
- Khoảng ngày

---

## 12. Cấu trúc project

```text
D:\BigData_Electricity
│
├── .gitignore
├── README.md
├── requirements.txt
├── run_demo.ps1
│
├── bdg2/                         # Dataset BDG2 (không commit lên Git)
├── data/
│   ├── raw/                      # Dữ liệu đầu vào
│   └── processed/                # Dữ liệu Parquet đã xử lý
│
├── influxdb/                     # InfluxDB 3 binary
├── influxdb-data/                # Data directory của InfluxDB
├── influxdb-token.txt            # Token InfluxDB (không commit)
├── hadoop/                       # winutils/hadoop native binary cho Windows
│
├── 02_wide_to_long.py
├── 07_clean_electricity.py
├── 13_clean_weather.py
├── 14_join_all.py
├── 15_feature_engineering.py
├── 18_analyze_outliers.py
│
├── 15_create_lag_features.py
├── 15_mark_outliers.py
├── 15b_validate_outlier_method.py
├── 16_create_rolling_features.py
├── 17_join_weather_ml.py
├── 18_split_train_test.py
├── 19_train_linear_baseline.py
├── 20_evaluate_baseline.py
├── 21_train_random_forest.py
├── 22_fix_rf_predictions.py
├── 23_evaluate_random_forest.py
├── 24_tune_ridge_regression.py
├── 25_evaluate_ridge.py
├── 26_prepare_final_outputs.py
├── 27_export_influx_parquet.py
├── 28_write_influx.py
├── 29_write_influx_sync.py
├── 30_prepare_influx_utc.py
├── 31_write_influx_utc.py
├── 32_project_final_check.py
│
├── dashboard.py                 # Streamlit Dashboard
│
└── src/                         # Source code / module của project
```

> Một số thư mục dữ liệu và file cấu hình local được liệt kê ở trên nhưng **không được commit lên Git** theo `.gitignore`.

---

## 13. Git và `.gitignore`

Project sử dụng Git để quản lý mã nguồn.

`.gitignore` được cấu hình để loại trừ các file/thư mục không nên đưa lên repository, gồm:

```text
.venv/
.venv_test/
__pycache__/
*.py[cod]

bdg2/
data/processed/
influxdb-data/
influxdb-token.txt

*.log
*.tmp
hs_err_pid*.log

.vscode/
.idea/
```

Đặc biệt, **không commit `influxdb-token.txt`** vì đây là thông tin xác thực của InfluxDB.

---

## 14. Yêu cầu môi trường

Project hiện được kiểm thử trên **Windows** với:

- Python
- Java JDK
- PySpark 4.2.0
- InfluxDB 3
- PowerShell

Môi trường Python sử dụng virtual environment `.venv`.

---

## 15. Cài đặt

### 15.1 Clone project

```powershell
git clone <repository-url>
cd BigData_Electricity
```

### 15.2 Tạo virtual environment

```powershell
python -m venv .venv
```

### 15.3 Kích hoạt virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 15.4 Cài thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 16. Cấu hình InfluxDB

Project sử dụng InfluxDB 3 chạy local.

Cấu trúc thư mục dự kiến:

```text
influxdb/
└── influxdb3.exe
```

Data directory:

```text
influxdb-data/
```

Token được lưu riêng trong:

```text
influxdb-token.txt
```

Không đưa token lên Git hoặc chia sẻ công khai.

---

## 17. Chạy project

### Cách 1: Chạy toàn bộ demo

Mở PowerShell tại thư mục project:

```powershell
cd D:\BigData_Electricity
```

Kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

Chạy script demo:

```powershell
.\run_demo.ps1
```

Script sẽ kiểm tra môi trường, kiểm tra InfluxDB và khởi động Streamlit Dashboard.

### Dashboard

Sau khi chạy thành công, mở:

```text
http://localhost:8501
```

---

## 18. Chạy kiểm tra project

Project có script kiểm tra tổng thể các kết quả đầu ra:

```powershell
python 32_project_final_check.py
```

Script kiểm tra các thành phần chính như:

- Clean electricity
- Site hourly features
- Train/Test dataset
- Final predictions
- Dashboard datasets
- Số lượng dữ liệu dự kiến ghi vào InfluxDB

Kết quả kiểm tra cuối của project:

```text
ALL LOCAL PROJECT CHECKS PASSED
```

---

## 19. Kiểm tra dependency

Để kiểm tra môi trường Python và các thư viện:

```powershell
python -c "import pyspark, pandas, pyarrow, streamlit, plotly, influxdb_client_3; print('Python environment: OK'); print('PySpark:', pyspark.__version__)"
```

---

## 20. Kết quả đầu ra chính

Các output quan trọng của project gồm:

```text
Electricity cleaned data
Metadata + electricity
Weather cleaned data
Site hourly features
ML lag / rolling / weather dataset
Train / Test dataset
Linear predictions
Random Forest predictions
Ridge predictions
Final predictions
InfluxDB forecast data
Dashboard datasets
```

Dữ liệu xử lý dạng Parquet được lưu trong `data/processed/` và được loại khỏi Git bởi `.gitignore`.

---

## 21. Trạng thái project

Project hiện đã hoàn thành pipeline chính:

```text
Raw Data
   -> Cleaning
   -> Feature Engineering
   -> Train/Test
   -> Regression
   -> Evaluation
   -> Final Prediction
   -> InfluxDB
   -> Dashboard
```

Các bước kiểm tra dữ liệu, huấn luyện mô hình, tạo prediction, nạp InfluxDB và chạy dashboard đã được kiểm tra trong môi trường local.

---

## 22. Tác giả

**Đồ án môn Nhập môn Big Data**

Chủ đề: **Phân tích dữ liệu tiêu thụ điện năng**

---

## 23. Lưu ý

- Không commit `.venv/` lên Git.
- Không commit `bdg2/` và `data/processed/` nếu repository chỉ quản lý source code.
- Không commit `influxdb-data/`.
- Không commit `influxdb-token.txt`.
- Khi clone project trên máy khác, cần chuẩn bị lại dataset, InfluxDB và môi trường Python local.
