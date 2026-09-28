import os

from influxdb_client_3 import (
    InfluxDBClient3,
    WriteOptions,
    write_client_options
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

HOST = "http://127.0.0.1:8181"

DATABASE = "electricity"

MEASUREMENT = "electricity_forecast"

PARQUET_FILE = (
    r"D:\BigData_Electricity\data\processed"
    r"\influx_export\electricity_forecast.parquet"
)

TOKEN = os.getenv("INFLUXDB3_AUTH_TOKEN")


# ============================================================
# 2. KIỂM TRA TOKEN
# ============================================================

if not TOKEN:
    raise RuntimeError(
        "INFLUXDB3_AUTH_TOKEN is missing."
    )


# ============================================================
# 3. KIỂM TRA FILE
# ============================================================

if not os.path.isfile(PARQUET_FILE):
    raise FileNotFoundError(
        f"Parquet file not found: {PARQUET_FILE}"
    )


print("\n================ CONFIG ================")

print("Host       :", HOST)
print("Database   :", DATABASE)
print("Measurement:", MEASUREMENT)
print("Parquet    :", PARQUET_FILE)


# ============================================================
# 4. CẤU HÌNH BATCH WRITE
#
# Server giới hạn mỗi HTTP request khoảng 10 MB.
#
# Mỗi batch chỉ chứa 500 points để tránh tạo request quá lớn.
#
# Nếu một batch thất bại, client sẽ tự retry theo cấu hình.
# ============================================================

write_options = WriteOptions(
    batch_size=500,
    flush_interval=1000,
    retry_interval=5000,
    max_retries=5,
    max_retry_delay=30000,
    exponential_base=2
)


write_client_options_config = write_client_options(
    write_options=write_options
)


# ============================================================
# 5. KẾT NỐI INFLUXDB
# ============================================================

print("\n================ WRITING ================")

with InfluxDBClient3(
    host=HOST,
    database=DATABASE,
    token=TOKEN,
    write_client_options=write_client_options_config
) as client:

    client.write_file(
        file=PARQUET_FILE,

        measurement_name=MEASUREMENT,

        timestamp_column="timestamp",

        tag_columns=[
            "site_id",
            "is_outlier"
        ]
    )

    print(
        "All batches submitted successfully."
    )


# ============================================================
# 6. KẾT THÚC
#
# Khi thoát khỏi with-block, client sẽ flush các batch còn lại.
# ============================================================

print(
    "\nPASS: Data written to InfluxDB using batches."
)