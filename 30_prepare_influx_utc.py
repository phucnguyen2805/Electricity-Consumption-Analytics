import pandas as pd


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

INPUT_FILE = (
    r"D:\BigData_Electricity\data\processed"
    r"\influx_export\electricity_forecast.parquet"
)

OUTPUT_FILE = (
    r"D:\BigData_Electricity\data\processed"
    r"\influx_export\electricity_forecast_utc.parquet"
)

EXPECTED_ROWS = 153016


# ============================================================
# 2. ĐỌC FILE NGUỒN
# ============================================================

print("\n================ READ SOURCE ================")

df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {len(df)}"
    )


# ============================================================
# 3. KIỂM TRA TIMESTAMP ĐANG CÓ TRONG PARQUET
#
# Do Spark đã lưu timestamp theo instant UTC,
# khi PyArrow đọc ra sẽ thấy:
#
# 2016-12-31 17:00
# ...
# 2017-12-31 16:00
#
# Trong pipeline Spark, đây tương ứng với:
#
# 2017-01-01 00:00
# ...
# 2017-12-31 23:00
# ============================================================

print("\n================ ORIGINAL PARQUET TIMESTAMP ================")

print("Min:", df["timestamp"].min())
print("Max:", df["timestamp"].max())
print("Dtype:", df["timestamp"].dtype)


# ============================================================
# 4. KHÔI PHỤC GIỜ CỦA DATASET SPARK
#
# Dataset của project đang được xử lý theo UTC+7.
#
# Vì Parquet hiện chứa timestamp thấp hơn 7 giờ,
# cộng lại 7 giờ để khôi phục timestamp ban đầu.
# ============================================================

df["timestamp"] = (
    pd.to_datetime(df["timestamp"])
    + pd.Timedelta(hours=7)
)


# ============================================================
# 5. GÁN UTC
#
# Sau khi cộng 7 giờ:
#
# 2017-01-01 00:00
#       ↓
# 2017-01-01 00:00+00:00
#
# Đây là "giữ nguyên giờ hiển thị" của pipeline Spark.
# ============================================================

df["timestamp"] = (
    df["timestamp"]
    .dt.tz_localize("UTC")
)


# ============================================================
# 6. KIỂM TRA TIMESTAMP SAU KHI SỬA
# ============================================================

print("\n================ CORRECTED UTC TIMESTAMP ================")

print("Min:", df["timestamp"].min())
print("Max:", df["timestamp"].max())
print("Dtype:", df["timestamp"].dtype)


# ============================================================
# 7. EXPECTED RANGE
# ============================================================

expected_min = pd.Timestamp(
    "2017-01-01 00:00:00",
    tz="UTC"
)

expected_max = pd.Timestamp(
    "2017-12-31 23:00:00",
    tz="UTC"
)


# ============================================================
# 8. VALIDATE RANGE
# ============================================================

if df["timestamp"].min() != expected_min:
    raise ValueError(
        "Unexpected minimum timestamp: "
        f"{df['timestamp'].min()}"
    )

if df["timestamp"].max() != expected_max:
    raise ValueError(
        "Unexpected maximum timestamp: "
        f"{df['timestamp'].max()}"
    )


# ============================================================
# 9. KIỂM TRA TIMESTAMP NULL
# ============================================================

timestamp_null = (
    df["timestamp"]
    .isna()
    .sum()
)

print(
    "Timestamp NULL:",
    timestamp_null
)

if timestamp_null != 0:
    raise ValueError(
        "Timestamp contains NULL values."
    )


# ============================================================
# 10. GHI FILE
# ============================================================

print("\n================ WRITE UTC PARQUET ================")

df.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine="pyarrow",
    compression="snappy"
)

print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# 11. ĐỌC LẠI VERIFY
# ============================================================

check = pd.read_parquet(
    OUTPUT_FILE
)

print("\n================ FINAL CHECK ================")

print("Rows :", len(check))
print("Min  :", check["timestamp"].min())
print("Max  :", check["timestamp"].max())
print("Dtype:", check["timestamp"].dtype)


if (
    len(check) == EXPECTED_ROWS
    and check["timestamp"].min() == expected_min
    and check["timestamp"].max() == expected_max
    and check["timestamp"].isna().sum() == 0
):
    print(
        "PASS: UTC Parquet created correctly."
    )
else:
    print(
        "FAIL: UTC Parquet verification failed."
    )