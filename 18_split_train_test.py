from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("SplitTrainTest")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

INPUT_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\electricity_ml_dataset"
)

TRAIN_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_train_2016"
)

TEST_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_test_2017"
)


# ============================================================
# 3. ĐỌC DATASET
# ============================================================

df = spark.read.parquet(INPUT_PATH)

print("\n================ INPUT ================")
print("Total rows:", df.count())
print("Total sites:", df.select("site_id").distinct().count())


# ============================================================
# 4. CHỈ GIỮ CÁC DÒNG MODEL-READY
#
# Các dòng không đủ lag/rolling sẽ không được đưa vào model.
# ============================================================

df_ready = (
    df
    .filter(F.col("model_ready") == 1)
)


print("\n================ MODEL READY ================")
print("Model-ready rows:", df_ready.count())


# ============================================================
# 5. CHIA THEO NĂM
#
# Train:
#   toàn bộ dữ liệu hợp lệ trong năm 2016
#
# Test:
#   toàn bộ dữ liệu hợp lệ trong năm 2017
#
# Không random.
# ============================================================

train_df = (
    df_ready
    .filter(F.col("year") == 2016)
)

test_df = (
    df_ready
    .filter(F.col("year") == 2017)
)


# ============================================================
# 6. THỐNG KÊ TRAIN / TEST
# ============================================================

train_count = train_df.count()
test_count = test_df.count()

print("\n================ TRAIN / TEST ================")
print("Train rows:", train_count)
print("Test rows :", test_count)
print("Total     :", train_count + test_count)


# ============================================================
# 7. KIỂM TRA THỜI GIAN
# ============================================================

print("\n================ TRAIN TIMESTAMP ================")

train_df.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).show(truncate=False)


print("\n================ TEST TIMESTAMP ================")

test_df.select(
    F.min("timestamp").alias("min_timestamp"),
    F.max("timestamp").alias("max_timestamp")
).show(truncate=False)


# ============================================================
# 8. KIỂM TRA SITE
#
# Xác nhận train/test vẫn chứa đúng các site cần sử dụng.
# ============================================================

print("\n================ TRAIN SITE COUNT ================")
print(
    "Train sites:",
    train_df.select("site_id").distinct().count()
)

print("\n================ TEST SITE COUNT ================")
print(
    "Test sites :",
    test_df.select("site_id").distinct().count()
)


# ============================================================
# 9. KIỂM TRA NULL TRONG CÁC FEATURE SẼ ĐƯA VÀO MODEL
#
# Chưa bao gồm:
#   is_outlier
#
# vì biến này chỉ dùng cho phân tích chất lượng dữ liệu,
# không dùng làm feature của mô hình.
# ============================================================

feature_columns = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "active_buildings",
    "hour",
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "weather_air_temperature",
    "weather_dew_temperature",
    "weather_sea_level_pressure",
    "weather_wind_speed",
    "wind_dir_sin",
    "wind_dir_cos"
]


print("\n================ FEATURE NULL CHECK ================")

for column in feature_columns:

    train_null = (
        train_df
        .filter(F.col(column).isNull())
        .count()
    )

    test_null = (
        test_df
        .filter(F.col(column).isNull())
        .count()
    )

    print(
        f"{column:30s} "
        f"train_null={train_null:5d} "
        f"test_null={test_null:5d}"
    )


# ============================================================
# 10. KIỂM TRA TARGET NULL
# ============================================================

train_target_null = (
    train_df
    .filter(F.col("total_consumption").isNull())
    .count()
)

test_target_null = (
    test_df
    .filter(F.col("total_consumption").isNull())
    .count()
)

print("\n================ TARGET CHECK ================")
print("Train target null:", train_target_null)
print("Test target null :", test_target_null)


# ============================================================
# 11. LƯU TRAIN / TEST
# ============================================================

(
    train_df
    .write
    .mode("overwrite")
    .parquet(TRAIN_PATH)
)

(
    test_df
    .write
    .mode("overwrite")
    .parquet(TEST_PATH)
)


# ============================================================
# 12. KIỂM TRA SAU KHI GHI
# ============================================================

train_check = spark.read.parquet(TRAIN_PATH)
test_check = spark.read.parquet(TEST_PATH)

train_check_count = train_check.count()
test_check_count = test_check.count()

print("\n================ FINAL CHECK ================")
print("Train rows after save:", train_check_count)
print("Test rows after save :", test_check_count)

if (
    train_check_count == train_count
    and test_check_count == test_count
):
    print("PASS: Train/Test split saved correctly.")
else:
    print("FAIL: Train/Test verification failed.")


# ============================================================
# 13. DỪNG SPARK
# ============================================================

spark.stop()