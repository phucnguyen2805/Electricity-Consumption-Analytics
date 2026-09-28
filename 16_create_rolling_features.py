from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("CreateRollingFeatures")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

INPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_lag_features"

OUTPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_ml_features"


# ============================================================
# 3. ĐỌC DATASET
# ============================================================

df = spark.read.parquet(INPUT_PATH)

print("\n================ INPUT ================")
print("Rows:", df.count())
print("Sites:", df.select("site_id").distinct().count())


# ============================================================
# 4. WINDOW CHO ROLLING
#
# Vì full grid có đúng 1 dòng cho mỗi site mỗi giờ:
#
# rowsBetween(-24, -1)
#     = đúng 24 giờ trước
#
# rowsBetween(-168, -1)
#     = đúng 168 giờ trước = 7 ngày
#
# Quan trọng:
# - Không bao gồm dòng hiện tại.
# - Vì vậy không lấy target hiện tại vào feature.
# ============================================================

window_24h = (
    Window
    .partitionBy("site_id")
    .orderBy("timestamp")
    .rowsBetween(-24, -1)
)

window_168h = (
    Window
    .partitionBy("site_id")
    .orderBy("timestamp")
    .rowsBetween(-168, -1)
)


# ============================================================
# 5. TẠO ROLLING MEAN + SỐ LƯỢNG QUAN SÁT
#
# count() giúp xác nhận rolling thực sự có đủ dữ liệu.
# Spark không tính NULL vào count().
# ============================================================

df_ml = (
    df
    .withColumn(
        "rolling_mean_24h",
        F.avg("total_consumption").over(window_24h)
    )
    .withColumn(
        "rolling_count_24h",
        F.count("total_consumption").over(window_24h)
    )
    .withColumn(
        "rolling_mean_168h",
        F.avg("total_consumption").over(window_168h)
    )
    .withColumn(
        "rolling_count_168h",
        F.count("total_consumption").over(window_168h)
    )
)


# ============================================================
# 6. XÁC ĐỊNH MODEL READY
#
# Chỉ sử dụng dòng khi:
#
# - target hiện tại có dữ liệu
# - lag 1h có dữ liệu
# - lag 24h có dữ liệu
# - lag 168h có dữ liệu
# - đủ 24 quan sát cho rolling 24h
# - đủ 168 quan sát cho rolling 7 ngày
# ============================================================

df_ml = (
    df_ml
    .withColumn(
        "model_ready",
        (
            (F.col("target_available") == 1)
            &
            F.col("lag_1h").isNotNull()
            &
            F.col("lag_24h").isNotNull()
            &
            F.col("lag_168h").isNotNull()
            &
            (F.col("rolling_count_24h") == 24)
            &
            (F.col("rolling_count_168h") == 168)
        ).cast("int")
    )
)


# ============================================================
# 7. THỐNG KÊ
# ============================================================

total_rows = df_ml.count()

model_ready = (
    df_ml
    .filter(F.col("model_ready") == 1)
    .count()
)

not_ready = total_rows - model_ready

print("\n================ ROLLING RESULT ================")
print("Total rows     :", total_rows)
print("Model-ready    :", model_ready)
print("Not model-ready:", not_ready)


# ============================================================
# 8. SAMPLE
# ============================================================

print("\n================ SAMPLE ================")

(
    df_ml
    .filter(F.col("model_ready") == 1)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "rolling_count_24h",
        "rolling_mean_168h",
        "rolling_count_168h",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_outlier",
        "model_ready"
    )
    .orderBy("site_id", "timestamp")
    .show(20, truncate=False)
)


# ============================================================
# 9. KIỂM TRA ROLLING
#
# Không dùng phép so sánh:
#
# rolling_mean_24h == total_consumption
#
# để kết luận leakage, vì hai giá trị có thể tình cờ bằng nhau.
#
# Thay vào đó kiểm tra:
# - rolling 24h có đúng 24 quan sát không
# - rolling 168h có đúng 168 quan sát không
# ============================================================

incomplete_24h = (
    df_ml
    .filter(
        (F.col("target_available") == 1)
        &
        (
            F.col("rolling_count_24h") < 24
        )
    )
    .count()
)

incomplete_168h = (
    df_ml
    .filter(
        (F.col("target_available") == 1)
        &
        (
            F.col("rolling_count_168h") < 168
        )
    )
    .count()
)

print("\n================ ROLLING COMPLETENESS ================")
print("Rows with incomplete 24h window :", incomplete_24h)
print("Rows with incomplete 168h window:", incomplete_168h)

print("\nRolling windows exclude current target because:")
print("24h window  = rowsBetween(-24, -1)")
print("168h window = rowsBetween(-168, -1)")
print("PASS: Current target is excluded from rolling features.")


# ============================================================
# 10. LƯU
# ============================================================

(
    df_ml
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


# ============================================================
# 11. CHECKPOINT SAU KHI GHI
# ============================================================

check_df = spark.read.parquet(OUTPUT_PATH)

check_rows = check_df.count()

check_ready = (
    check_df
    .filter(F.col("model_ready") == 1)
    .count()
)

print("\n================ FINAL CHECK ================")
print("Rows after save :", check_rows)
print("Model-ready rows:", check_ready)

if check_rows == total_rows and check_ready == model_ready:
    print("PASS: ML feature dataset saved correctly.")
else:
    print("FAIL: ML feature dataset verification failed.")


# ============================================================
# 12. DỪNG SPARK
# ============================================================

spark.stop()