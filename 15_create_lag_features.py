from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("CreateLagFeatures")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

INPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_features_final"

OUTPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_lag_features"


# ============================================================
# 3. ĐỌC DATASET ĐÃ XỬ LÝ
# ============================================================

df = spark.read.parquet(INPUT_PATH)

print("\n================ INPUT ================")
print("Rows:", df.count())
print("Sites:", df.select("site_id").distinct().count())


# ============================================================
# 4. LẤY DANH SÁCH SITE
# ============================================================

sites = (
    df
    .select("site_id")
    .distinct()
)


# ============================================================
# 5. LẤY TOÀN BỘ TIMESTAMP
#
# Dataset gốc có timeline liên tục từ:
# 2016-01-01 00:00
# đến
# 2017-12-31 23:00
#
# Có tổng cộng 17.544 giờ.
# ============================================================

timestamps = (
    df
    .select("timestamp")
    .distinct()
)


print("\n================ TIMELINE ================")
print("Timestamps:", timestamps.count())


# ============================================================
# 6. TẠO FULL SITE × TIMESTAMP GRID
#
# 18 site × 17.544 timestamp
# = 315.792 site-hour
#
# Các site-hour không có dữ liệu điện sẽ có
# total_consumption = NULL.
# ============================================================

full_grid = sites.crossJoin(timestamps)

expected_rows = 18 * 17544

print("\n================ FULL GRID ================")
print("Expected rows:", expected_rows)
print("Actual rows  :", full_grid.count())


# ============================================================
# 7. LEFT JOIN DỮ LIỆU ĐIỆN NĂNG
#
# Giữ nguyên tất cả site-hour trong full grid.
# Những timestamp không có dữ liệu điện sẽ có
# total_consumption = NULL.
# ============================================================

df_grid = (
    full_grid
    .join(
        df,
        on=["site_id", "timestamp"],
        how="left"
    )
)


# ============================================================
# 8. ĐÁNH DẤU TARGET CÓ SẴN HAY KHÔNG
#
# 1 = có dữ liệu điện năng
# 0 = site-hour bị thiếu dữ liệu điện
# ============================================================

df_grid = (
    df_grid
    .withColumn(
        "target_available",
        F.when(
            F.col("total_consumption").isNotNull(),
            1
        ).otherwise(0)
    )
)


# ============================================================
# 9. TẠO LẠI CÁC ĐẶC TRƯNG THỜI GIAN
#
# Các timestamp mới được tạo từ full grid nên phải
# tính lại time features cho cả những dòng trước đây bị thiếu.
# ============================================================

df_grid = (
    df_grid
    .withColumn("year", F.year("timestamp"))
    .withColumn("month", F.month("timestamp"))
    .withColumn("day", F.dayofmonth("timestamp"))
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)
    )
    .withColumn("quarter", F.quarter("timestamp"))
)


# ============================================================
# 10. TẠO WINDOW THEO SITE
#
# Vì full_grid đã có đủ từng giờ liên tục,
# nên:
#
# lag(1)   = đúng 1 giờ trước
# lag(24)  = đúng 24 giờ trước
# lag(168) = đúng 7 ngày trước
# ============================================================

window_spec = (
    Window
    .partitionBy("site_id")
    .orderBy("timestamp")
)


# ============================================================
# 11. TẠO CÁC LAG FEATURES
# ============================================================

df_lag = (
    df_grid
    .withColumn(
        "lag_1h",
        F.lag("total_consumption", 1).over(window_spec)
    )
    .withColumn(
        "lag_24h",
        F.lag("total_consumption", 24).over(window_spec)
    )
    .withColumn(
        "lag_168h",
        F.lag("total_consumption", 168).over(window_spec)
    )
)


# ============================================================
# 12. XÁC ĐỊNH DÒNG ĐỦ ĐIỀU KIỆN CHO MODEL
#
# target hiện tại phải có dữ liệu.
#
# Đồng thời cần có:
# - điện năng 1 giờ trước
# - điện năng 24 giờ trước
# - điện năng 168 giờ trước
#
# Nếu một trong các giá trị bị thiếu thì chưa sử dụng
# dòng đó cho model.
# ============================================================

df_lag = (
    df_lag
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
        ).cast("int")
    )
)


# ============================================================
# 13. THỐNG KÊ KẾT QUẢ
# ============================================================

total_grid = df_lag.count()

missing_target = (
    df_lag
    .filter(F.col("target_available") == 0)
    .count()
)

model_ready = (
    df_lag
    .filter(F.col("model_ready") == 1)
    .count()
)

not_ready = total_grid - model_ready


print("\n================ LAG RESULT ================")
print("Full grid rows      :", total_grid)
print("Missing target rows:", missing_target)
print("Model-ready rows    :", model_ready)
print("Not model-ready     :", not_ready)


# ============================================================
# 14. KIỂM TRA SAMPLE
# ============================================================

print("\n================ SAMPLE LAG ================")

(
    df_lag
    .filter(F.col("model_ready") == 1)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "active_buildings",
        "is_outlier",
        "model_ready"
    )
    .orderBy("site_id", "timestamp")
    .show(20, truncate=False)
)


# ============================================================
# 15. LƯU DATASET
# ============================================================

(
    df_lag
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


# ============================================================
# 16. KIỂM TRA SAU KHI GHI
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

if check_rows == total_grid and check_ready == model_ready:
    print("PASS: Lag dataset saved correctly.")
else:
    print("FAIL: Lag dataset verification failed.")


# ============================================================
# 17. DỪNG SPARK
# ============================================================

spark.stop()