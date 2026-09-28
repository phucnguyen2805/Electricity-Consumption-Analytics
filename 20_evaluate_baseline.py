from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("EvaluateLinearBaseline")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_linear_baseline"
)


# ============================================================
# 3. ĐỌC KẾT QUẢ DỰ ĐOÁN
# ============================================================

df = spark.read.parquet(PREDICTION_PATH)

print("\n================ INPUT ================")
print("Prediction rows:", df.count())
print("Sites:", df.select("site_id").distinct().count())


# ============================================================
# 4. TẠO CÁC CỘT SAI SỐ
#
# actual:
#     total_consumption
#
# prediction:
#     kết quả Linear Regression
#
# absolute_error:
#     |actual - prediction|
#
# squared_error:
#     (actual - prediction)^2
# ============================================================

df_error = (
    df
    .withColumn(
        "error",
        F.col("total_consumption") - F.col("prediction")
    )
    .withColumn(
        "absolute_error",
        F.abs(F.col("error"))
    )
    .withColumn(
        "squared_error",
        F.pow(F.col("error"), 2)
    )
)


# ============================================================
# 5. PHÂN TÍCH SAI SỐ TOÀN BỘ TEST
# ============================================================

print("\n================ OVERALL ERROR ================")

df_error.select(
    F.avg("absolute_error").alias("MAE"),
    F.sqrt(F.avg("squared_error")).alias("RMSE"),
    F.max("absolute_error").alias("max_absolute_error")
).show(truncate=False)


# ============================================================
# 6. PHÂN VỊ ABSOLUTE ERROR
#
# Giúp biết:
# - 50% prediction có sai số <= mức nào
# - 90%
# - 95%
# - 99%
#
# Đây là thông tin bổ sung rất hữu ích cho báo cáo.
# ============================================================

print("\n================ ERROR QUANTILES ================")

quantiles = (
    df_error
    .approxQuantile(
        "absolute_error",
        [0.50, 0.90, 0.95, 0.99],
        0.001
    )
)

print(f"P50 absolute error: {quantiles[0]:.4f}")
print(f"P90 absolute error: {quantiles[1]:.4f}")
print(f"P95 absolute error: {quantiles[2]:.4f}")
print(f"P99 absolute error: {quantiles[3]:.4f}")


# ============================================================
# 7. METRIC THEO SITE
#
# RMSE và MAE được tính riêng cho từng site.
# ============================================================

site_metrics = (
    df_error
    .groupBy("site_id")
    .agg(
        F.count("*").alias("rows"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        F.max("absolute_error").alias("MaxAE")
    )
    .orderBy(F.desc("RMSE"))
)


print("\n================ METRIC BY SITE ================")

site_metrics.show(
    30,
    truncate=False
)


# ============================================================
# 8. PHÂN TÍCH OUTLIER VS NORMAL
#
# is_outlier = 1:
#     observation được đánh dấu IQR outlier
#
# is_outlier = 0:
#     observation bình thường
#
# Không loại outlier khỏi dataset.
# Chỉ phân tích sai số riêng.
# ============================================================

outlier_metrics = (
    df_error
    .groupBy("is_outlier")
    .agg(
        F.count("*").alias("rows"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        F.max("absolute_error").alias("MaxAE")
    )
    .orderBy("is_outlier")
)


print("\n================ NORMAL VS OUTLIER ================")

outlier_metrics.show(
    truncate=False
)


# ============================================================
# 9. TOP 30 SAI SỐ LỚN NHẤT
# ============================================================

print("\n================ TOP 30 ABSOLUTE ERROR ================")

(
    df_error
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "prediction",
        "lag_24h",
        "absolute_error",
        "is_outlier"
    )
    .orderBy(F.desc("absolute_error"))
    .show(30, truncate=False)
)


# ============================================================
# 10. TOP SITE THEO MAE
# ============================================================

print("\n================ TOP SITE BY MAE ================")

(
    site_metrics
    .orderBy(F.desc("MAE"))
    .show(18, truncate=False)
)


# ============================================================
# 11. DỪNG SPARK
# ============================================================

spark.stop()