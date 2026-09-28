from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("EvaluateRandomForest")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_random_forest"
)

TEST_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_test_2017"
)


# ============================================================
# 3. ĐỌC PREDICTION
# ============================================================

df = spark.read.parquet(PREDICTION_PATH)

print("\n================ INPUT ================")
print("Prediction rows:",
      df.count())

print("Sites:",
      df.select("site_id").distinct().count())


# ============================================================
# 4. TẠO ERROR
# ============================================================

df_error = (
    df
    .withColumn(
        "error",
        F.col("actual") - F.col("prediction")
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
# 5. OVERALL METRICS
# ============================================================

print("\n================ OVERALL ERROR ================")

df_error.select(
    F.avg("absolute_error").alias("MAE"),
    F.sqrt(F.avg("squared_error")).alias("RMSE"),
    F.max("absolute_error").alias("MaxAE")
).show(truncate=False)


# ============================================================
# 6. ERROR QUANTILES
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
# 8. NORMAL VS OUTLIER
#
# is_outlier chỉ dùng để phân tích.
# Không loại outlier khỏi dữ liệu.
# ============================================================

outlier_metrics = (
    df_error
    .groupBy("is_outlier")
    .agg(
        F.count("*").alias("rows"),
        F.avg("absolute_error").alias("MAE"),
        F.sqrt(F.avg("squared_error")).alias("RMSE"),
        F.max("absolute_error").alias("MaxAE"),
        F.sum("squared_error").alias("SSE")
    )
    .orderBy("is_outlier")
)


print("\n================ NORMAL VS OUTLIER ================")

outlier_metrics.show(
    truncate=False
)


# ============================================================
# 9. TÍNH TỶ TRỌNG SSE CỦA OUTLIER
#
# Cho biết outlier đóng góp bao nhiêu phần vào tổng
# bình phương sai số.
# ============================================================

sse_total = (
    df_error
    .agg(F.sum("squared_error").alias("sse"))
    .collect()[0]["sse"]
)

sse_outlier = (
    df_error
    .filter(F.col("is_outlier") == 1)
    .agg(F.sum("squared_error").alias("sse"))
    .collect()[0]["sse"]
)

outlier_sse_share = (
    sse_outlier / sse_total * 100
)

print("\n================ OUTLIER SSE CONTRIBUTION ================")

print(
    f"Outlier SSE contribution: "
    f"{outlier_sse_share:.4f}%"
)


# ============================================================
# 10. TOP 30 ERROR
# ============================================================

print("\n================ TOP 30 ABSOLUTE ERROR ================")

(
    df_error
    .select(
        "site_id",
        "timestamp",
        "actual",
        "prediction",
        "absolute_error",
        "is_outlier"
    )
    .orderBy(F.desc("absolute_error"))
    .show(30, truncate=False)
)


# ============================================================
# 11. KIỂM TRA PREDICTION NULL
# ============================================================

prediction_null = (
    df
    .filter(F.col("prediction").isNull())
    .count()
)

actual_null = (
    df
    .filter(F.col("actual").isNull())
    .count()
)

timestamp_null = (
    df
    .filter(F.col("timestamp").isNull())
    .count()
)

print("\n================ NULL CHECK ================")

print("Prediction NULL:", prediction_null)
print("Actual NULL    :", actual_null)
print("Timestamp NULL :", timestamp_null)


# ============================================================
# 12. SO SÁNH VỚI LINEAR REGRESSION
# ============================================================

linear_rmse = 1296.931754
linear_mae = 377.429933
linear_r2 = 0.985025

rf_metrics = (
    df_error
    .agg(
        F.avg("absolute_error").alias("mae"),
        F.sqrt(F.avg("squared_error")).alias("rmse")
    )
    .collect()[0]
)

rf_mae = rf_metrics["mae"]
rf_rmse = rf_metrics["rmse"]

print("\n================ LINEAR VS RANDOM FOREST ================")

print(f"Linear Regression RMSE : {linear_rmse:.6f}")
print(f"Random Forest RMSE     : {rf_rmse:.6f}")

print(f"Linear Regression MAE  : {linear_mae:.6f}")
print(f"Random Forest MAE      : {rf_mae:.6f}")

print(f"Linear Regression R2   : {linear_r2:.6f}")


# ============================================================
# 13. FINAL CHECK
# ============================================================

if (
    prediction_null == 0
    and actual_null == 0
    and timestamp_null == 0
):
    print("\nPASS: Random Forest evaluation completed.")
else:
    print("\nFAIL: Prediction dataset contains NULL values.")


# ============================================================
# 14. DỪNG SPARK
# ============================================================

spark.stop()