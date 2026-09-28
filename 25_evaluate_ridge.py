from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("EvaluateRidge")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_ridge_regression"
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
# 5. OVERALL
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

q = (
    df_error
    .approxQuantile(
        "absolute_error",
        [0.50, 0.90, 0.95, 0.99],
        0.001
    )
)

print(f"P50 absolute error: {q[0]:.4f}")
print(f"P90 absolute error: {q[1]:.4f}")
print(f"P95 absolute error: {q[2]:.4f}")
print(f"P99 absolute error: {q[3]:.4f}")


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
# 9. TỶ TRỌNG SSE CỦA OUTLIER
# ============================================================

total_sse = (
    df_error
    .agg(F.sum("squared_error").alias("sse"))
    .collect()[0]["sse"]
)

outlier_sse = (
    df_error
    .filter(F.col("is_outlier") == 1)
    .agg(F.sum("squared_error").alias("sse"))
    .collect()[0]["sse"]
)

outlier_sse_share = (
    outlier_sse / total_sse * 100
)

print(
    "\n================ OUTLIER SSE CONTRIBUTION ================"
)

print(
    f"Outlier SSE contribution: "
    f"{outlier_sse_share:.4f}%"
)


# ============================================================
# 10. TOP 30 ERROR
# ============================================================

print(
    "\n================ TOP 30 ABSOLUTE ERROR ================"
)

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
# 11. NULL CHECK
# ============================================================

prediction_null = (
    df_error
    .filter(F.col("prediction").isNull())
    .count()
)

actual_null = (
    df_error
    .filter(F.col("actual").isNull())
    .count()
)

timestamp_null = (
    df_error
    .filter(F.col("timestamp").isNull())
    .count()
)


print("\n================ NULL CHECK ================")

print("Prediction NULL:", prediction_null)
print("Actual NULL    :", actual_null)
print("Timestamp NULL :", timestamp_null)


# ============================================================
# 12. FINAL
# ============================================================

if (
    prediction_null == 0
    and actual_null == 0
    and timestamp_null == 0
):
    print("\nPASS: Ridge evaluation completed.")
else:
    print("\nFAIL: Prediction dataset contains NULL values.")


# ============================================================
# 13. DỪNG SPARK
# ============================================================

spark.stop()