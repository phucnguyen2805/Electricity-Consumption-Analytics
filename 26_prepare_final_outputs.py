from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("PrepareFinalOutputs")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

RIDGE_PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_ridge_regression"
)

FINAL_PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\final_predictions"
)

HOURLY_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\dashboard_hourly"
)

DAILY_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\dashboard_daily"
)

SITE_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\dashboard_site"
)


# ============================================================
# 3. ĐỌC PREDICTION RIDGE
# ============================================================

df = spark.read.parquet(
    RIDGE_PREDICTION_PATH
)


print("\n================ INPUT ================")

print(
    "Prediction rows:",
    df.count()
)

print(
    "Sites:",
    df.select("site_id").distinct().count()
)


# ============================================================
# 4. TẠO CÁC CỘT ERROR
# ============================================================

df_final = (
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
    .withColumn(
        "date",
        F.to_date("timestamp")
    )
    .withColumn(
        "hour_of_day",
        F.hour("timestamp")
    )
)


# ============================================================
# 5. ERROR PERCENTAGE
#
# Không chia cho 0.
# Những observation có actual = 0 sẽ nhận NULL.
# ============================================================

df_final = (
    df_final
    .withColumn(
        "absolute_percentage_error",
        F.when(
            F.col("actual") != 0,
            F.abs(
                F.col("error") /
                F.col("actual")
            ) * 100
        )
    )
)


# ============================================================
# 6. DATASET DỰ BÁO CHI TIẾT
# ============================================================

final_predictions = (
    df_final
    .select(
        "site_id",
        "timestamp",
        "date",
        "hour_of_day",
        "actual",
        "prediction",
        "error",
        "absolute_error",
        "absolute_percentage_error",
        "is_outlier"
    )
)


# ============================================================
# 7. DASHBOARD THEO GIỜ
#
# Ví dụ:
# hour = 8
# → tổng actual trung bình/đại diện cho các 08:00
#    trong toàn bộ test 2017.
#
# Dùng AVG để nhìn xu hướng theo giờ trong ngày.
# ============================================================

hourly_dashboard = (
    df_final
    .groupBy("hour_of_day")
    .agg(
        F.avg("actual").alias(
            "actual_consumption"
        ),
        F.avg("prediction").alias(
            "predicted_consumption"
        ),
        F.avg("absolute_error").alias(
            "MAE"
        ),
        F.sqrt(
            F.avg("squared_error")
        ).alias(
            "RMSE"
        )
    )
    .orderBy("hour_of_day")
)


# ============================================================
# 8. DASHBOARD THEO NGÀY
#
# Tổng điện năng thực tế và dự báo của toàn bộ 18 site
# trong từng ngày.
# ============================================================

daily_dashboard = (
    df_final
    .groupBy("date")
    .agg(
        F.sum("actual").alias(
            "actual_consumption"
        ),
        F.sum("prediction").alias(
            "predicted_consumption"
        ),
        F.avg("absolute_error").alias(
            "MAE"
        ),
        F.sqrt(
            F.avg("squared_error")
        ).alias(
            "RMSE"
        )
    )
    .orderBy("date")
)


# ============================================================
# 9. DASHBOARD THEO SITE
#
# site được xem như khu vực/area trong dashboard.
# ============================================================

site_dashboard = (
    df_final
    .groupBy("site_id")
    .agg(
        F.sum("actual").alias(
            "actual_consumption"
        ),
        F.sum("prediction").alias(
            "predicted_consumption"
        ),
        F.avg("absolute_error").alias(
            "MAE"
        ),
        F.sqrt(
            F.avg("squared_error")
        ).alias(
            "RMSE"
        ),
        F.count("*").alias(
            "records"
        )
    )
    .orderBy(F.desc("actual_consumption"))
)


# ============================================================
# 10. HIỂN THỊ KẾT QUẢ
# ============================================================

print("\n================ HOURLY DASHBOARD ================")

hourly_dashboard.show(
    24,
    truncate=False
)


print("\n================ DAILY DASHBOARD ================")

daily_dashboard.show(
    10,
    truncate=False
)


print("\n================ SITE DASHBOARD ================")

site_dashboard.show(
    18,
    truncate=False
)


# ============================================================
# 11. LƯU FINAL PREDICTIONS
# ============================================================

(
    final_predictions
    .write
    .mode("overwrite")
    .parquet(
        FINAL_PREDICTION_PATH
    )
)


# ============================================================
# 12. LƯU HOURLY
# ============================================================

(
    hourly_dashboard
    .write
    .mode("overwrite")
    .parquet(
        HOURLY_PATH
    )
)


# ============================================================
# 13. LƯU DAILY
# ============================================================

(
    daily_dashboard
    .write
    .mode("overwrite")
    .parquet(
        DAILY_PATH
    )
)


# ============================================================
# 14. LƯU SITE
# ============================================================

(
    site_dashboard
    .write
    .mode("overwrite")
    .parquet(
        SITE_PATH
    )
)


# ============================================================
# 15. FINAL CHECK
# ============================================================

final_check = spark.read.parquet(
    FINAL_PREDICTION_PATH
)

hourly_check = spark.read.parquet(
    HOURLY_PATH
)

daily_check = spark.read.parquet(
    DAILY_PATH
)

site_check = spark.read.parquet(
    SITE_PATH
)


print("\n================ FINAL CHECK ================")

print(
    "Final predictions:",
    final_check.count()
)

print(
    "Hourly rows      :",
    hourly_check.count()
)

print(
    "Daily rows        :",
    daily_check.count()
)

print(
    "Site rows         :",
    site_check.count()
)


timestamp_null = (
    final_check
    .filter(F.col("timestamp").isNull())
    .count()
)

actual_null = (
    final_check
    .filter(F.col("actual").isNull())
    .count()
)

prediction_null = (
    final_check
    .filter(F.col("prediction").isNull())
    .count()
)


print(
    "Timestamp NULL   :",
    timestamp_null
)

print(
    "Actual NULL      :",
    actual_null
)

print(
    "Prediction NULL  :",
    prediction_null
)


if (
    final_check.count() == 153016
    and timestamp_null == 0
    and actual_null == 0
    and prediction_null == 0
    and hourly_check.count() == 24
    and site_check.count() == 18
):
    print(
        "PASS: Final dashboard datasets prepared successfully."
    )
else:
    print(
        "FAIL: Final dashboard dataset verification failed."
    )


# ============================================================
# 16. DỪNG SPARK
# ============================================================

spark.stop()