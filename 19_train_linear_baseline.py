import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("LinearRegressionBaseline")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

TRAIN_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_train_2016"
)

TEST_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_test_2017"
)

MODEL_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\models\linear_regression_baseline"
)

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_linear_baseline"
)


# ============================================================
# 3. ĐẢM BẢO THƯ MỤC MODEL TỒN TẠI
# ============================================================

os.makedirs(
    r"D:\BigData_Electricity\data\processed\models",
    exist_ok=True
)


# ============================================================
# 4. ĐỌC TRAIN / TEST
# ============================================================

train_df = spark.read.parquet(TRAIN_PATH)
test_df = spark.read.parquet(TEST_PATH)


print("\n================ INPUT ================")

print("Train rows:",
      train_df.count())

print("Test rows :",
      test_df.count())

print("Train sites:",
      train_df.select("site_id").distinct().count())

print("Test sites :",
      test_df.select("site_id").distinct().count())


# ============================================================
# 5. TẠO MONTH CYCLIC FEATURES
#
# Tháng 12 và tháng 1 gần nhau về mặt thời gian.
# Nếu dùng month = 12 và month = 1 trực tiếp,
# Linear Regression có thể hiểu sai khoảng cách.
#
# Vì vậy dùng sin/cos.
# ============================================================

def add_month_features(df):

    return (
        df
        .withColumn(
            "month_rad",
            2 * F.pi() * (F.col("month") - 1) / 12
        )
        .withColumn(
            "month_sin",
            F.sin("month_rad")
        )
        .withColumn(
            "month_cos",
            F.cos("month_rad")
        )
        .drop("month_rad")
    )


train_df = add_month_features(train_df)
test_df = add_month_features(test_df)


# ============================================================
# 6. DANH SÁCH FEATURE
#
# Chỉ sử dụng các biến có thể biết tại thời điểm dự báo:
#
# Lịch sử:
#   lag_1h
#   lag_24h
#   lag_168h
#
# Rolling:
#   rolling_mean_24h
#   rolling_mean_168h
#
# Thời gian:
#   hour_sin / hour_cos
#   day_of_week_sin / day_of_week_cos
#   month_sin / month_cos
#   is_weekend
#
# Site:
#   site_id -> OneHotEncoder
#
# KHÔNG dùng:
#   total_consumption hiện tại
#   is_outlier
#   active_buildings
#   weather tại chính timestamp hiện tại
# ============================================================

numeric_features = [
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend"
]


# ============================================================
# 7. MÃ HÓA SITE_ID
#
# StringIndexer:
#   site_id -> số nguyên
#
# OneHotEncoder:
#   số nguyên -> vector categorical
#
# Ví dụ:
#   Bear   -> [1,0,0,...]
#   Bobcat -> [0,1,0,...]
# ============================================================

site_indexer = StringIndexer(
    inputCol="site_id",
    outputCol="site_index",
    handleInvalid="keep"
)

site_encoder = OneHotEncoder(
    inputCol="site_index",
    outputCol="site_ohe",
    handleInvalid="error"
)


# ============================================================
# 8. GỘP FEATURE THÀNH VECTOR
# ============================================================

assembler = VectorAssembler(
    inputCols=numeric_features + ["site_ohe"],
    outputCol="features",
    handleInvalid="error"
)


# ============================================================
# 9. LINEAR REGRESSION
#
# label = total_consumption
#
# regParam = 0:
#   baseline không regularization
#
# elasticNetParam = 0:
#   thuần Ridge/L2 khi regParam > 0,
#   nhưng hiện tại regParam = 0 nên thực chất là
#   Linear Regression thông thường.
# ============================================================

lr = LinearRegression(
    featuresCol="features",
    labelCol="total_consumption",
    predictionCol="prediction",
    regParam=0.0,
    elasticNetParam=0.0,
    maxIter=100
)


# ============================================================
# 10. TẠO PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=[
        site_indexer,
        site_encoder,
        assembler,
        lr
    ]
)


# ============================================================
# 11. CACHE DATA
#
# Dữ liệu nhỏ đủ để local Spark xử lý nhanh hơn khi
# evaluator sử dụng prediction nhiều lần.
# ============================================================

train_df = train_df.cache()
test_df = test_df.cache()

train_df.count()
test_df.count()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n================ TRAINING ================")

model = pipeline.fit(train_df)

print("Linear Regression training completed.")


# ============================================================
# 13. PREDICT TRÊN TEST
# ============================================================

predictions = model.transform(test_df)


# ============================================================
# 14. EVALUATOR
# ============================================================

rmse_evaluator = RegressionEvaluator(
    labelCol="total_consumption",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="total_consumption",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="total_consumption",
    predictionCol="prediction",
    metricName="r2"
)


# ============================================================
# 15. ĐÁNH GIÁ LINEAR REGRESSION
# ============================================================

rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)


print("\n================ LINEAR REGRESSION RESULT ================")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")


# ============================================================
# 16. NAIVE BASELINE
#
# Dùng:
#
# prediction = lag_24h
#
# Nghĩa là:
# "Điện năng giờ này ≈ điện năng đúng 24 giờ trước"
#
# Đây là baseline rất hữu ích cho dữ liệu có tính mùa vụ
# theo ngày.
# ============================================================

naive_predictions = (
    test_df
    .withColumn(
        "prediction",
        F.col("lag_24h")
    )
)


naive_rmse = rmse_evaluator.evaluate(
    naive_predictions
)

naive_mae = mae_evaluator.evaluate(
    naive_predictions
)

naive_r2 = r2_evaluator.evaluate(
    naive_predictions
)


print("\n================ NAIVE LAG-24H RESULT ================")
print(f"RMSE: {naive_rmse:.6f}")
print(f"MAE : {naive_mae:.6f}")
print(f"R2  : {naive_r2:.6f}")


# ============================================================
# 17. SO SÁNH VỚI NAIVE
# ============================================================

print("\n================ MODEL COMPARISON ================")

print(
    f"Linear Regression RMSE : {rmse:.6f}"
)

print(
    f"Naive lag-24h RMSE     : {naive_rmse:.6f}"
)

print(
    f"Linear Regression MAE  : {mae:.6f}"
)

print(
    f"Naive lag-24h MAE      : {naive_mae:.6f}"
)


# ============================================================
# 18. SAMPLE PREDICTION
# ============================================================

print("\n================ SAMPLE PREDICTIONS ================")

(
    predictions
    .select(
        "site_id",
        "timestamp",
        F.col("total_consumption").alias("actual"),
        "prediction",
        "lag_24h",
        "is_outlier"
    )
    .orderBy("site_id", "timestamp")
    .show(20, truncate=False)
)


# ============================================================
# 19. TÍNH SAI SỐ TUYỆT ĐỐI
# ============================================================

predictions_output = (
    predictions
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "prediction",
        "lag_24h",
        "is_outlier"
    )
    .withColumn(
        "absolute_error",
        F.abs(
            F.col("total_consumption") -
            F.col("prediction")
        )
    )
)


# ============================================================
# 20. LƯU PREDICTIONS
# ============================================================

(
    predictions_output
    .write
    .mode("overwrite")
    .parquet(PREDICTION_PATH)
)


# ============================================================
# 21. LƯU MODEL
# ============================================================

model.write().overwrite().save(MODEL_PATH)


# ============================================================
# 22. FINAL CHECK
# ============================================================

prediction_check = spark.read.parquet(
    PREDICTION_PATH
)

prediction_count = prediction_check.count()

print("\n================ FINAL CHECK ================")

print("Test rows          :", test_df.count())
print("Prediction rows    :", prediction_count)
print("Model path         :", MODEL_PATH)
print("Prediction path    :", PREDICTION_PATH)

if prediction_count == test_df.count():
    print("PASS: Linear Regression baseline completed.")
else:
    print("FAIL: Prediction row count does not match test set.")


# ============================================================
# 23. DỪNG SPARK
# ============================================================

train_df.unpersist()
test_df.unpersist()

spark.stop()