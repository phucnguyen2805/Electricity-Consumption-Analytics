from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. KHỞI TẠO SPARK
#
# Chưa thay đổi Java/Spark.
# Chỉ giảm độ nặng của Random Forest.
# ============================================================

spark = (
    SparkSession.builder
    .appName("RandomForestRegression")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
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
    r"\models\random_forest_regression"
)

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_random_forest"
)


# ============================================================
# 3. ĐỌC DATA
# ============================================================

train_df = spark.read.parquet(TRAIN_PATH)
test_df = spark.read.parquet(TEST_PATH)


print("\n================ INPUT ================")

print("Train rows :", train_df.count())
print("Test rows  :", test_df.count())
print("Train sites:", train_df.select("site_id").distinct().count())
print("Test sites :", test_df.select("site_id").distinct().count())


# ============================================================
# 4. CHỌN FEATURE
#
# Chỉ sử dụng feature đã xác định là hợp lệ cho forecasting:
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
#   hour_sin
#   hour_cos
#   day_of_week_sin
#   day_of_week_cos
#   is_weekend
#
# Site:
#   site_id
#
# Không dùng:
#   total_consumption hiện tại
#   is_outlier
#   active_buildings
#   weather tại timestamp hiện tại
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
    "is_weekend"
]


# ============================================================
# 5. CHỈ GIỮ CÁC CỘT CẦN THIẾT
#
# Giảm lượng dữ liệu Spark phải mang trong quá trình train.
# ============================================================

required_columns = (
    ["site_id", "total_consumption"]
    + numeric_features
)

train_df = train_df.select(*required_columns)
test_df = test_df.select(*required_columns)


# ============================================================
# 6. MÃ HÓA SITE
#
# Không dùng OneHotEncoder.
#
# Random Forest có thể sử dụng metadata categorical
# được StringIndexer tạo ra.
# ============================================================

site_indexer = StringIndexer(
    inputCol="site_id",
    outputCol="site_index",
    handleInvalid="keep"
)


# ============================================================
# 7. GỘP FEATURE
# ============================================================

assembler = VectorAssembler(
    inputCols=numeric_features + ["site_index"],
    outputCol="features",
    handleInvalid="error"
)


# ============================================================
# 8. RANDOM FOREST - CẤU HÌNH NHẸ
#
# numTrees = 30
#   đủ để làm model thứ hai nhưng nhẹ hơn 100 cây.
#
# maxDepth = 8
#   giới hạn độ phức tạp của mỗi cây.
#
# maxBins = 24
#   giảm số lượng bins cần xử lý.
#
# seed = 42
#   giúp tái lập kết quả.
# ============================================================

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_consumption",
    predictionCol="prediction",
    numTrees=30,
    maxDepth=8,
    maxBins=24,
    seed=42
)


# ============================================================
# 9. PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=[
        site_indexer,
        assembler,
        rf
    ]
)


# ============================================================
# 10. TRAIN
#
# Không cache train/test để tránh giữ quá nhiều dữ liệu
# trong JVM heap.
# ============================================================

print("\n================ TRAINING RANDOM FOREST ================")

model = pipeline.fit(train_df)

print("Random Forest training completed.")


# ============================================================
# 11. PREDICT
# ============================================================

predictions = model.transform(test_df)


# ============================================================
# 12. EVALUATORS
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
# 13. ĐÁNH GIÁ RANDOM FOREST
# ============================================================

rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)


print("\n================ RANDOM FOREST RESULT ================")

print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")


# ============================================================
# 14. SO SÁNH VỚI LINEAR REGRESSION
# ============================================================

linear_rmse = 1296.931754485764
linear_mae = 377.42993326687156
linear_r2 = 0.985025


print("\n================ LINEAR VS RANDOM FOREST ================")

print(
    f"Linear Regression RMSE : {linear_rmse:.6f}"
)

print(
    f"Random Forest RMSE     : {rmse:.6f}"
)

print(
    f"Linear Regression MAE  : {linear_mae:.6f}"
)

print(
    f"Random Forest MAE      : {mae:.6f}"
)

print(
    f"Linear Regression R2   : {linear_r2:.6f}"
)

print(
    f"Random Forest R2       : {r2:.6f}"
)


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

print("\n================ SAMPLE PREDICTIONS ================")

(
    predictions
    .select(
        "site_id",
        F.col("timestamp")
        if "timestamp" in predictions.columns
        else F.lit(None).alias("timestamp"),
        F.col("total_consumption").alias("actual"),
        "prediction"
    )
    .show(20, truncate=False)
)


# ============================================================
# 16. TẠO DATASET PREDICTION
#
# Vì ở trên ta select chỉ các feature cần thiết,
# timestamp không còn trong train/test.
#
# Do đó ở phiên bản này ta lưu prediction theo thứ tự
# dòng của test nhưng không có timestamp.
#
# Để tránh mất thông tin timestamp, phần dưới sẽ đọc lại
# test gốc và join theo một khóa.
# ============================================================

# ------------------------------------------------------------
# Đọc lại test gốc để giữ timestamp + is_outlier
# ------------------------------------------------------------

original_test = spark.read.parquet(TEST_PATH)

original_test = (
    original_test
    .filter(F.col("model_ready") == 1)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "is_outlier"
    )
)


# ------------------------------------------------------------
# Dùng prediction trên test_df.
# Join theo site_id + timestamp không có ở predictions,
# vì vậy ta cần tạo prediction từ test gốc thay vì test_df
# rút gọn.
# ------------------------------------------------------------

model_predictions = model.transform(
    original_test.join(
        spark.read.parquet(TEST_PATH)
        .select(
            "site_id",
            "timestamp",
            *numeric_features
        ),
        on=["site_id", "timestamp"],
        how="inner"
    )
)


# ============================================================
# 17. TẠO OUTPUT
# ============================================================

predictions_output = (
    model_predictions
    .select(
        "site_id",
        "timestamp",
        F.col("total_consumption").alias("actual"),
        "prediction",
        "is_outlier"
    )
    .withColumn(
        "absolute_error",
        F.abs(
            F.col("actual") -
            F.col("prediction")
        )
    )
)


# ============================================================
# 18. LƯU PREDICTIONS
# ============================================================

(
    predictions_output
    .write
    .mode("overwrite")
    .parquet(PREDICTION_PATH)
)


# ============================================================
# 19. LƯU MODEL
# ============================================================

model.write().overwrite().save(MODEL_PATH)


# ============================================================
# 20. FINAL CHECK
# ============================================================

check_df = spark.read.parquet(PREDICTION_PATH)

prediction_count = check_df.count()

print("\n================ FINAL CHECK ================")

print("Prediction rows :", prediction_count)
print("Model path      :", MODEL_PATH)
print("Prediction path :", PREDICTION_PATH)


if prediction_count > 0:
    print("PASS: Random Forest regression completed.")
else:
    print("FAIL: No prediction rows were generated.")


# ============================================================
# 21. DỪNG SPARK
# ============================================================

spark.stop()