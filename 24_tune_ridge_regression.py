from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("TuneRidgeRegression")
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
    r"\models\ridge_regression_final"
)

PREDICTION_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_ridge_regression"
)


# ============================================================
# 3. ĐỌC DATA
# ============================================================

train_full = spark.read.parquet(TRAIN_PATH)
test_df = spark.read.parquet(TEST_PATH)


print("\n================ INPUT ================")
print("Full train rows:", train_full.count())
print("Test rows      :", test_df.count())


# ============================================================
# 4. CHIA TRAIN / VALIDATION THEO THỜI GIAN
#
# Không random split.
#
# Train:
#   trước 01/10/2016
#
# Validation:
#   từ 01/10/2016 đến hết 2016
#
# Test 2017 hoàn toàn giữ riêng.
# ============================================================

split_date = F.to_timestamp(
    F.lit("2016-10-01 00:00:00")
)

train_part = (
    train_full
    .filter(F.col("timestamp") < split_date)
)

validation_part = (
    train_full
    .filter(F.col("timestamp") >= split_date)
)


print("\n================ TIME SPLIT ================")

print(
    "Train part:",
    train_part.count()
)

print(
    "Validation:",
    validation_part.count()
)

print(
    "Test 2017 :",
    test_df.count()
)


# ============================================================
# 5. FEATURE
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
# 6. EVALUATORS
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


# ============================================================
# 7. CÁC REGULARIZATION VALUES
#
# regParam càng lớn:
#   regularization càng mạnh.
#
# Ta không dùng test 2017 để chọn.
# ============================================================

reg_params = [
    0.01,
    0.1,
    1.0,
    10.0
]


results = []


# ============================================================
# 8. THỬ TỪNG REGULARIZATION
# ============================================================

print("\n================ RIDGE VALIDATION ================")

for reg_param in reg_params:

    print(
        f"\nTraining regParam = {reg_param}"
    )

    # --------------------------------------------------------
    # StringIndexer cho site
    # --------------------------------------------------------

    site_indexer = StringIndexer(
        inputCol="site_id",
        outputCol="site_index",
        handleInvalid="keep"
    )

    # --------------------------------------------------------
    # One-Hot Encoding cho site
    # --------------------------------------------------------

    site_encoder = OneHotEncoder(
        inputCol="site_index",
        outputCol="site_ohe",
        handleInvalid="error"
    )

    # --------------------------------------------------------
    # VectorAssembler
    # --------------------------------------------------------

    assembler = VectorAssembler(
        inputCols=numeric_features + ["site_ohe"],
        outputCol="features",
        handleInvalid="error"
    )

    # --------------------------------------------------------
    # Ridge Regression
    #
    # elasticNetParam = 0
    # => pure Ridge / L2 regularization.
    # --------------------------------------------------------

    lr = LinearRegression(
        featuresCol="features",
        labelCol="total_consumption",
        predictionCol="prediction",

        regParam=reg_param,

        elasticNetParam=0.0,

        maxIter=100
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        stages=[
            site_indexer,
            site_encoder,
            assembler,
            lr
        ]
    )

    # --------------------------------------------------------
    # Train trên phần đầu 2016
    # --------------------------------------------------------

    model = pipeline.fit(train_part)

    # --------------------------------------------------------
    # Validate trên cuối 2016
    # --------------------------------------------------------

    validation_prediction = model.transform(
        validation_part
    )

    validation_rmse = (
        rmse_evaluator.evaluate(
            validation_prediction
        )
    )

    validation_mae = (
        mae_evaluator.evaluate(
            validation_prediction
        )
    )

    results.append(
        (
            reg_param,
            validation_rmse,
            validation_mae
        )
    )

    print(
        f"regParam={reg_param:.2f} | "
        f"Validation RMSE={validation_rmse:.6f} | "
        f"Validation MAE={validation_mae:.6f}"
    )


# ============================================================
# 9. IN BẢNG KẾT QUẢ
# ============================================================

print("\n================ ALL VALIDATION RESULTS ================")

for reg_param, rmse, mae in results:

    print(
        f"regParam={reg_param:.2f} | "
        f"RMSE={rmse:.6f} | "
        f"MAE={mae:.6f}"
    )


# ============================================================
# 10. CHỌN REGPARAM TỐT NHẤT
#
# Chọn theo Validation RMSE thấp nhất.
# ============================================================

best_reg_param, best_val_rmse, best_val_mae = min(
    results,
    key=lambda x: x[1]
)


print("\n================ BEST PARAMETER ================")

print(
    f"Best regParam        : {best_reg_param}"
)

print(
    f"Validation RMSE      : {best_val_rmse:.6f}"
)

print(
    f"Validation MAE       : {best_val_mae:.6f}"
)


# ============================================================
# 11. TRAIN LẠI MODEL TỐT NHẤT TRÊN TOÀN BỘ 2016
#
# Sau khi chọn regParam bằng validation,
# ta được phép dùng toàn bộ dữ liệu 2016
# để train model cuối.
# ============================================================

print(
    "\n================ FINAL TRAINING ================"
)

site_indexer_final = StringIndexer(
    inputCol="site_id",
    outputCol="site_index",
    handleInvalid="keep"
)

site_encoder_final = OneHotEncoder(
    inputCol="site_index",
    outputCol="site_ohe",
    handleInvalid="error"
)

assembler_final = VectorAssembler(
    inputCols=numeric_features + ["site_ohe"],
    outputCol="features",
    handleInvalid="error"
)

lr_final = LinearRegression(
    featuresCol="features",
    labelCol="total_consumption",
    predictionCol="prediction",
    regParam=best_reg_param,
    elasticNetParam=0.0,
    maxIter=100
)

final_pipeline = Pipeline(
    stages=[
        site_indexer_final,
        site_encoder_final,
        assembler_final,
        lr_final
    ]
)

final_model = final_pipeline.fit(
    train_full
)

print("Final Ridge model training completed.")


# ============================================================
# 12. DỰ ĐOÁN TEST 2017
# ============================================================

test_predictions = final_model.transform(
    test_df
)


# ============================================================
# 13. ĐÁNH GIÁ TEST
# ============================================================

test_rmse = (
    rmse_evaluator.evaluate(
        test_predictions
    )
)

test_mae = (
    mae_evaluator.evaluate(
        test_predictions
    )
)


r2_evaluator = RegressionEvaluator(
    labelCol="total_consumption",
    predictionCol="prediction",
    metricName="r2"
)

test_r2 = (
    r2_evaluator.evaluate(
        test_predictions
    )
)


print(
    "\n================ FINAL RIDGE TEST RESULT ================"
)

print(
    f"Best regParam: {best_reg_param}"
)

print(
    f"Test RMSE    : {test_rmse:.6f}"
)

print(
    f"Test MAE     : {test_mae:.6f}"
)

print(
    f"Test R2      : {test_r2:.6f}"
)


# ============================================================
# 14. SO SÁNH VỚI LINEAR REGRESSION BASELINE
# ============================================================

baseline_rmse = 1296.931754
baseline_mae = 377.429933
baseline_r2 = 0.985025

print(
    "\n================ BASELINE VS RIDGE ================"
)

print(
    f"Baseline LR RMSE : {baseline_rmse:.6f}"
)

print(
    f"Ridge RMSE       : {test_rmse:.6f}"
)

print(
    f"Baseline LR MAE  : {baseline_mae:.6f}"
)

print(
    f"Ridge MAE        : {test_mae:.6f}"
)

print(
    f"Baseline LR R2   : {baseline_r2:.6f}"
)

print(
    f"Ridge R2         : {test_r2:.6f}"
)


# ============================================================
# 15. SAMPLE
# ============================================================

print(
    "\n================ SAMPLE RIDGE PREDICTIONS ================"
)

(
    test_predictions
    .select(
        "site_id",
        "timestamp",
        F.col("total_consumption").alias("actual"),
        "prediction",
        "is_outlier"
    )
    .orderBy(
        "site_id",
        "timestamp"
    )
    .show(
        20,
        truncate=False
    )
)


# ============================================================
# 16. TẠO OUTPUT
# ============================================================

prediction_output = (
    test_predictions
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
# 17. LƯU PREDICTIONS
# ============================================================

(
    prediction_output
    .write
    .mode("overwrite")
    .parquet(PREDICTION_PATH)
)


# ============================================================
# 18. LƯU MODEL
# ============================================================

final_model.write().overwrite().save(
    MODEL_PATH
)


# ============================================================
# 19. VERIFY
# ============================================================

check_df = spark.read.parquet(
    PREDICTION_PATH
)

prediction_count = check_df.count()

timestamp_null = (
    check_df
    .filter(F.col("timestamp").isNull())
    .count()
)

prediction_null = (
    check_df
    .filter(F.col("prediction").isNull())
    .count()
)


print(
    "\n================ FINAL CHECK ================"
)

print(
    "Prediction rows:",
    prediction_count
)

print(
    "Timestamp NULL :",
    timestamp_null
)

print(
    "Prediction NULL:",
    prediction_null
)

print(
    "Model path     :",
    MODEL_PATH
)

print(
    "Prediction path:",
    PREDICTION_PATH
)


if (
    prediction_count == test_df.count()
    and timestamp_null == 0
    and prediction_null == 0
):
    print(
        "PASS: Ridge Regression completed successfully."
    )
else:
    print(
        "FAIL: Ridge Regression verification failed."
    )


# ============================================================
# 20. DỪNG SPARK
# ============================================================

spark.stop()