from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("FixRandomForestPredictions")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

TEST_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\ml_test_2017"
)

MODEL_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\models\random_forest_regression"
)

OUTPUT_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\predictions_random_forest"
)


# ============================================================
# 3. ĐỌC TEST DATA GỐC
#
# Không select bỏ timestamp.
# Giữ nguyên:
# - site_id
# - timestamp
# - total_consumption
# - is_outlier
# - toàn bộ feature cần thiết cho model
# ============================================================

test_df = (
    spark.read
    .parquet(TEST_PATH)
    .filter(F.col("model_ready") == 1)
)


print("\n================ INPUT ================")
print("Test rows:", test_df.count())
print("Test sites:", test_df.select("site_id").distinct().count())


# ============================================================
# 4. LOAD MODEL ĐÃ TRAIN
#
# Không train lại Random Forest.
# ============================================================

model = PipelineModel.load(MODEL_PATH)

print("\n================ MODEL ================")
print("Loaded model:", MODEL_PATH)


# ============================================================
# 5. PREDICT
#
# Model transform trực tiếp test data gốc,
# do đó timestamp được giữ nguyên.
# ============================================================

predictions = model.transform(test_df)


# ============================================================
# 6. TẠO OUTPUT
# ============================================================

predictions_output = (
    predictions
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
# 7. KIỂM TRA SAMPLE
# ============================================================

print("\n================ SAMPLE PREDICTIONS ================")

(
    predictions_output
    .orderBy("site_id", "timestamp")
    .show(20, truncate=False)
)


# ============================================================
# 8. KIỂM TRA TIMESTAMP NULL
# ============================================================

timestamp_null = (
    predictions_output
    .filter(F.col("timestamp").isNull())
    .count()
)

prediction_count = predictions_output.count()


print("\n================ TIMESTAMP CHECK ================")
print("Prediction rows :", prediction_count)
print("Timestamp NULL   :", timestamp_null)


# ============================================================
# 9. LƯU LẠI
# ============================================================

(
    predictions_output
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


# ============================================================
# 10. ĐỌC LẠI ĐỂ VERIFY
# ============================================================

check_df = spark.read.parquet(OUTPUT_PATH)

check_count = check_df.count()

check_timestamp_null = (
    check_df
    .filter(F.col("timestamp").isNull())
    .count()
)


print("\n================ FINAL CHECK ================")
print("Rows after save :", check_count)
print("Timestamp NULL  :", check_timestamp_null)

if (
    check_count == prediction_count
    and check_timestamp_null == 0
):
    print("PASS: Random Forest prediction file fixed correctly.")
else:
    print("FAIL: Prediction file verification failed.")


# ============================================================
# 11. DỪNG SPARK
# ============================================================

spark.stop()