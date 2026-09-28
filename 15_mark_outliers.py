from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("MarkOutliers")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

INPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_features"

OUTPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_features_final"


# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================

df = spark.read.parquet(INPUT_PATH)

print("\n================ INPUT DATA ================")
print("Rows:", df.count())
print("Sites:", df.select("site_id").distinct().count())


# ============================================================
# 4. TÍNH Q1, Q3 THEO TỪNG SITE
#
# Mỗi site có một ngưỡng outlier riêng.
# Điều này quan trọng vì quy mô tiêu thụ điện giữa các site
# rất khác nhau.
# ============================================================

quartiles = (
    df
    .groupBy("site_id")
    .agg(
        F.percentile("total_consumption", 0.25).alias("q1"),
        F.percentile("total_consumption", 0.75).alias("q3")
    )
    .withColumn(
        "iqr",
        F.col("q3") - F.col("q1")
    )
    .withColumn(
        "lower_bound",
        F.col("q1") - 1.5 * F.col("iqr")
    )
    .withColumn(
        "upper_bound",
        F.col("q3") + 1.5 * F.col("iqr")
    )
)



# ============================================================
# 5. GẮN NGƯỠNG IQR VÀO DATA
# ============================================================

df_final = (
    df
    .join(
        quartiles.select(
            "site_id",
            "q1",
            "q3",
            "lower_bound",
            "upper_bound"
        ),
        on="site_id",
        how="left"
    )
    .withColumn(
        "is_outlier",
        (
            (F.col("total_consumption") < F.col("lower_bound")) |
            (F.col("total_consumption") > F.col("upper_bound"))
        ).cast("int")
    )
)


# ============================================================
# 6. KIỂM TRA KẾT QUẢ
# ============================================================

total_rows = df_final.count()

outlier_rows = (
    df_final
    .filter(F.col("is_outlier") == 1)
    .count()
)

normal_rows = total_rows - outlier_rows

outlier_rate = outlier_rows / total_rows * 100

print("\n================ OUTLIER RESULT ================")
print(f"Total rows   : {total_rows}")
print(f"Normal rows  : {normal_rows}")
print(f"Outlier rows : {outlier_rows}")
print(f"Outlier rate : {outlier_rate:.4f}%")


# ============================================================
# 7. KIỂM TRA GIÁ TRỊ 0
#
# Đảm bảo chúng ta không loại bỏ hoặc thay thế consumption = 0.
# ============================================================

zero_rows = (
    df_final
    .filter(F.col("total_consumption") == 0)
    .count()
)

print("\n================ ZERO CHECK ================")
print("Zero consumption rows:", zero_rows)


# ============================================================
# 8. LƯU DATASET
# ============================================================

(
    df_final
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

print("\n================ SAVE ================")
print("Output:", OUTPUT_PATH)


# ============================================================
# 9. CHECKPOINT CUỐI
# ============================================================

check_df = spark.read.parquet(OUTPUT_PATH)

check_rows = check_df.count()
check_outliers = (
    check_df
    .filter(F.col("is_outlier") == 1)
    .count()
)

print("\n================ FINAL CHECK ================")
print("Rows after save     :", check_rows)
print("Outliers after save :", check_outliers)

if check_rows == total_rows and check_outliers == outlier_rows:
    print("PASS: Dataset saved correctly.")
else:
    print("FAIL: Dataset verification failed.")


# ============================================================
# 10. DỪNG SPARK
# ============================================================

spark.stop()