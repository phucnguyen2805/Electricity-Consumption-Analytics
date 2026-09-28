from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("ValidateOutlierMethod")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐỌC DỮ LIỆU SITE-HOURLY
# ============================================================

INPUT_PATH = r"D:\BigData_Electricity\data\processed\site_hourly_features"

df = spark.read.parquet(INPUT_PATH)

print("\n================ DATASET INFO ================")
print("Total rows:", df.count())
print("Total sites:", df.select("site_id").distinct().count())


# ============================================================
# 3. TÍNH Q1 VÀ Q3 THEO TỪNG SITE
#
# percentile() được dùng thay vì percentile_approx()
# để có kết quả nhất quán hơn cho bước kiểm tra cuối.
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
# 4. JOIN NGƯỠNG IQR TRỞ LẠI DATA
# ============================================================

df_check = (
    df
    .join(quartiles, on="site_id", how="left")
    .withColumn(
        "is_outlier",
        (
            (F.col("total_consumption") < F.col("lower_bound")) |
            (F.col("total_consumption") > F.col("upper_bound"))
        ).cast("int")
    )
)


# ============================================================
# 5. TỔNG SỐ OUTLIER
# ============================================================

total_rows = df_check.count()

outlier_rows = (
    df_check
    .filter(F.col("is_outlier") == 1)
    .count()
)

outlier_rate = outlier_rows / total_rows * 100

print("\n================ IQR VALIDATION ================")
print(f"Total rows       : {total_rows}")
print(f"Outlier rows     : {outlier_rows}")
print(f"Outlier rate     : {outlier_rate:.4f}%")


# ============================================================
# 6. THỐNG KÊ THEO SITE
# ============================================================

site_result = (
    df_check
    .groupBy("site_id")
    .agg(
        F.count("*").alias("total_rows"),
        F.sum("is_outlier").alias("outlier_rows"),
        F.first("q1").alias("q1"),
        F.first("q3").alias("q3"),
        F.first("upper_bound").alias("upper_bound")
    )
    .withColumn(
        "outlier_rate",
        F.col("outlier_rows") / F.col("total_rows") * 100
    )
    .orderBy(F.desc("outlier_rate"))
)

print("\n================ OUTLIER THEO SITE ================")

site_result.show(30, truncate=False)


# ============================================================
# 7. KIỂM TRA RIÊNG CÁC GIÁ TRỊ LỚN NHẤT
# ============================================================

print("\n================ TOP OUTLIER ================")

(
    df_check
    .filter(F.col("is_outlier") == 1)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings",
        "q1",
        "q3",
        "upper_bound"
    )
    .orderBy(F.desc("total_consumption"))
    .show(30, truncate=False)
)


# ============================================================
# 8. KIỂM TRA TÍNH NHẤT QUÁN
# ============================================================

calculated_sum = (
    site_result
    .agg(F.sum("outlier_rows").alias("total"))
    .collect()[0]["total"]
)

print("\n================ CHECK ================")
print("Outlier count direct :", outlier_rows)
print("Outlier count by site:", calculated_sum)

if outlier_rows == calculated_sum:
    print("PASS: Outlier count is consistent.")
else:
    print("FAIL: Outlier count is NOT consistent.")


# ============================================================
# 9. DỪNG SPARK
# ============================================================

spark.stop()