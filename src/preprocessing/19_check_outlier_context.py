from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("CheckOutlierContext")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
input_path = (
    r"D:\BigData_Electricity\data\processed\site_hourly_features"
)


# ============================================================
# 3. ĐỌC DATASET
# ============================================================
df = (
    spark.read
    .parquet(input_path)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings"
    )
)


# ============================================================
# 4. TÍNH Q1 / Q3 THEO SITE
# ============================================================
site_percentiles = (
    df
    .groupBy("site_id")
    .agg(
        F.percentile_approx(
            "total_consumption",
            0.25,
            10000
        ).alias("Q1"),

        F.percentile_approx(
            "total_consumption",
            0.75,
            10000
        ).alias("Q3")
    )
    .withColumn(
        "IQR",
        F.col("Q3") - F.col("Q1")
    )
    .withColumn(
        "upper_bound",
        F.col("Q3") + 1.5 * F.col("IQR")
    )
)


# ============================================================
# 5. ĐÁNH DẤU OUTLIER
# ============================================================
df_outliers = (
    df
    .join(
        F.broadcast(site_percentiles),
        on="site_id",
        how="left"
    )
    .withColumn(
        "is_outlier",
        F.when(
            F.col("total_consumption")
            > F.col("upper_bound"),
            1
        ).otherwise(0)
    )
)


# ============================================================
# 6. TẠO CHỈ SỐ TIÊU THỤ TRÊN BUILDING HOẠT ĐỘNG
# ============================================================
# Chỉ số này KHÔNG thay thế total_consumption.
#
# Mục đích:
# kiểm tra xem một giá trị cực cao đến từ:
#
#     nhiều building cùng tiêu thụ cao
#
# hay:
#
#     rất ít building nhưng tổng lại bất thường.
#
# Nếu active_buildings > 0:
#
#     consumption_per_active_building
#       = total_consumption / active_buildings
#
# Nếu active_buildings = 0:
#     không thể tính nên để NULL.
df_outliers = (
    df_outliers
    .withColumn(
        "consumption_per_active_building",
        F.when(
            F.col("active_buildings") > 0,
            F.col("total_consumption")
            / F.col("active_buildings")
        )
    )
)


# ============================================================
# 7. THỐNG KÊ OUTLIER THEO ACTIVE_BUILDINGS
# ============================================================
print(
    "\n================ OUTLIER THEO ACTIVE BUILDINGS ================\n"
)

(
    df_outliers
    .filter(
        F.col("is_outlier") == 1
    )
    .groupBy("active_buildings")
    .agg(
        F.count("*").alias(
            "outlier_rows"
        ),

        F.avg(
            "total_consumption"
        ).alias(
            "avg_consumption"
        ),

        F.max(
            "total_consumption"
        ).alias(
            "max_consumption"
        )
    )
    .orderBy(
        F.desc("outlier_rows")
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 8. 30 OUTLIER CAO NHẤT
# ============================================================
print(
    "\n================ 30 OUTLIER CAO NHẤT ================\n"
)

(
    df_outliers
    .filter(
        F.col("is_outlier") == 1
    )
    .orderBy(
        F.desc("total_consumption")
    )
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings",
        "consumption_per_active_building",
        "Q1",
        "Q3",
        "upper_bound"
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 9. OUTLIER CÓ CONSUMPTION / ACTIVE_BUILDING CAO NHẤT
# ============================================================
print(
    "\n================ OUTLIER CAO NHẤT THEO BUILDING HOẠT ĐỘNG ================\n"
)

(
    df_outliers
    .filter(
        (F.col("is_outlier") == 1)
        & (F.col("active_buildings") > 0)
    )
    .orderBy(
        F.desc(
            "consumption_per_active_building"
        )
    )
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings",
        "consumption_per_active_building"
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 10. THỐNG KÊ TỶ LỆ OUTLIER THEO SITE
# ============================================================
print(
    "\n================ OUTLIER THEO SITE ================\n"
)

(
    df_outliers
    .groupBy("site_id")
    .agg(
        F.count("*").alias(
            "total_rows"
        ),

        F.sum(
            "is_outlier"
        ).alias(
            "outlier_rows"
        )
    )
    .withColumn(
        "outlier_rate",
        F.col("outlier_rows")
        / F.col("total_rows")
        * 100
    )
    .orderBy(
        F.desc("outlier_rate")
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 11. KIỂM TRA ZERO
# ============================================================
print(
    "\n================ ZERO THEO SITE ================\n"
)

(
    df
    .groupBy("site_id")
    .agg(
        F.count("*").alias(
            "total_rows"
        ),

        F.sum(
            F.when(
                F.col("total_consumption") == 0,
                1
            ).otherwise(0)
        ).alias(
            "zero_rows"
        )
    )
    .withColumn(
        "zero_rate",
        F.col("zero_rows")
        / F.col("total_rows")
        * 100
    )
    .orderBy(
        F.desc("zero_rate")
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 12. KẾT LUẬN
# ============================================================
print(
    "\n================ KẾT LUẬN ================\n"
)

print(
    "Outlier chỉ được sử dụng để đánh dấu và phân tích."
)

print(
    "Không xóa hoặc thay thế total_consumption."
)

print(
    "Giá trị 0 vẫn được giữ nguyên vì 0 là một giá trị điện năng hợp lệ."
)


# ============================================================
# 13. KẾT THÚC
# ============================================================
spark.stop()