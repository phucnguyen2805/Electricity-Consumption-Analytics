from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("CheckSiteCoverage")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
electricity_final_path = (
    r"D:\BigData_Electricity\data\processed\electricity_final"
)

hourly_path = (
    r"D:\BigData_Electricity\data\processed\site_hourly_features"
)

metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)


# ============================================================
# 3. ĐỌC METADATA
# ============================================================
# Metadata chứa danh sách đầy đủ 19 site.
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
)

all_sites = (
    metadata_df
    .select("site_id")
    .filter(F.col("site_id").isNotNull())
    .distinct()
)


# ============================================================
# 4. ĐỌC ELECTRICITY FINAL
# ============================================================
electricity_df = (
    spark.read
    .parquet(electricity_final_path)
    .select(
        "site_id",
        "timestamp",
        "building_id",
        "consumption"
    )
)


# ============================================================
# 5. THỐNG KÊ THEO SITE
# ============================================================
site_stats = (
    electricity_df
    .groupBy("site_id")
    .agg(
        F.count("*").alias("electricity_rows"),

        F.countDistinct(
            "building_id"
        ).alias("building_count"),

        F.countDistinct(
            "timestamp"
        ).alias("timestamp_count"),

        F.min(
            "timestamp"
        ).alias("min_timestamp"),

        F.max(
            "timestamp"
        ).alias("max_timestamp")
    )
    .orderBy("site_id")
)


print(
    "\n================ COVERAGE ELECTRICITY THEO SITE ================\n"
)

site_stats.show(
    30,
    truncate=False
)


# ============================================================
# 6. TÌM SITE CÓ TRONG METADATA NHƯNG KHÔNG CÓ ELECTRICITY
# ============================================================
missing_sites = (
    all_sites
    .join(
        site_stats.select("site_id"),
        on="site_id",
        how="left_anti"
    )
)

missing_site_count = missing_sites.count()


print(
    "\n================ SITE KHÔNG CÓ ELECTRICITY ================\n"
)

print(
    f"Số site không có electricity: "
    f"{missing_site_count:,}"
)

if missing_site_count > 0:
    missing_sites.show(
        30,
        truncate=False
    )


# ============================================================
# 7. KIỂM TRA SITE-HOURLY
# ============================================================
hourly_df = (
    spark.read
    .parquet(hourly_path)
)


hourly_site_stats = (
    hourly_df
    .groupBy("site_id")
    .agg(
        F.count("*").alias("hourly_rows"),

        F.countDistinct(
            "timestamp"
        ).alias("hourly_timestamps"),

        F.min(
            "timestamp"
        ).alias("min_timestamp"),

        F.max(
            "timestamp"
        ).alias("max_timestamp")
    )
    .orderBy("site_id")
)


print(
    "\n================ COVERAGE SITE-HOURLY ================\n"
)

hourly_site_stats.show(
    30,
    truncate=False
)


# ============================================================
# 8. TÍNH SỐ GIỜ THIẾU CỦA TỪNG SITE
# ============================================================
hourly_site_stats = (
    hourly_site_stats
    .withColumn(
        "expected_timestamps",
        F.lit(17544)
    )
    .withColumn(
        "missing_timestamps",
        F.col("expected_timestamps")
        - F.col("hourly_timestamps")
    )
    .withColumn(
        "coverage_rate",
        F.col("hourly_timestamps")
        / F.col("expected_timestamps")
        * 100
    )
)


print(
    "\n================ TIMESTAMP COVERAGE SITE-HOURLY ================\n"
)

hourly_site_stats.show(
    30,
    truncate=False
)


# ============================================================
# 9. SO SÁNH SITE METADATA VÀ SITE-HOURLY
# ============================================================
hourly_sites = (
    hourly_df
    .select("site_id")
    .distinct()
)


extra_check = (
    all_sites
    .join(
        hourly_sites,
        on="site_id",
        how="left"
    )
    .withColumn(
        "has_hourly",
        F.when(
            F.col("site_id").isNotNull()
            & F.col("site_id").isin(
                [
                    row["site_id"]
                    for row in hourly_sites.collect()
                ]
            ),
            1
        ).otherwise(0)
    )
)


# ============================================================
# 10. TỔNG QUAN
# ============================================================
all_site_count = all_sites.count()

hourly_site_count = hourly_sites.count()

total_hourly_rows = hourly_df.count()

expected_for_hourly_sites = (
    hourly_site_count * 17544
)


print(
    "\n================ TỔNG QUAN SITE ================\n"
)

print(
    f"Số site trong metadata: "
    f"{all_site_count:,}"
)

print(
    f"Số site trong site-hourly: "
    f"{hourly_site_count:,}"
)

print(
    f"Số dòng site-hourly: "
    f"{total_hourly_rows:,}"
)

print(
    f"Nếu các site đều đủ 17.544 giờ: "
    f"{expected_for_hourly_sites:,}"
)

print(
    f"Số timestamp site-hourly bị thiếu so với kỳ vọng: "
    f"{expected_for_hourly_sites - total_hourly_rows:,}"
)


print(
    "\n================ HOÀN TẤT ================\n"
)


spark.stop()