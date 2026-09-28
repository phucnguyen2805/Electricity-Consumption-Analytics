from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("ValidateSiteFilter")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)

electricity_final_path = (
    r"D:\BigData_Electricity\data\processed\electricity_final"
)

hourly_path = (
    r"D:\BigData_Electricity\data\processed\site_hourly_features"
)


# ============================================================
# 3. ĐỌC METADATA
# ============================================================
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
    .select(
        "building_id",
        "site_id"
    )
)


# ============================================================
# 4. BUILDING TRONG METADATA
# ============================================================
metadata_site_stats = (
    metadata_df
    .groupBy("site_id")
    .agg(
        F.countDistinct(
            "building_id"
        ).alias(
            "metadata_buildings"
        )
    )
)


# ============================================================
# 5. ĐỌC ELECTRICITY FINAL
# ============================================================
# electricity_final đã được JOIN metadata nên có:
#
#     building_id
#     site_id
#
# Đây là dataset đúng để xác định building nào còn lại
# sau khi lọc missing >50%.
electricity_final_df = (
    spark.read
    .parquet(electricity_final_path)
    .select(
        "building_id",
        "site_id"
    )
)


# ============================================================
# 6. LẤY DANH SÁCH BUILDING DUY NHẤT
# ============================================================
cleaned_buildings = (
    electricity_final_df
    .distinct()
)


# ============================================================
# 7. ĐẾM BUILDING CÒN LẠI THEO SITE
# ============================================================
final_site_stats = (
    cleaned_buildings
    .groupBy("site_id")
    .agg(
        F.countDistinct(
            "building_id"
        ).alias(
            "final_buildings"
        )
    )
)


# ============================================================
# 8. SO SÁNH 19 SITE
# ============================================================
site_comparison = (
    metadata_site_stats
    .join(
        final_site_stats,
        on="site_id",
        how="left"
    )
    .fillna(
        0,
        subset=["final_buildings"]
    )
    .withColumn(
        "removed_buildings",
        F.col("metadata_buildings")
        - F.col("final_buildings")
    )
    .orderBy("site_id")
)


print(
    "\n================ SO SÁNH BUILDING THEO SITE ================\n"
)

site_comparison.show(
    30,
    truncate=False
)


# ============================================================
# 9. KIỂM TRA RIÊNG SITE SWAN
# ============================================================
swan_metadata = (
    metadata_df
    .filter(
        F.col("site_id") == "Swan"
    )
)

swan_final = (
    cleaned_buildings
    .filter(
        F.col("site_id") == "Swan"
    )
)


swan_metadata_count = (
    swan_metadata.count()
)

swan_final_count = (
    swan_final.count()
)


print(
    "\n================ KIỂM TRA SITE SWAN ================\n"
)

print(
    f"Building Swan trong metadata: "
    f"{swan_metadata_count:,}"
)

print(
    f"Building Swan còn trong electricity_final: "
    f"{swan_final_count:,}"
)


# ============================================================
# 10. BUILDING SWAN BỊ LOẠI
# ============================================================
removed_swan = (
    swan_metadata
    .join(
        swan_final,
        on="building_id",
        how="left_anti"
    )
    .orderBy("building_id")
)


print(
    "\n================ BUILDING SWAN BỊ LOẠI ================\n"
)

removed_swan.show(
    200,
    truncate=False
)


# ============================================================
# 11. KIỂM TRA SITE-HOURLY
# ============================================================
hourly_df = (
    spark.read
    .parquet(hourly_path)
)


hourly_site_stats = (
    hourly_df
    .groupBy("site_id")
    .agg(
        F.count("*").alias(
            "hourly_rows"
        ),

        F.countDistinct(
            "timestamp"
        ).alias(
            "timestamp_count"
        )
    )
    .withColumn(
        "expected_timestamps",
        F.lit(17544)
    )
    .withColumn(
        "missing_timestamps",
        F.col("expected_timestamps")
        - F.col("timestamp_count")
    )
    .withColumn(
        "coverage_rate",
        F.col("timestamp_count")
        / F.col("expected_timestamps")
        * 100
    )
    .orderBy(
        F.col("coverage_rate").asc()
    )
)


print(
    "\n================ SITE-HOURLY COVERAGE ================\n"
)

hourly_site_stats.show(
    30,
    truncate=False
)


# ============================================================
# 12. TỔNG QUAN
# ============================================================
all_site_count = (
    metadata_df
    .select("site_id")
    .distinct()
    .count()
)

hourly_site_count = (
    hourly_df
    .select("site_id")
    .distinct()
    .count()
)

hourly_row_count = (
    hourly_df.count()
)

expected_active_site_rows = (
    hourly_site_count
    * 17544
)

missing_site_hour = (
    expected_active_site_rows
    - hourly_row_count
)


print(
    "\n================ TỔNG QUAN SITE ================\n"
)

print(
    f"Số site trong metadata: "
    f"{all_site_count:,}"
)

print(
    f"Số site có electricity sau lọc: "
    f"{hourly_site_count:,}"
)

print(
    f"Số dòng site-hourly: "
    f"{hourly_row_count:,}"
)

print(
    f"Kỳ vọng nếu {hourly_site_count} site đều đủ 17.544 giờ: "
    f"{expected_active_site_rows:,}"
)

print(
    f"Site-hour bị thiếu trong các site còn lại: "
    f"{missing_site_hour:,}"
)


# ============================================================
# 13. KẾT LUẬN
# ============================================================
print(
    "\n================ KẾT LUẬN ================\n"
)

if (
    swan_final_count == 0
    and hourly_site_count == 18
):

    print(
        "Site Swan không còn dữ liệu electricity sau bước "
        "lọc building missing >50%."
    )

    print(
        "Dataset site-hourly hiện chứa 18 site có dữ liệu."
    )

else:

    print(
        "Cần kiểm tra thêm nguyên nhân site coverage."
    )


print(
    "\n================ HOÀN TẤT ================\n"
)


spark.stop()