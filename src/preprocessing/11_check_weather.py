from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Weather dataset nhỏ hơn electricity rất nhiều nên Spark xử lý
# khá nhẹ.
spark = (
    SparkSession.builder
    .appName("CheckWeather")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
weather_path = (
    r"D:\BigData_Electricity\bdg2\data\weather\weather.csv"
)

metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)


# ============================================================
# 3. ĐỌC WEATHER
# ============================================================
weather_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(weather_path)
)


print(
    "\n================ THÔNG TIN WEATHER ================\n"
)

weather_count = weather_df.count()

print(
    f"Số bản ghi weather: {weather_count:,}"
)

print(
    f"Số cột weather: {len(weather_df.columns):,}"
)

print(
    "\nCác cột:"
)

for column in weather_df.columns:
    print(f"- {column}")


# ============================================================
# 4. SCHEMA
# ============================================================
print(
    "\n================ SCHEMA WEATHER ================\n"
)

weather_df.printSchema()


# ============================================================
# 5. CHUẨN HÓA TIMESTAMP
# ============================================================
# Chuyển timestamp về kiểu timestamp để khóa ghép giữa
# electricity và weather có cùng kiểu dữ liệu.
weather_df = (
    weather_df
    .withColumn(
        "timestamp",
        F.to_timestamp("timestamp")
    )
)


# ============================================================
# 6. SỐ SITE VÀ KHOẢNG THỜI GIAN
# ============================================================
weather_site_count = (
    weather_df
    .select("site_id")
    .distinct()
    .count()
)

weather_min_max = (
    weather_df
    .agg(
        F.min("timestamp").alias("min_timestamp"),
        F.max("timestamp").alias("max_timestamp")
    )
    .collect()[0]
)


print(
    "\n================ SITE VÀ THỜI GIAN ================\n"
)

print(
    f"Số site trong weather: "
    f"{weather_site_count:,}"
)

print(
    f"Timestamp đầu tiên: "
    f"{weather_min_max['min_timestamp']}"
)

print(
    f"Timestamp cuối cùng: "
    f"{weather_min_max['max_timestamp']}"
)


# ============================================================
# 7. KIỂM TRA DUPLICATE THEO SITE + TIMESTAMP
# ============================================================
# Khóa logic của weather là:
#
#     site_id + timestamp
#
# Một cặp khóa chỉ nên xuất hiện một lần.
duplicate_weather = (
    weather_df
    .groupBy(
        "site_id",
        "timestamp"
    )
    .count()
    .filter(
        F.col("count") > 1
    )
)


duplicate_weather_count = (
    duplicate_weather.count()
)


print(
    "\n================ DUPLICATE WEATHER ================\n"
)

print(
    f"Số cặp site_id + timestamp bị trùng: "
    f"{duplicate_weather_count:,}"
)


if duplicate_weather_count > 0:

    duplicate_weather.show(
        20,
        truncate=False
    )

else:

    print(
        "Không phát hiện duplicate theo site_id + timestamp."
    )


# ============================================================
# 8. KIỂM TRA SỐ TIMESTAMP CỦA TỪNG SITE
# ============================================================
# Electricity đã được xác nhận có:
#
#     17.544 timestamp
#
# Nếu weather đầy đủ cho một site trong cùng khoảng thời gian
# thì mỗi site dự kiến cũng có 17.544 timestamp.
weather_site_stats = (
    weather_df
    .groupBy("site_id")
    .agg(
        F.countDistinct("timestamp").alias(
            "distinct_timestamps"
        ),
        F.min("timestamp").alias(
            "min_timestamp"
        ),
        F.max("timestamp").alias(
            "max_timestamp"
        )
    )
    .orderBy("site_id")
)


print(
    "\n================ COVERAGE THEO SITE ================\n"
)

weather_site_stats.show(
    30,
    truncate=False
)


# ============================================================
# 9. TÌM SITE KHÔNG ĐỦ 17.544 TIMESTAMP
# ============================================================
incomplete_sites = (
    weather_site_stats
    .filter(
        F.col("distinct_timestamps") != 17544
    )
)


incomplete_site_count = (
    incomplete_sites.count()
)


print(
    "\n================ SITE CÓ THIẾU TIMESTAMP ================\n"
)

print(
    f"Số site không đủ 17.544 timestamp: "
    f"{incomplete_site_count:,}"
)


if incomplete_site_count > 0:

    incomplete_sites.show(
        30,
        truncate=False
    )

else:

    print(
        "Tất cả site đều có đủ 17.544 timestamp."
    )


# ============================================================
# 10. ĐỌC METADATA ĐỂ LẤY DANH SÁCH SITE THỰC TẾ
# ============================================================
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
)


# Chỉ cần danh sách site_id.
metadata_sites = (
    metadata_df
    .select("site_id")
    .filter(F.col("site_id").isNotNull())
    .distinct()
)


metadata_site_count = (
    metadata_sites.count()
)


# ============================================================
# 11. KIỂM TRA SITE CÓ TRONG METADATA NHƯNG KHÔNG CÓ WEATHER
# ============================================================
weather_sites = (
    weather_df
    .select("site_id")
    .distinct()
)


missing_weather_sites = (
    metadata_sites
    .join(
        weather_sites,
        on="site_id",
        how="left_anti"
    )
)


missing_weather_site_count = (
    missing_weather_sites.count()
)


print(
    "\n================ KIỂM TRA SITE METADATA - WEATHER ================\n"
)

print(
    f"Số site trong metadata: "
    f"{metadata_site_count:,}"
)

print(
    f"Số site trong weather: "
    f"{weather_site_count:,}"
)

print(
    f"Site có metadata nhưng không có weather: "
    f"{missing_weather_site_count:,}"
)


if missing_weather_site_count > 0:

    missing_weather_sites.show(
        30,
        truncate=False
    )

else:

    print(
        "Tất cả site trong metadata đều có weather."
    )


# ============================================================
# 12. KIỂM TRA SITE WEATHER KHÔNG CÓ TRONG METADATA
# ============================================================
extra_weather_sites = (
    weather_sites
    .join(
        metadata_sites,
        on="site_id",
        how="left_anti"
    )
)


extra_weather_site_count = (
    extra_weather_sites.count()
)


print(
    "\n================ WEATHER SITE KHÔNG CÓ METADATA ================\n"
)

print(
    f"Số site weather không có trong metadata: "
    f"{extra_weather_site_count:,}"
)


if extra_weather_site_count > 0:

    extra_weather_sites.show(
        30,
        truncate=False
    )

else:

    print(
        "Không có site weather dư."
    )


# ============================================================
# 13. KIỂM TRA MISSING CÁC TRƯỜNG WEATHER
# ============================================================
weather_columns = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "windDirection",
    "windSpeed"
]


print(
    "\n================ MISSING WEATHER ================\n"
)

missing_weather_expressions = []

for column in weather_columns:

    missing_weather_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(column)
    )


missing_weather_summary = (
    weather_df
    .agg(*missing_weather_expressions)
)


missing_weather_summary.show(
    truncate=False
)


# ============================================================
# 14. SAMPLE WEATHER
# ============================================================
print(
    "\n================ SAMPLE WEATHER ================\n"
)

weather_df.show(
    10,
    truncate=False
)


# ============================================================
# 15. KẾT LUẬN
# ============================================================
print(
    "\n================ KẾT LUẬN ================\n"
)

if (
    duplicate_weather_count == 0
    and incomplete_site_count == 0
    and missing_weather_site_count == 0
    and extra_weather_site_count == 0
):

    print(
        "Weather có thể ghép với dataset electricity + metadata "
        "theo khóa site_id + timestamp."
    )

else:

    print(
        "Weather còn vấn đề cần xử lý trước khi JOIN."
    )


# ============================================================
# 16. KẾT THÚC
# ============================================================
spark.stop()