from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Dataset electricity có khoảng 25,9 triệu bản ghi.
# Metadata và weather nhỏ hơn rất nhiều.
#
# Vì vậy:
# - broadcast metadata
# - broadcast weather
#
# giúp tránh shuffle toàn bộ bảng electricity.
spark = (
    SparkSession.builder
    .appName("JoinAllElectricityMetadataWeather")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.autoBroadcastJoinThreshold", "50MB")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\data\processed\electricity_cleaned"
)

metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)

weather_path = (
    r"D:\BigData_Electricity\data\processed\weather_cleaned"
)

output_path = (
    r"D:\BigData_Electricity\data\processed\electricity_final"
)


# ============================================================
# 3. ĐỌC ELECTRICITY
# ============================================================
electricity_df = (
    spark.read
    .parquet(electricity_path)
)

electricity_count = (
    electricity_df.count()
)


print(
    "\n================ ELECTRICITY ================\n"
)

print(
    f"Số bản ghi: {electricity_count:,}"
)

print(
    f"Số building: "
    f"{electricity_df.select('building_id').distinct().count():,}"
)


# ============================================================
# 4. ĐỌC METADATA
# ============================================================
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
)


# ============================================================
# 5. CHỌN METADATA CẦN THIẾT
# ============================================================
metadata_selected = (
    metadata_df
    .select(
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "sqft",
        "lat",
        "lng",
        "timezone",
        "yearbuilt",
        "numberoffloors",
        "occupants",
        "eui",
        "site_eui",
        "source_eui",
        "leed_level",
        "rating"
    )
    .withColumn(
        "eui",
        F.expr("try_cast(eui AS DOUBLE)")
    )
    .withColumn(
        "site_eui",
        F.expr("try_cast(site_eui AS DOUBLE)")
    )
    .withColumn(
        "source_eui",
        F.expr("try_cast(source_eui AS DOUBLE)")
    )
)


print(
    "\n================ METADATA ================\n"
)

print(
    f"Số dòng metadata: "
    f"{metadata_selected.count():,}"
)


# ============================================================
# 6. ĐỌC WEATHER SẠCH
# ============================================================
weather_df = (
    spark.read
    .parquet(weather_path)
)


# Chỉ lấy những cột weather đã được xử lý sạch.
weather_selected = (
    weather_df
    .select(
        "site_id",
        "timestamp",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed"
    )
)


weather_count = (
    weather_selected.count()
)


print(
    "\n================ WEATHER ================\n"
)

print(
    f"Số bản ghi: {weather_count:,}"
)

print(
    f"Số site: "
    f"{weather_selected.select('site_id').distinct().count():,}"
)


# ============================================================
# 7. JOIN METADATA VÀO ELECTRICITY
# ============================================================
# LEFT JOIN đảm bảo không làm mất bản ghi electricity.
#
# Metadata có building_id duy nhất nên mỗi bản ghi electricity
# chỉ nhận tối đa một dòng metadata.
print(
    "\n================ JOIN METADATA ================\n"
)

electricity_with_metadata = (
    electricity_df
    .join(
        F.broadcast(metadata_selected),
        on="building_id",
        how="left"
    )
)


# ============================================================
# 8. JOIN WEATHER
# ============================================================
# Khóa:
#
#     site_id
#     +
#     timestamp
#
# Weather đã được chuẩn hóa thành đầy đủ 19 site × 17.544
# timestamp và không có duplicate.
#
# Vì vậy LEFT JOIN sẽ giữ nguyên toàn bộ electricity.
print(
    "\n================ JOIN WEATHER ================\n"
)

final_df = (
    electricity_with_metadata
    .join(
        F.broadcast(weather_selected),
        on=["site_id", "timestamp"],
        how="left"
    )
)


# ============================================================
# 9. KIỂM TRA SCHEMA
# ============================================================
print(
    "\n================ SCHEMA DATASET CUỐI ================\n"
)

final_df.printSchema()


# ============================================================
# 10. KIỂM TRA SAMPLE
# ============================================================
print(
    "\n================ SAMPLE DATASET CUỐI ================\n"
)

final_df.show(
    20,
    truncate=False
)


# ============================================================
# 11. KIỂM TRA SỐ BUILDING CÓ NULL METADATA
# ============================================================
# Chỉ kiểm tra site_id vì site_id là khóa cần thiết cho weather.
missing_site_id = (
    final_df
    .filter(F.col("site_id").isNull())
    .count()
)


print(
    "\n================ KIỂM TRA METADATA SAU JOIN ================\n"
)

print(
    f"Bản ghi không có site_id: "
    f"{missing_site_id:,}"
)


# ============================================================
# 12. KIỂM TRA MISSING WEATHER
# ============================================================
weather_columns = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windDirection",
    "windSpeed"
]


weather_missing_expressions = []

for column in weather_columns:

    weather_missing_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(column)
    )


weather_missing = (
    final_df
    .agg(*weather_missing_expressions)
)


print(
    "\n================ MISSING WEATHER SAU JOIN ================\n"
)

weather_missing.show(
    truncate=False
)


# ============================================================
# 13. GHI DATASET FINAL
# ============================================================
print(
    "\n================ ĐANG LƯU DATASET FINAL ================\n"
)

(
    final_df
    .write
    .mode("overwrite")
    .parquet(output_path)
)


# ============================================================
# 14. ĐỌC LẠI DATASET
# ============================================================
# Đọc lại để xác nhận file Parquet thực sự được tạo đúng.
verification_df = (
    spark.read
    .parquet(output_path)
)


verification_count = (
    verification_df.count()
)


verification_building_count = (
    verification_df
    .select("building_id")
    .distinct()
    .count()
)


# ============================================================
# 15. KIỂM TRA KẾT QUẢ CUỐI
# ============================================================
print(
    "\n================ XÁC NHẬN DATASET FINAL ================\n"
)

print(
    f"Bản ghi electricity ban đầu: "
    f"{electricity_count:,}"
)

print(
    f"Bản ghi dataset final: "
    f"{verification_count:,}"
)

print(
    f"Building dataset final: "
    f"{verification_building_count:,}"
)


if verification_count == electricity_count:

    print(
        "PASS: Số bản ghi không thay đổi sau JOIN."
    )

else:

    print(
        "FAIL: Số bản ghi thay đổi sau JOIN."
    )


if verification_building_count == 1514:

    print(
        "PASS: Dataset vẫn có 1.514 building."
    )

else:

    print(
        "FAIL: Số building không đúng."
    )


# ============================================================
# 16. KIỂM TRA WEATHER KEY
# ============================================================
# Với weather đã chuẩn hóa:
#
# 19 site × 17.544 timestamp = 333.336
#
# và electricity có site_id hợp lệ, tất cả các bản ghi
# electricity trong khoảng thời gian chuẩn phải lấy được
# weather.
print(
    "\n================ KIỂM TRA WEATHER KEY ================\n"
)

missing_weather_key = (
    final_df
    .filter(
        F.col("airTemperature").isNull()
        | F.col("dewTemperature").isNull()
        | F.col("seaLvlPressure").isNull()
        | F.col("windDirection").isNull()
        | F.col("windSpeed").isNull()
    )
    .count()
)


print(
    f"Bản ghi thiếu weather sau JOIN: "
    f"{missing_weather_key:,}"
)


if missing_weather_key == 0:

    print(
        "PASS: Tất cả bản ghi electricity đều có dữ liệu weather."
    )

else:

    print(
        "CẢNH BÁO: Vẫn còn bản ghi thiếu weather."
    )


print(
    "\n================ HOÀN TẤT ================\n"
)

print(
    f"Dataset final đã lưu tại:\n{output_path}"
)


# ============================================================
# 17. KẾT THÚC SPARK
# ============================================================
spark.stop()