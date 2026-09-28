from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime, timedelta


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("CleanWeatherDataFinal")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
input_path = (
    r"D:\BigData_Electricity\bdg2\data\weather\weather.csv"
)

output_path = (
    r"D:\BigData_Electricity\data\processed\weather_cleaned"
)


# ============================================================
# 3. ĐỌC WEATHER GỐC
# ============================================================
weather_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
    .withColumn(
        "timestamp",
        F.to_timestamp("timestamp")
    )
)


# ============================================================
# 4. THÔNG TIN BAN ĐẦU
# ============================================================
original_count = weather_df.count()

site_count = (
    weather_df
    .select("site_id")
    .distinct()
    .count()
)

time_range = (
    weather_df
    .agg(
        F.min("timestamp").alias("min_timestamp"),
        F.max("timestamp").alias("max_timestamp")
    )
    .collect()[0]
)

min_timestamp = time_range["min_timestamp"]
max_timestamp = time_range["max_timestamp"]


print(
    "\n================ WEATHER GỐC ================\n"
)

print(
    f"Số bản ghi: {original_count:,}"
)

print(
    f"Số site: {site_count:,}"
)

print(
    f"Timestamp đầu: {min_timestamp}"
)

print(
    f"Timestamp cuối: {max_timestamp}"
)


# ============================================================
# 5. XỬ LÝ PRECIPITATION ÂM
# ============================================================
# Giá trị âm của precipitation không được xem là lượng mưa
# hợp lệ trong pipeline này.
#
# Không chuyển thành 0 vì:
#     -1 không đồng nghĩa với 0 mm.
#
# Thay vào đó chuyển thành NULL để xử lý bằng median sau.
weather_df = (
    weather_df
    .withColumn(
        "precipDepth1HR",
        F.when(
            F.col("precipDepth1HR") >= 0,
            F.col("precipDepth1HR")
        )
    )
    .withColumn(
        "precipDepth6HR",
        F.when(
            F.col("precipDepth6HR") >= 0,
            F.col("precipDepth6HR")
        )
    )
)


# ============================================================
# 6. CHỈ GIỮ CÁC BIẾN WEATHER SỬ DỤNG
# ============================================================
# Giữ:
# - airTemperature
# - dewTemperature
# - seaLvlPressure
# - windDirection
# - windSpeed
#
# Bỏ:
# - cloudCoverage
# - precipDepth1HR
# - precipDepth6HR
#
# vì các biến bị thiếu quá nhiều.
weather_clean = (
    weather_df
    .select(
        "timestamp",
        "site_id",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed"
    )
)


# ============================================================
# 7. TẠO DANH SÁCH SITE
# ============================================================
sites_df = (
    weather_clean
    .select("site_id")
    .distinct()
)


# ============================================================
# 8. TẠO ĐẦY ĐỦ 17.544 TIMESTAMP THEO GIỜ
# ============================================================
# Weather gốc đang thiếu một số cặp site + timestamp.
#
# Ta tạo lại toàn bộ timeline:
#
# 2016-01-01 00:00:00
#         ...
# 2017-12-31 23:00:00
#
# Tổng cộng:
#     17.544 timestamp
#
# 19 site × 17.544 timestamp
#     = 333.336 dòng
#
# Đây là "khung chuẩn" để sau đó JOIN dữ liệu weather thật.
timestamp_list = []

current_timestamp = min_timestamp

while current_timestamp <= max_timestamp:

    timestamp_list.append(
        (current_timestamp,)
    )

    current_timestamp += timedelta(hours=1)


timestamps_df = (
    spark.createDataFrame(
        timestamp_list,
        ["timestamp"]
    )
)


print(
    "\n================ TIMELINE CHUẨN ================\n"
)

print(
    f"Số timestamp chuẩn: "
    f"{len(timestamp_list):,}"
)

print(
    f"Số site: "
    f"{site_count:,}"
)

print(
    f"Số dòng kỳ vọng sau khi tạo timeline đầy đủ: "
    f"{len(timestamp_list) * site_count:,}"
)


# ============================================================
# 9. TẠO KHUNG SITE + TIMESTAMP ĐẦY ĐỦ
# ============================================================
# Vì chỉ có 19 site và 17.544 timestamp nên tổng cộng
# chỉ có 333.336 dòng.
#
# Kích thước này rất nhỏ so với 25+ triệu dòng electricity.
full_grid = (
    sites_df
    .crossJoin(timestamps_df)
)


# ============================================================
# 10. JOIN WEATHER THẬT VÀO KHUNG ĐẦY ĐỦ
# ============================================================
# Những cặp site + timestamp thực sự tồn tại trong weather
# sẽ lấy giá trị thật.
#
# Những cặp không tồn tại sẽ có NULL và được xử lý ở bước sau.
weather_full = (
    full_grid
    .join(
        weather_clean,
        on=["site_id", "timestamp"],
        how="left"
    )
)


# ============================================================
# 11. TÍNH MEDIAN WEATHER THEO SITE
# ============================================================
site_medians = (
    weather_clean
    .groupBy("site_id")
    .agg(
        F.percentile_approx(
            "airTemperature",
            0.5,
            10000
        ).alias("airTemperature_site_median"),

        F.percentile_approx(
            "dewTemperature",
            0.5,
            10000
        ).alias("dewTemperature_site_median"),

        F.percentile_approx(
            "seaLvlPressure",
            0.5,
            10000
        ).alias("seaLvlPressure_site_median"),

        F.percentile_approx(
            "windDirection",
            0.5,
            10000
        ).alias("windDirection_site_median"),

        F.percentile_approx(
            "windSpeed",
            0.5,
            10000
        ).alias("windSpeed_site_median")
    )
)


# ============================================================
# 12. TÍNH MEDIAN TOÀN BỘ DATASET
# ============================================================
# Đây là giá trị dự phòng.
#
# Ví dụ:
# site Lamb gần như không có seaLvlPressure.
#
# Khi đó:
#     site median = NULL
#
# nên ta dùng:
#     global median
#
# Điều này tốt hơn việc để NULL hoặc thay bằng 0.
global_medians = (
    weather_clean
    .agg(
        F.percentile_approx(
            "airTemperature",
            0.5,
            10000
        ).alias("airTemperature_global_median"),

        F.percentile_approx(
            "dewTemperature",
            0.5,
            10000
        ).alias("dewTemperature_global_median"),

        F.percentile_approx(
            "seaLvlPressure",
            0.5,
            10000
        ).alias("seaLvlPressure_global_median"),

        F.percentile_approx(
            "windDirection",
            0.5,
            10000
        ).alias("windDirection_global_median"),

        F.percentile_approx(
            "windSpeed",
            0.5,
            10000
        ).alias("windSpeed_global_median")
    )
)


# ============================================================
# 13. GHÉP MEDIAN VÀO FULL GRID
# ============================================================
weather_full = (
    weather_full
    .join(
        F.broadcast(site_medians),
        on="site_id",
        how="left"
    )
    .crossJoin(
        F.broadcast(global_medians)
    )
)


# ============================================================
# 14. ĐIỀN MISSING THEO 2 CẤP
# ============================================================
# Ưu tiên:
#
#     Giá trị thật
#          ↓
#     Median của site
#          ↓
#     Median toàn bộ dataset
#
# Như vậy không dùng 0 giả tạo.
weather_filled = (
    weather_full

    .withColumn(
        "airTemperature",
        F.coalesce(
            F.col("airTemperature"),
            F.col("airTemperature_site_median"),
            F.col("airTemperature_global_median")
        )
    )

    .withColumn(
        "dewTemperature",
        F.coalesce(
            F.col("dewTemperature"),
            F.col("dewTemperature_site_median"),
            F.col("dewTemperature_global_median")
        )
    )

    .withColumn(
        "seaLvlPressure",
        F.coalesce(
            F.col("seaLvlPressure"),
            F.col("seaLvlPressure_site_median"),
            F.col("seaLvlPressure_global_median")
        )
    )

    .withColumn(
        "windDirection",
        F.coalesce(
            F.col("windDirection"),
            F.col("windDirection_site_median"),
            F.col("windDirection_global_median")
        )
    )

    .withColumn(
        "windSpeed",
        F.coalesce(
            F.col("windSpeed"),
            F.col("windSpeed_site_median"),
            F.col("windSpeed_global_median")
        )
    )

    .drop(
        "airTemperature_site_median",
        "dewTemperature_site_median",
        "seaLvlPressure_site_median",
        "windDirection_site_median",
        "windSpeed_site_median",
        "airTemperature_global_median",
        "dewTemperature_global_median",
        "seaLvlPressure_global_median",
        "windDirection_global_median",
        "windSpeed_global_median"
    )
)


# ============================================================
# 15. SẮP XẾP CỘT
# ============================================================
weather_filled = (
    weather_filled
    .select(
        "timestamp",
        "site_id",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed"
    )
)


# ============================================================
# 16. KIỂM TRA MISSING
# ============================================================
weather_columns = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windDirection",
    "windSpeed"
]


missing_expressions = []

for column in weather_columns:

    missing_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(column)
    )


missing_result = (
    weather_filled
    .agg(*missing_expressions)
    .collect()[0]
)


print(
    "\n================ MISSING SAU XỬ LÝ ================\n"
)

for column in weather_columns:

    print(
        f"{column:<20}: "
        f"{missing_result[column]:,}"
    )


# ============================================================
# 17. KIỂM TRA DUPLICATE
# ============================================================
duplicate_count = (
    weather_filled
    .groupBy(
        "site_id",
        "timestamp"
    )
    .count()
    .filter(
        F.col("count") > 1
    )
    .count()
)


print(
    "\n================ DUPLICATE ================\n"
)

print(
    f"Duplicate site_id + timestamp: "
    f"{duplicate_count:,}"
)


# ============================================================
# 18. KIỂM TRA SỐ DÒNG
# ============================================================
final_count = (
    weather_filled.count()
)

expected_count = (
    site_count
    * len(timestamp_list)
)


print(
    "\n================ KIỂM TRA SỐ DÒNG ================\n"
)

print(
    f"Số dòng kỳ vọng: "
    f"{expected_count:,}"
)

print(
    f"Số dòng thực tế: "
    f"{final_count:,}"
)


# ============================================================
# 19. SAMPLE
# ============================================================
print(
    "\n================ SAMPLE WEATHER SẠCH ================\n"
)

weather_filled.show(
    20,
    truncate=False
)


# ============================================================
# 20. LƯU PARQUET
# ============================================================
print(
    "\n================ ĐANG LƯU WEATHER SẠCH ================\n"
)

(
    weather_filled
    .write
    .mode("overwrite")
    .parquet(output_path)
)


# ============================================================
# 21. ĐỌC LẠI XÁC NHẬN
# ============================================================
verification_df = (
    spark.read
    .parquet(output_path)
)


verification_count = (
    verification_df.count()
)


print(
    "\n================ XÁC NHẬN OUTPUT ================\n"
)

print(
    f"Số dòng sau khi đọc lại: "
    f"{verification_count:,}"
)

if verification_count == expected_count:

    print(
        "Kiểm tra thành công: Weather có đầy đủ "
        "19 site × 17.544 timestamp."
    )

else:

    print(
        "CẢNH BÁO: số dòng Weather không đúng."
    )


print(
    "\n================ HOÀN TẤT ================\n"
)

print(
    f"Weather sạch đã lưu tại:\n{output_path}"
)


# ============================================================
# 22. KẾT THÚC SPARK
# ============================================================
spark.stop()