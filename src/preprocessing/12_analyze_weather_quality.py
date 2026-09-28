from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("AnalyzeWeatherQuality")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN WEATHER
# ============================================================
weather_path = (
    r"D:\BigData_Electricity\bdg2\data\weather\weather.csv"
)


# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================
weather_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(weather_path)
    .withColumn(
        "timestamp",
        F.to_timestamp("timestamp")
    )
)


# ============================================================
# 4. CÁC CỘT WEATHER
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


total_rows = weather_df.count()


print(
    "\n================ TỔNG QUAN WEATHER ================\n"
)

print(
    f"Tổng số bản ghi: {total_rows:,}"
)

print(
    f"Số site: "
    f"{weather_df.select('site_id').distinct().count():,}"
)


# ============================================================
# 5. TỶ LỆ MISSING TOÀN BỘ WEATHER
# ============================================================
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


overall_missing = (
    weather_df
    .agg(*missing_expressions)
    .collect()[0]
)


print(
    "\n================ MISSING TOÀN BỘ WEATHER ================\n"
)

for column in weather_columns:

    missing_count = overall_missing[column]

    if missing_count is None:
        missing_count = 0

    missing_rate = (
        missing_count
        / total_rows
        * 100
    )

    print(
        f"{column:<20} "
        f"| Missing: {missing_count:>8,} "
        f"| Rate: {missing_rate:>8.4f}%"
    )


# ============================================================
# 6. MISSING THEO SITE
# ============================================================
# Việc này rất quan trọng vì weather được JOIN theo site_id.
#
# Ví dụ:
# Site A có 0,1% missing
# Site B có 30% missing
#
# thì chiến lược xử lý có thể khác nhau.
site_missing_expressions = []

for column in weather_columns:

    site_missing_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(
            f"{column}_missing"
        )
    )


site_missing = (
    weather_df
    .groupBy("site_id")
    .agg(
        F.count("*").alias("total_rows"),
        *site_missing_expressions
    )
)


print(
    "\n================ MISSING THEO SITE ================\n"
)

site_missing.show(
    25,
    truncate=False
)


# ============================================================
# 7. KIỂM TRA CÁC GIÁ TRỊ ĐẶC BIỆT
# ============================================================
# Dataset có sample precipDepth1HR = -1.
#
# Không nên tự động thay thế ngay.
# Trước tiên thống kê chúng xuất hiện bao nhiêu lần.
print(
    "\n================ GIÁ TRỊ ĐẶC BIỆT / ÂM ================\n"
)


negative_columns = [
    "airTemperature",
    "cloudCoverage",
    "dewTemperature",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "windDirection",
    "windSpeed"
]


for column in negative_columns:

    negative_count = (
        weather_df
        .filter(
            F.col(column).isNotNull()
            & (F.col(column) < 0)
        )
        .count()
    )

    print(
        f"{column:<20} "
        f"| Số giá trị âm: {negative_count:,}"
    )


# ============================================================
# 8. THỐNG KÊ MIN / MAX
# ============================================================
print(
    "\n================ MIN / MAX WEATHER ================\n"
)

min_max_expressions = []

for column in weather_columns:

    min_max_expressions.extend(
        [
            F.min(column).alias(
                f"{column}_min"
            ),
            F.max(column).alias(
                f"{column}_max"
            )
        ]
    )


min_max_row = (
    weather_df
    .agg(*min_max_expressions)
    .collect()[0]
)


for column in weather_columns:

    print(
        f"{column:<20} "
        f"| min = {min_max_row[f'{column}_min']} "
        f"| max = {min_max_row[f'{column}_max']}"
    )


# ============================================================
# 9. KIỂM TRA DUPLICATE
# ============================================================
duplicate_count = (
    weather_df
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
# 10. KIỂM TRA TIMESTAMP CỦA WEATHER
# ============================================================
# Việc này kiểm tra khoảng thời gian của từng site.
site_time = (
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
    "\n================ TIME COVERAGE THEO SITE ================\n"
)

site_time.show(
    25,
    truncate=False
)


# ============================================================
# 11. KẾT LUẬN
# ============================================================
print(
    "\n================ HOÀN TẤT ================\n"
)

print(
    "Đã hoàn thành kiểm tra missing, giá trị âm, "
    "min/max, duplicate và time coverage của weather."
)


# ============================================================
# 12. KẾT THÚC SPARK
# ============================================================
spark.stop()