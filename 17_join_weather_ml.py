from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("JoinWeatherForML")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

ML_INPUT = (
    r"D:\BigData_Electricity\data\processed"
    r"\site_hourly_ml_features"
)

WEATHER_INPUT = (
    r"D:\BigData_Electricity\data\processed"
    r"\weather_cleaned"
)

OUTPUT_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\electricity_ml_dataset"
)


# ============================================================
# 3. ĐỌC DATA
# ============================================================

df_ml = spark.read.parquet(ML_INPUT)

weather = spark.read.parquet(WEATHER_INPUT)


print("\n================ INPUT ================")
print("ML rows :", df_ml.count())
print("ML sites:", df_ml.select("site_id").distinct().count())

print("\n================ WEATHER ================")
print("Weather rows :", weather.count())
print("Weather sites:", weather.select("site_id").distinct().count())


# ============================================================
# 4. CHỈ GIỮ CÁC WEATHER FEATURES
#
# Đổi tên cột weather ngay từ đầu để tránh trường hợp
# trùng tên với các cột đã có trong dataset ML.
# ============================================================

weather_features = (
    weather
    .select(
        "site_id",
        "timestamp",
        F.col("airTemperature").alias("weather_air_temperature"),
        F.col("dewTemperature").alias("weather_dew_temperature"),
        F.col("seaLvlPressure").alias("weather_sea_level_pressure"),
        F.col("windDirection").alias("weather_wind_direction"),
        F.col("windSpeed").alias("weather_wind_speed")
    )
)


# ============================================================
# 5. LEFT JOIN WEATHER VÀO DATASET ML
#
# LEFT JOIN để giữ nguyên toàn bộ full grid.
# ============================================================

df_result = (
    df_ml
    .join(
        weather_features,
        on=["site_id", "timestamp"],
        how="left"
    )
)


# ============================================================
# 6. TẠO FEATURES CHO WIND DIRECTION
#
# Hướng gió là dữ liệu tuần hoàn:
# 0° và 360° thực chất cùng một hướng.
#
# Vì vậy chuyển sang:
#   sin(direction)
#   cos(direction)
# ============================================================

df_result = (
    df_result
    .withColumn(
        "wind_dir_rad",
        F.radians(F.col("weather_wind_direction"))
    )
    .withColumn(
        "wind_dir_sin",
        F.sin(F.col("wind_dir_rad"))
    )
    .withColumn(
        "wind_dir_cos",
        F.cos(F.col("wind_dir_rad"))
    )
    .drop("wind_dir_rad")
)


# ============================================================
# 7. KIỂM TRA MISSING WEATHER
# ============================================================

print("\n================ WEATHER MISSING CHECK ================")

weather_columns = [
    "weather_air_temperature",
    "weather_dew_temperature",
    "weather_sea_level_pressure",
    "weather_wind_direction",
    "weather_wind_speed"
]

for column in weather_columns:

    missing_count = (
        df_result
        .filter(F.col(column).isNull())
        .count()
    )

    print(f"{column:20s}: {missing_count}")


# ============================================================
# 8. KIỂM TRA ROW COUNT
#
# LEFT JOIN đúng phải giữ nguyên:
# 315.792 rows
# ============================================================

total_rows = df_ml.count()
result_rows = df_result.count()

print("\n================ ROW CHECK ================")
print("Before join:", total_rows)
print("After join :", result_rows)


# ============================================================
# 9. KIỂM TRA MODEL-READY
#
# Việc join weather không được làm thay đổi trạng thái
# model_ready hiện tại.
# ============================================================

model_ready_before = (
    df_ml
    .filter(F.col("model_ready") == 1)
    .count()
)

model_ready_after = (
    df_result
    .filter(F.col("model_ready") == 1)
    .count()
)

print("\n================ MODEL READY CHECK ================")
print("Before join:", model_ready_before)
print("After join :", model_ready_after)


# ============================================================
# 10. SAMPLE
# ============================================================

print("\n================ SAMPLE ================")

(
    df_result
    .filter(F.col("model_ready") == 1)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "active_buildings",

        "weather_air_temperature",
        "weather_dew_temperature",
        "weather_sea_level_pressure",
        "weather_wind_direction",
        "weather_wind_speed",

        "wind_dir_sin",
        "wind_dir_cos",

        "hour",
        "day_of_week",
        "is_weekend",
        "is_outlier",
        "model_ready"
    )
    .orderBy("site_id", "timestamp")
    .show(20, truncate=False)
)


# ============================================================
# 11. LƯU
# ============================================================

(
    df_result
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


# ============================================================
# 12. KIỂM TRA SAU KHI GHI
# ============================================================

check_df = spark.read.parquet(OUTPUT_PATH)

check_rows = check_df.count()

check_ready = (
    check_df
    .filter(F.col("model_ready") == 1)
    .count()
)

print("\n================ FINAL CHECK ================")
print("Rows after save :", check_rows)
print("Model-ready rows:", check_ready)

if (
    check_rows == total_rows
    and check_ready == model_ready_before
):
    print("PASS: ML dataset with weather saved correctly.")
else:
    print("FAIL: ML dataset verification failed.")


# ============================================================
# 13. DỪNG SPARK
# ============================================================

spark.stop()