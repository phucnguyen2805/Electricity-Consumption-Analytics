from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Dataset đầu vào có khoảng 25,9 triệu dòng.
#
# Điểm quan trọng:
# - Không groupBy các đặc trưng thời gian ngay từ đầu.
# - Không dùng countDistinct(building_id).
# - Không gọi count()/show() nhiều lần trước khi ghi.
#
# Chỉ aggregate theo:
#     site_id + timestamp
#
# Sau đó mới tạo các đặc trưng hour/month/day...
spark = (
    SparkSession.builder
    .appName("FeatureEngineeringElectricity")
    .config("spark.sql.shuffle.partitions", "32")
    .config(
        "spark.sql.adaptive.enabled",
        "true"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
input_path = (
    r"D:\BigData_Electricity\data\processed\electricity_final"
)

hourly_output_path = (
    r"D:\BigData_Electricity\data\processed\site_hourly_features"
)

daily_output_path = (
    r"D:\BigData_Electricity\data\processed\site_daily_features"
)


# ============================================================
# 3. ĐỌC DATASET FINAL
# ============================================================
# Chỉ đọc những cột thực sự cần thiết cho bước site-level.
df = (
    spark.read
    .parquet(input_path)
    .select(
        "site_id",
        "timestamp",
        "building_id",
        "consumption",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed"
    )
)


print(
    "\n================ DATASET ĐẦU VÀO ================\n"
)

print(
    f"Số building: "
    f"{df.select('building_id').distinct().count():,}"
)


# ============================================================
# 4. AGGREGATE THEO SITE + TIMESTAMP
# ============================================================
# Đây là bước nặng nhất nên cố ý làm đơn giản.
#
# Mỗi dòng electricity_cleaned tương ứng với một building
# tại một timestamp.
#
# Vì vậy:
#
#     count("*")
#
# chính là số building có dữ liệu tại site + timestamp.
#
# Không cần:
#
#     countDistinct("building_id")
#
# vì việc đó tốn bộ nhớ hơn.
#
# Weather của cùng site + timestamp là giống nhau cho tất cả
# building nên có thể lấy first().
print(
    "\n================ ĐANG AGGREGATE SITE-HOURLY ================\n"
)

site_hourly = (
    df
    .groupBy(
        "site_id",
        "timestamp"
    )
    .agg(
        F.sum(
            "consumption"
        ).alias(
            "total_consumption"
        ),

        F.count(
            "*"
        ).alias(
            "active_buildings"
        ),

        F.first(
            "airTemperature",
            ignorenulls=True
        ).alias(
            "airTemperature"
        ),

        F.first(
            "dewTemperature",
            ignorenulls=True
        ).alias(
            "dewTemperature"
        ),

        F.first(
            "seaLvlPressure",
            ignorenulls=True
        ).alias(
            "seaLvlPressure"
        ),

        F.first(
            "windDirection",
            ignorenulls=True
        ).alias(
            "windDirection"
        ),

        F.first(
            "windSpeed",
            ignorenulls=True
        ).alias(
            "windSpeed"
        )
    )
)


# ============================================================
# 5. TẠO ĐẶC TRƯNG THỜI GIAN SAU AGGREGATE
# ============================================================
# Lúc này dữ liệu chỉ còn tối đa:
#
#     19 site × 17.544 giờ
#     = 333.336 dòng
#
# nên tạo thêm nhiều cột thời gian rất nhẹ.
site_hourly = (
    site_hourly

    .withColumn(
        "year",
        F.year("timestamp")
    )

    .withColumn(
        "month",
        F.month("timestamp")
    )

    .withColumn(
        "day",
        F.dayofmonth("timestamp")
    )

    .withColumn(
        "hour",
        F.hour("timestamp")
    )

    .withColumn(
        "day_of_week",
        F.dayofweek("timestamp")
    )

    .withColumn(
        "quarter",
        F.quarter("timestamp")
    )
)


# ============================================================
# 6. TẠO BIẾN WEEKEND
# ============================================================
# Spark dayofweek:
#
# 1 = Chủ nhật
# 7 = Thứ bảy
#
# Hai giá trị này được xem là cuối tuần.
site_hourly = (
    site_hourly
    .withColumn(
        "is_weekend",
        F.when(
            F.col("day_of_week").isin(1, 7),
            1
        ).otherwise(0)
    )
)


# ============================================================
# 7. MÃ HÓA CHU KỲ CỦA GIỜ
# ============================================================
# Điện năng thường có chu kỳ 24 giờ.
#
# Dùng sin/cos giúp mô hình hiểu:
#
#     23 giờ gần 0 giờ
#
# thay vì xem chúng là hai giá trị rất xa nhau.
pi = 3.141592653589793

site_hourly = (
    site_hourly

    .withColumn(
        "hour_sin",
        F.sin(
            F.lit(2 * pi)
            * F.col("hour")
            / F.lit(24)
        )
    )

    .withColumn(
        "hour_cos",
        F.cos(
            F.lit(2 * pi)
            * F.col("hour")
            / F.lit(24)
        )
    )
)


# ============================================================
# 8. MÃ HÓA CHU KỲ CỦA THỨ
# ============================================================
site_hourly = (
    site_hourly

    .withColumn(
        "day_of_week_sin",
        F.sin(
            F.lit(2 * pi)
            * (F.col("day_of_week") - 1)
            / F.lit(7)
        )
    )

    .withColumn(
        "day_of_week_cos",
        F.cos(
            F.lit(2 * pi)
            * (F.col("day_of_week") - 1)
            / F.lit(7)
        )
    )
)


# ============================================================
# 9. SẮP XẾP LẠI CỘT
# ============================================================
site_hourly = (
    site_hourly
    .select(
        "site_id",
        "timestamp",

        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "is_weekend",
        "quarter",

        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",

        "total_consumption",
        "active_buildings",

        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed"
    )
)


# ============================================================
# 10. GHI SITE-HOURLY
# ============================================================
# Không gọi show() hoặc count() trước write để tránh Spark
# phải thực hiện lại phép aggregate lớn.
#
# Coalesce về 4 file giúp output không tạo quá nhiều file nhỏ.
print(
    "\n================ ĐANG LƯU SITE-HOURLY ================\n"
)

(
    site_hourly
    .coalesce(4)
    .write
    .mode("overwrite")
    .parquet(hourly_output_path)
)


# ============================================================
# 11. ĐỌC LẠI SITE-HOURLY
# ============================================================
# Sau khi write thành công, output chỉ còn khoảng 333k dòng.
# Những thao tác kiểm tra sau đây sẽ nhẹ.
hourly_verify = (
    spark.read
    .parquet(hourly_output_path)
)


hourly_count = (
    hourly_verify.count()
)

hourly_site_count = (
    hourly_verify
    .select("site_id")
    .distinct()
    .count()
)


print(
    "\n================ SITE-HOURLY ================\n"
)

print(
    f"Số bản ghi site-hourly: "
    f"{hourly_count:,}"
)

print(
    f"Số site: "
    f"{hourly_site_count:,}"
)


print(
    "\nSample:"
)

hourly_verify.show(
    20,
    truncate=False
)


# ============================================================
# 12. KIỂM TRA MISSING SITE-HOURLY
# ============================================================
hourly_check_columns = [
    "total_consumption",
    "active_buildings",
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windDirection",
    "windSpeed"
]

missing_expressions = []

for column in hourly_check_columns:

    missing_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull()
                | F.isnan(F.col(column)),
                1
            ).otherwise(0)
        ).alias(column)
    )


print(
    "\n================ MISSING SITE-HOURLY ================\n"
)

hourly_verify.agg(
    *missing_expressions
).show(
    truncate=False
)


# ============================================================
# 13. TẠO DATASET DAILY
# ============================================================
# Bây giờ dùng dataset site-hourly đã lưu.
#
# Đây là dataset nhỏ nên việc aggregate theo ngày nhẹ hơn rất
# nhiều so với việc aggregate trực tiếp 25,9 triệu dòng.
site_daily = (
    hourly_verify

    .withColumn(
        "date",
        F.to_date("timestamp")
    )

    .groupBy(
        "site_id",
        "date",
        "year",
        "month",
        "quarter",
        "is_weekend"
    )

    .agg(
        F.sum(
            "total_consumption"
        ).alias(
            "total_daily_consumption"
        ),

        F.avg(
            "total_consumption"
        ).alias(
            "avg_hourly_consumption"
        ),

        F.min(
            "total_consumption"
        ).alias(
            "min_hourly_consumption"
        ),

        F.max(
            "total_consumption"
        ).alias(
            "max_hourly_consumption"
        ),

        F.max(
            "active_buildings"
        ).alias(
            "active_buildings"
        ),

        F.avg(
            "airTemperature"
        ).alias(
            "avg_airTemperature"
        ),

        F.avg(
            "dewTemperature"
        ).alias(
            "avg_dewTemperature"
        ),

        F.avg(
            "seaLvlPressure"
        ).alias(
            "avg_seaLvlPressure"
        ),

        F.avg(
            "windDirection"
        ).alias(
            "avg_windDirection"
        ),

        F.avg(
            "windSpeed"
        ).alias(
            "avg_windSpeed"
        )
    )
)


# ============================================================
# 14. LƯU SITE-DAILY
# ============================================================
print(
    "\n================ ĐANG LƯU SITE-DAILY ================\n"
)

(
    site_daily
    .coalesce(4)
    .write
    .mode("overwrite")
    .parquet(daily_output_path)
)


# ============================================================
# 15. ĐỌC LẠI SITE-DAILY
# ============================================================
daily_verify = (
    spark.read
    .parquet(daily_output_path)
)


daily_count = (
    daily_verify.count()
)

daily_site_count = (
    daily_verify
    .select("site_id")
    .distinct()
    .count()
)


# ============================================================
# 16. HIỂN THỊ KẾT QUẢ
# ============================================================
print(
    "\n================ SITE-DAILY ================\n"
)

print(
    f"Số bản ghi site-daily: "
    f"{daily_count:,}"
)

print(
    f"Số site: "
    f"{daily_site_count:,}"
)

print(
    "\nSample:"
)

daily_verify.show(
    20,
    truncate=False
)


# ============================================================
# 17. KẾT QUẢ FEATURE ENGINEERING
# ============================================================
print(
    "\n================ KẾT QUẢ FEATURE ENGINEERING ================\n"
)

print(
    f"Site-hourly : "
    f"{hourly_count:,} dòng"
)

print(
    f"Site-daily  : "
    f"{daily_count:,} dòng"
)

print(
    f"\nSite-hourly output:\n{hourly_output_path}"
)

print(
    f"\nSite-daily output:\n{daily_output_path}"
)

print(
    "\n================ HOÀN TẤT ================\n"
)


# ============================================================
# 18. KẾT THÚC SPARK
# ============================================================
spark.stop()