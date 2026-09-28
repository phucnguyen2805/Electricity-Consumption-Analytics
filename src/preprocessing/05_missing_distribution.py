from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Spark dùng để phân tích dữ liệu điện năng trên toàn bộ
# 1.578 building và 17.544 mốc thời gian.
spark = (
    SparkSession.builder
    .appName("MissingDistribution")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN FILE DỮ LIỆU
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(electricity_path)
)


# ============================================================
# 4. XÁC ĐỊNH CÁC CỘT BUILDING
# ============================================================
# Cột timestamp là thời gian.
# Tất cả cột còn lại tương ứng với một building.
meter_cols = [
    column
    for column in df.columns
    if column != "timestamp"
]


print("\n================ THÔNG TIN DỮ LIỆU ================\n")
print(f"Số building: {len(meter_cols):,}")
print(f"Số mốc thời gian: {df.count():,}")


# ============================================================
# 5. CHUẨN HÓA KIỂU DỮ LIỆU
# ============================================================
# Một số cột có thể được Spark nhận diện là StringType.
# try_cast giúp chuyển giá trị hợp lệ về DOUBLE.
#
# Giá trị không thể chuyển thành số sẽ trở thành NULL.
numeric_df = df.select(
    "timestamp",
    *[
        F.expr(f"`{column}`").cast("double").alias(column)
        for column in meter_cols
    ]
)


# ============================================================
# 6. CHUYỂN WIDE -> LONG
# ============================================================
# Kết quả:
# timestamp       building_id       consumption
#
# Cách này giúp Spark phân tích missing giống nhau cho
# tất cả building thay vì phải xử lý từng cột riêng lẻ.
long_df = numeric_df.unpivot(
    ids=["timestamp"],
    values=meter_cols,
    variableColumnName="building_id",
    valueColumnName="consumption"
)


# ============================================================
# 7. TÍNH MISSING RATE CHO TỪNG BUILDING
# ============================================================
building_stats = (
    long_df
    .groupBy("building_id")
    .agg(
        F.count("*").alias("total_rows"),

        F.sum(
            F.when(
                F.col("consumption").isNull()
                | F.isnan(F.col("consumption")),
                1
            ).otherwise(0)
        ).alias("missing_rows")
    )
    .withColumn(
        "missing_rate",
        F.col("missing_rows") / F.col("total_rows") * 100
    )
)


# ============================================================
# 8. PHÂN NHÓM TỶ LỆ MISSING
# ============================================================
# Các nhóm được chia theo mức độ thiếu dữ liệu.
#
# Mục đích:
# - 0%: hoàn toàn đầy đủ
# - >0% đến <=10%: thiếu ít
# - >10% đến <=30%: thiếu vừa
# - >30% đến <=50%: thiếu khá cao
# - >50% đến <=80%: thiếu cao
# - >80% đến <100%: gần như không có dữ liệu
# - 100%: hoàn toàn không có dữ liệu
building_distribution = (
    building_stats
    .withColumn(
        "missing_group",
        F.when(
            F.col("missing_rate") == 0,
            "0%"
        )
        .when(
            F.col("missing_rate") <= 10,
            ">0% - 10%"
        )
        .when(
            F.col("missing_rate") <= 30,
            ">10% - 30%"
        )
        .when(
            F.col("missing_rate") <= 50,
            ">30% - 50%"
        )
        .when(
            F.col("missing_rate") <= 80,
            ">50% - 80%"
        )
        .when(
            F.col("missing_rate") < 100,
            ">80% - <100%"
        )
        .otherwise("100%")
    )
)


# ============================================================
# 9. ĐẾM SỐ BUILDING TRONG TỪNG NHÓM
# ============================================================
distribution_summary = (
    building_distribution
    .groupBy("missing_group")
    .agg(
        F.count("*").alias("number_of_buildings"),
        F.sum("total_rows").alias("total_rows"),
        F.sum("missing_rows").alias("missing_rows")
    )
)


# ============================================================
# 10. SẮP XẾP NHÓM THEO THỨ TỰ LOGIC
# ============================================================
distribution_summary = (
    distribution_summary
    .withColumn(
        "sort_order",
        F.when(F.col("missing_group") == "0%", 1)
        .when(F.col("missing_group") == ">0% - 10%", 2)
        .when(F.col("missing_group") == ">10% - 30%", 3)
        .when(F.col("missing_group") == ">30% - 50%", 4)
        .when(F.col("missing_group") == ">50% - 80%", 5)
        .when(F.col("missing_group") == ">80% - <100%", 6)
        .when(F.col("missing_group") == "100%", 7)
    )
    .orderBy("sort_order")
    .drop("sort_order")
)


# ============================================================
# 11. HIỂN THỊ KẾT QUẢ
# ============================================================
print(
    "\n================ PHÂN BỐ BUILDING THEO TỶ LỆ MISSING ================\n"
)

distribution_summary.show(
    truncate=False
)


# ============================================================
# 12. KIỂM TRA CÁC BUILDING CÓ MISSING > 50%
# ============================================================
print(
    "\n================ BUILDING CÓ MISSING > 50% ================\n"
)

(
    building_stats
    .filter(F.col("missing_rate") > 50)
    .orderBy(F.desc("missing_rate"))
    .show(30, truncate=False)
)


# ============================================================
# 13. KẾT THÚC SPARK
# ============================================================
spark.stop()