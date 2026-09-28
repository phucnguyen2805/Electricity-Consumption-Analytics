from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    when,
    isnan,
    round
)


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# SparkSession là điểm bắt đầu để làm việc với Spark.
# local[*] cho phép Spark sử dụng các CPU core trên máy.
# ============================================================

spark = (
    SparkSession.builder
    .appName("BDG2-Missing-Analysis")
    .master("local[*]")
    .getOrCreate()
)


# ============================================================
# 2. ĐƯỜNG DẪN FILE ELECTRICITY RAW
# ============================================================

electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. ĐỌC DỮ LIỆU GỐC
# ============================================================
# header=True:
#   Dòng đầu tiên là tên cột.
#
# inferSchema=True:
#   Spark tự suy luận kiểu dữ liệu.
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
# Cột "timestamp" là cột thời gian.
# Những cột còn lại là building.
# ============================================================

building_columns = [
    column
    for column in df.columns
    if column != "timestamp"
]


# ============================================================
# 5. CHUẨN HÓA CÁC CỘT BUILDING VỀ DOUBLE
# ============================================================
# Một số cột trong dữ liệu gốc có thể bị Spark nhận thành
# string do dữ liệu không nhất quán.
#
# try_cast(... AS DOUBLE):
#   - số hợp lệ -> DOUBLE
#   - giá trị không chuyển được -> NULL
#
# Lưu ý:
# Đây chỉ là chuẩn hóa kiểu dữ liệu.
# CHƯA phải là bước xử lý missing cuối cùng.
# ============================================================

numeric_df = df.selectExpr(
    "`timestamp`",
    *[
        (
            f"try_cast(`{column.replace('`', '``')}` AS DOUBLE) "
            f"AS `{column.replace('`', '``')}`"
        )
        for column in building_columns
    ]
)


# ============================================================
# 6. CHUYỂN WIDE -> LONG
# ============================================================
# Sau bước này:
#
# timestamp | building_id | consumption
#
# Mỗi dòng đại diện cho một building tại một timestamp.
# ============================================================

long_df = numeric_df.unpivot(
    ids=["timestamp"],
    values=building_columns,
    variableColumnName="building_id",
    valueColumnName="consumption"
)


# ============================================================
# 7. TẠO CỘT ĐÁNH DẤU MISSING
# ============================================================
# Một giá trị được xem là missing nếu:
#
#   consumption IS NULL
#
# hoặc:
#
#   consumption là NaN
#
# NaN = Not a Number.
#
# Chúng ta KHÔNG coi 0.0 là missing.
# ============================================================

missing_condition = (
    col("consumption").isNull()
    | isnan(col("consumption"))
)


analysis_df = long_df.withColumn(
    "is_missing",
    when(missing_condition, 1).otherwise(0)
)


# ============================================================
# 8. TÍNH TỔNG SỐ BẢN GHI VÀ SỐ MISSING
# ============================================================
# count("*"):
#   Tổng số bản ghi.
#
# sum("is_missing"):
#   Tổng số bản ghi bị missing.
#
# Hai phép tính được thực hiện trong cùng một aggregation.
# ============================================================

summary = (
    analysis_df
    .agg(
        count("*").alias("total_rows"),
        spark_sum("is_missing").alias("missing_rows")
    )
    .collect()[0]
)


total_rows = summary["total_rows"]
missing_rows = summary["missing_rows"]


# ============================================================
# 9. TÍNH TỶ LỆ MISSING
# ============================================================
# Công thức:
#
# Missing Rate (%) =
#     Missing Rows / Total Rows * 100
#
# round(..., 4):
#   Làm tròn 4 chữ số thập phân để dễ đọc.
# ============================================================

missing_rate = (missing_rows / total_rows) * 100

print("================ TỔNG QUAN MISSING ================")

print(f"Tổng số bản ghi: {total_rows:,}")
print(f"Số bản ghi missing: {missing_rows:,}")
print(f"Tỷ lệ missing: {missing_rate:.4f} %")


# ============================================================
# 10. TÍNH MISSING THEO TỪNG BUILDING
# ============================================================
# Mục tiêu:
# biết building nào có nhiều dữ liệu thiếu nhất.
#
# count("*"):
#   Tổng số bản ghi của building.
#
# sum("is_missing"):
#   Số bản ghi missing của building.
# ============================================================

missing_by_building = (
    analysis_df
    .groupBy("building_id")
    .agg(
        count("*").alias("total_rows"),
        spark_sum("is_missing").alias("missing_rows")
    )
    .withColumn(
        "missing_rate",
        round(
            col("missing_rows") / col("total_rows") * 100,
            4
        )
    )
    .orderBy(
        col("missing_rows").desc()
    )
)


# ============================================================
# 11. HIỂN THỊ 20 BUILDING CÓ NHIỀU MISSING NHẤT
# ============================================================

print(
    "\n================ TOP 20 BUILDING CÓ NHIỀU MISSING ================\n"
)

missing_by_building.show(
    20,
    truncate=False
)


# ============================================================
# 12. KIỂM TRA MỘT SỐ BUILDING CÓ 0 MISSING
# ============================================================
# Đây chỉ là thông tin bổ sung.
# Nó giúp ta thấy mức độ phân bố missing giữa các building.
# ============================================================

print(
    "\n================ 10 BUILDING CÓ ÍT MISSING NHẤT ================\n"
)

(
    missing_by_building
    .orderBy(
        col("missing_rows").asc()
    )
    .show(
        10,
        truncate=False
    )
)


# ============================================================
# 13. ĐÓNG SPARK
# ============================================================

spark.stop()
