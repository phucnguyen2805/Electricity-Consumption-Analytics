from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Spark được sử dụng để xử lý dữ liệu electricity quy mô lớn.
#
# Ở bước này ta sẽ:
#   1. Đọc dữ liệu gốc
#   2. Chuẩn hóa kiểu dữ liệu
#   3. Loại building có missing > 50%
#   4. Chuyển dữ liệu Wide -> Long
#   5. Loại các dòng consumption bị missing
#   6. Lưu dữ liệu sạch dưới dạng Parquet
spark = (
    SparkSession.builder
    .appName("CleanElectricityData")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
input_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)

output_path = (
    r"D:\BigData_Electricity\data\processed\electricity_cleaned"
)


# ============================================================
# 3. ĐỌC DỮ LIỆU GỐC
# ============================================================
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
)


# ============================================================
# 4. XÁC ĐỊNH CÁC CỘT BUILDING
# ============================================================
meter_cols = [
    column
    for column in df.columns
    if column != "timestamp"
]

print(
    "\n================ DỮ LIỆU GỐC ================\n"
)

print(
    f"Số building ban đầu: {len(meter_cols):,}"
)

print(
    f"Số timestamp: {df.count():,}"
)


# ============================================================
# 5. CHUẨN HÓA TOÀN BỘ BUILDING VỀ DOUBLE
# ============================================================
# Một số building trong file gốc có thể được Spark nhận diện
# là StringType.
#
# try_cast giúp:
#
#     "123.45" -> 123.45
#
# và giá trị không hợp lệ -> NULL.
#
# Điều này làm cho việc xác định missing nhất quán.
numeric_df = df.select(
    "timestamp",
    *[
        F.expr(
            f"try_cast(`{column}` AS DOUBLE)"
        ).alias(column)
        for column in meter_cols
    ]
)


# ============================================================
# 6. TÍNH SỐ MISSING CHO TỪNG BUILDING
# ============================================================
# Ta tính aggregate trên dạng Wide.
#
# Kết quả aggregate chỉ có 1 dòng nên có thể collect() an toàn.
missing_expressions = [
    F.sum(
        F.when(
            F.col(column).isNull()
            | F.isnan(F.col(column)),
            1
        ).otherwise(0)
    ).alias(column)
    for column in meter_cols
]


aggregate_row = (
    numeric_df
    .agg(
        F.count("*").alias("total_rows"),
        *missing_expressions
    )
    .collect()[0]
)


total_rows = aggregate_row["total_rows"]


# ============================================================
# 7. PHÂN LOẠI BUILDING
# ============================================================
# Tiêu chí đã được xác định từ các bước phân tích:
#
# missing <= 50%  -> GIỮ
# missing > 50%   -> LOẠI
valid_buildings = []
removed_buildings = []

missing_by_building = {}

for building in meter_cols:

    missing_rows = aggregate_row[building]

    if missing_rows is None:
        missing_rows = 0

    missing_by_building[building] = missing_rows

    missing_rate = (
        missing_rows
        / total_rows
        * 100
    )

    if missing_rate <= 50:

        valid_buildings.append(building)

    else:

        removed_buildings.append(building)


# ============================================================
# 8. TÍNH THỐNG KÊ SAU KHI LOẠI BUILDING
# ============================================================
total_rows_after_building_filter = (
    total_rows
    * len(valid_buildings)
)

missing_remaining = sum(
    missing_by_building[building]
    for building in valid_buildings
)

expected_clean_rows = (
    total_rows_after_building_filter
    - missing_remaining
)


print(
    "\n================ LỌC BUILDING ================\n"
)

print(
    f"Building được giữ lại: "
    f"{len(valid_buildings):,}"
)

print(
    f"Building bị loại: "
    f"{len(removed_buildings):,}"
)

print(
    f"Tổng bản ghi sau khi lọc building: "
    f"{total_rows_after_building_filter:,}"
)

print(
    f"Missing còn lại: "
    f"{missing_remaining:,}"
)

print(
    f"Bản ghi dự kiến sau khi loại missing: "
    f"{expected_clean_rows:,}"
)


# ============================================================
# 9. CHỈ GIỮ CÁC BUILDING HỢP LỆ
# ============================================================
# Việc select trước khi unpivot giúp giảm dữ liệu cần xử lý.
filtered_wide_df = numeric_df.select(
    "timestamp",
    *valid_buildings
)


# ============================================================
# 10. WIDE -> LONG
# ============================================================
# Dạng ban đầu:
#
# timestamp | building_A | building_B | ...
#
# Chuyển thành:
#
# timestamp | building_id | consumption
#
# Dạng LONG thuận tiện cho:
# - Spark SQL
# - phân tích theo building
# - phân tích theo thời gian
# - lưu vào InfluxDB
# - xây dựng mô hình dự báo
long_df = filtered_wide_df.unpivot(
    ids=["timestamp"],
    values=valid_buildings,
    variableColumnName="building_id",
    valueColumnName="consumption"
)


# ============================================================
# 11. LOẠI CÁC GIÁ TRỊ MISSING
# ============================================================
# Không thay missing bằng 0.
#
# Lý do:
# NULL nghĩa là không có phép đo.
# Nó không đồng nghĩa với tiêu thụ điện bằng 0.
#
# Những dòng có consumption NULL hoặc NaN được loại khỏi
# dataset phân tích.
cleaned_df = (
    long_df
    .filter(
        F.col("consumption").isNotNull()
        & ~F.isnan(F.col("consumption"))
    )
)


# ============================================================
# 12. KIỂM TRA SCHEMA
# ============================================================
print(
    "\n================ SCHEMA DATASET SẠCH ================\n"
)

cleaned_df.printSchema()


# ============================================================
# 13. HIỂN THỊ MỘT SỐ DÒNG
# ============================================================
print(
    "\n================ SAMPLE DATASET SẠCH ================\n"
)

cleaned_df.show(
    20,
    truncate=False
)


# ============================================================
# 14. LƯU DẠNG PARQUET
# ============================================================
# Parquet phù hợp cho Spark vì:
# - lưu dạng cột
# - đọc nhanh
# - giảm kích thước dữ liệu
# - Spark tối ưu tốt với Parquet
#
# mode("overwrite"):
# nếu thư mục đã tồn tại thì thay bằng dataset mới.
print(
    "\n================ ĐANG LƯU PARQUET ================\n"
)

(
    cleaned_df
    .write
    .mode("overwrite")
    .parquet(output_path)
)


# ============================================================
# 15. KIỂM TRA THƯ MỤC OUTPUT
# ============================================================
print(
    "\n================ KIỂM TRA OUTPUT ================\n"
)

if os.path.exists(output_path):

    files = os.listdir(output_path)

    print(
        f"Đã tạo thư mục: {output_path}"
    )

    print(
        f"Số file trong thư mục: {len(files):,}"
    )

else:

    print(
        "CẢNH BÁO: Không tìm thấy thư mục output."
    )


# ============================================================
# 16. THỐNG KÊ CUỐI CÙNG
# ============================================================
# Ta tính count sau khi lưu để xác minh transformation.
final_count = cleaned_df.count()

print(
    "\n================ KẾT QUẢ CUỐI CÙNG ================\n"
)

print(
    f"Bản ghi dự kiến: "
    f"{expected_clean_rows:,}"
)

print(
    f"Bản ghi thực tế: "
    f"{final_count:,}"
)

print(
    f"Bản ghi bị loại do missing: "
    f"{total_rows_after_building_filter - final_count:,}"
)

print(
    f"Building còn lại: "
    f"{len(valid_buildings):,}"
)

print(
    "\nDataset electricity_cleaned đã được tạo thành công."
)


# ============================================================
# 17. KẾT THÚC SPARK
# ============================================================
spark.stop()