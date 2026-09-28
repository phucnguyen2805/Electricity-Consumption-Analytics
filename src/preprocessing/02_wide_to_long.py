from pyspark.sql import SparkSession


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# SparkSession là điểm bắt đầu để làm việc với Apache Spark.
#
# local[*]:
#   - Chạy Spark ngay trên máy hiện tại.
#   - "*" cho phép Spark sử dụng các CPU core có thể sử dụng.
#
# Giai đoạn phát triển đồ án chúng ta chạy local.
# ============================================================

spark = (
    SparkSession.builder
    .appName("BDG2-Wide-To-Long")
    .master("local[*]")
    .getOrCreate()
)


# ============================================================
# 2. KHAI BÁO ĐƯỜNG DẪN DỮ LIỆU
# ============================================================

electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. ĐỌC FILE ELECTRICITY
# ============================================================
# header=True:
#   Dòng đầu tiên của CSV chính là tên các cột.
#
# inferSchema=True:
#   Spark tự suy luận kiểu dữ liệu.
#
# Trong kết quả khám phá trước đó:
#   timestamp -> timestamp
#   phần lớn các building -> double
#
# Tuy nhiên đã phát hiện ít nhất một cột bị Spark nhận
# là string, vì vậy ở bước tiếp theo chúng ta sẽ chuẩn hóa
# kiểu dữ liệu trước khi unpivot.
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
# Cột đầu tiên là timestamp.
#
# Tất cả cột còn lại là các building trong electricity.csv.
#
# Ví dụ:
#   timestamp
#   Panther_parking_Lorriane
#   Panther_lodging_Cora
#   Panther_office_Hannah
#   ...
#
# Ta lưu danh sách các building vào biến meter_cols.
# ============================================================

meter_cols = [
    column
    for column in df.columns
    if column != "timestamp"
]


print("\n================ THÔNG TIN CỘT ================\n")

print("Số building columns:", len(meter_cols))

print("10 building đầu tiên:")
print(meter_cols[:10])


# ============================================================
# 5. CHUẨN HÓA TẤT CẢ BUILDING COLUMNS VỀ DOUBLE
# ============================================================
# Đây là bước rất quan trọng.
#
# Lý tưởng nhất:
#     tất cả giá trị consumption phải là số.
#
# Nhưng trong quá trình khám phá schema chúng ta đã phát hiện
# có cột bị Spark nhận thành STRING.
#
# Ví dụ:
#     Rat_public_Ulysses -> string
#
# Vì sau này chúng ta cần một cột consumption duy nhất,
# tất cả building columns phải có cùng kiểu dữ liệu.
#
# try_cast(... AS DOUBLE):
#     - Nếu giá trị có thể chuyển thành số -> chuyển thành số.
#     - Nếu giá trị không thể chuyển thành số -> NULL.
#
# Việc biến dữ liệu không hợp lệ thành NULL chưa phải là
# bước xử lý missing value cuối cùng.
#
# Ở bước này mục tiêu chính là:
#     chuẩn hóa kiểu dữ liệu để có thể unpivot.
# ============================================================

escaped_timestamp = "`timestamp`"

numeric_df = df.selectExpr(
    escaped_timestamp,
    *[
        (
            f"try_cast(`{column.replace('`', '``')}` AS DOUBLE) "
            f"AS `{column.replace('`', '``')}`"
        )
        for column in meter_cols
    ]
)


print("\n================ SCHEMA SAU KHI CHUẨN HÓA ================\n")

numeric_df.printSchema()


# ============================================================
# 6. CHUYỂN TỪ WIDE SANG LONG
# ============================================================
# Wide:
#
# timestamp | Building A | Building B | Building C
# ----------+------------+------------+------------
# 00:00     |    10      |    20      |    30
#
# Long:
#
# timestamp | building_id | consumption
# ----------+-------------+------------
# 00:00     | Building A  |    10
# 00:00     | Building B  |    20
# 00:00     | Building C  |    30
#
# unpivot() thực hiện đúng phép biến đổi này.
#
# ids:
#     Các cột được giữ nguyên.
#     Ở đây là timestamp.
#
# values:
#     Các cột sẽ được "bung" thành nhiều dòng.
#     Ở đây là tất cả building columns.
#
# variableColumnName:
#     Tên cột chứa tên building.
#
# valueColumnName:
#     Tên cột chứa giá trị consumption.
# ============================================================

long_df = numeric_df.unpivot(
    ids=["timestamp"],
    values=meter_cols,
    variableColumnName="building_id",
    valueColumnName="consumption"
)


# ============================================================
# 7. SẮP XẾP THỨ TỰ CỘT CHO DỄ ĐỌC
# ============================================================

long_df = long_df.select(
    "timestamp",
    "building_id",
    "consumption"
)


# ============================================================
# 8. KIỂM TRA SCHEMA
# ============================================================

print("\n================ SCHEMA LONG DATA ================\n")

long_df.printSchema()


# ============================================================
# 9. HIỂN THỊ MỘT SỐ DÒNG
# ============================================================
# truncate=False:
#   Không cắt ngắn nội dung của building_id.
# ============================================================

print("\n================ SAMPLE LONG DATA ================\n")

long_df.show(10, truncate=False)


# ============================================================
# 10. ĐẾM SỐ DÒNG SAU KHI UNPIVOT
# ============================================================
# Trước khi chuyển đổi:
#
#     17.544 timestamp
#
# Sau khi chuyển:
#
#     17.544 × số building
#
# Kết quả này sẽ giúp kiểm tra xem phép chuyển đổi có đúng
# hay không.
# ============================================================

long_row_count = long_df.count()

print("\n================ LONG ROW COUNT ================\n")

print("Số dòng sau Wide -> Long:", long_row_count)


# ============================================================
# 11. KIỂM TRA SỐ BUILDING KHÁC NHAU
# ============================================================
# distinct() giúp kiểm tra số building thực tế.
#
# Chúng ta chưa cần dùng groupBy phức tạp.
# ============================================================

building_count = long_df.select("building_id").distinct().count()

print("\n================ BUILDING COUNT ================\n")

print("Số building khác nhau:", building_count)


# ============================================================
# 12. ĐÓNG SPARK
# ============================================================

spark.stop()