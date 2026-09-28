from pyspark.sql import SparkSession


# ============================================================
# 1. KHỞI TẠO SPARK SESSION
# ============================================================
# SparkSession là đối tượng chính để chúng ta làm việc với
# Spark DataFrame và Spark SQL.
#
# .appName(...) đặt tên cho ứng dụng Spark.
#
# .master("local[*]"):
#   - local: chạy Spark ngay trên máy hiện tại.
#   - *    : sử dụng các CPU core mà Spark có thể sử dụng.
#
# Giai đoạn phát triển chúng ta chạy local.
# Khi demo, điều này vẫn cho phép xử lý dataset lớn hơn
# so với cách dùng Pandas đọc toàn bộ dữ liệu vào RAM.
# ============================================================

spark = (
    SparkSession.builder
    .appName("BDG2-Electricity-Exploration")
    .master("local[*]")
    .getOrCreate()
)


# ============================================================
# 2. KHAI BÁO ĐƯỜNG DẪN FILE
# ============================================================

electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. ĐỌC FILE CSV BẰNG SPARK
# ============================================================
# header=True:
#   Dòng đầu tiên của file được sử dụng làm tên cột.
#
# inferSchema=True:
#   Spark tự suy đoán kiểu dữ liệu của các cột.
#
# Ví dụ:
#   timestamp  -> timestamp/string tùy dữ liệu
#   12.6024    -> double
# ============================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(electricity_path)
)


# ============================================================
# 4. IN SCHEMA
# ============================================================
# Schema cho chúng ta biết:
#   - tên cột
#   - kiểu dữ liệu
#   - nullable hay không
#
# Đây là bước rất quan trọng trước khi thực hiện cleaning.
# ============================================================

print("\n================ SCHEMA ================\n")

df.printSchema()


# ============================================================
# 5. ĐẾM SỐ DÒNG
# ============================================================
# count() yêu cầu Spark thực hiện một action trên DataFrame.
#
# Chúng ta muốn biết tổng số timestamp trong dataset.
# ============================================================

row_count = df.count()

print("\n================ ROW COUNT ================\n")
print("So dong:", row_count)


# ============================================================
# 6. ĐẾM SỐ CỘT
# ============================================================
# Electricity hiện đang ở dạng wide format:
#
# timestamp | meter_1 | meter_2 | meter_3 | ...
#
# Vì vậy số cột sẽ rất lớn.
# ============================================================

column_count = len(df.columns)

print("\n================ COLUMN COUNT ================\n")
print("So cot:", column_count)


# ============================================================
# 7. IN RA MỘT SỐ TÊN CỘT ĐẦU TIÊN
# ============================================================
# Không in toàn bộ hàng nghìn tên cột để tránh output quá dài.
# ============================================================

print("\n================ FIRST COLUMNS ================\n")

print(df.columns[:10])


# ============================================================
# 8. HIỂN THỊ 5 DÒNG ĐẦU
# ============================================================
# show(5) chỉ hiển thị 5 dòng để chúng ta quan sát dữ liệu.
# ============================================================

print("\n================ SAMPLE DATA ================\n")

df.show(5, truncate=100)


# ============================================================
# 9. ĐÓNG SPARK
# ============================================================
# Sau khi hoàn thành chương trình khám phá dữ liệu,
# giải phóng SparkSession.
# ============================================================

spark.stop()