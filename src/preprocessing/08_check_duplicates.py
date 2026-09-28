import csv

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Bước này sử dụng Spark để kiểm tra duplicate timestamp.
#
# Ta chỉ đọc cột timestamp cho phần kiểm tra này để tránh
# phải xử lý toàn bộ 1.578 cột building.
spark = (
    SparkSession.builder
    .appName("CheckDuplicates")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN FILE
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. KIỂM TRA TRÙNG TÊN BUILDING TRONG HEADER
# ============================================================
# Việc này được kiểm tra bằng Python vì header chỉ có 1.579
# phần tử nên không tốn nhiều bộ nhớ.
print(
    "\n================ KIỂM TRA TÊN BUILDING ================\n"
)

with open(
    electricity_path,
    mode="r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.reader(file)

    header = next(reader)

building_names = header[1:]

unique_buildings = set(building_names)

duplicate_building_names = [
    building
    for building, count in
    __import__("collections")
    .Counter(building_names)
    .items()
    if count > 1
]

print(
    f"Tổng số building columns: "
    f"{len(building_names):,}"
)

print(
    f"Số building_id khác nhau: "
    f"{len(unique_buildings):,}"
)

print(
    f"Số building_id bị trùng tên: "
    f"{len(duplicate_building_names):,}"
)

if duplicate_building_names:

    print("\nDanh sách building_id bị trùng:")

    for building in duplicate_building_names[:20]:
        print(f"- {building}")

else:

    print(
        "Kết luận: Không có building_id nào bị trùng tên."
    )


# ============================================================
# 4. ĐỌC CHỈ CỘT TIMESTAMP
# ============================================================
# inferSchema=False giúp đọc nhanh và tránh các vấn đề kiểu dữ liệu
# vì ở bước này ta chỉ cần kiểm tra timestamp.
#
# Spark có thể thực hiện column pruning, vì vậy phần xử lý chỉ
# tập trung vào cột timestamp cần thiết.
timestamp_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv(electricity_path)
    .select("timestamp")
)


# ============================================================
# 5. TỔNG SỐ TIMESTAMP
# ============================================================
total_timestamps = timestamp_df.count()


# ============================================================
# 6. SỐ TIMESTAMP KHÁC NHAU
# ============================================================
distinct_timestamps = (
    timestamp_df
    .select("timestamp")
    .distinct()
    .count()
)


# ============================================================
# 7. XÁC ĐỊNH TIMESTAMP BỊ TRÙNG
# ============================================================
# Nếu cùng một timestamp xuất hiện nhiều hơn một lần thì
# timestamp đó được xem là duplicate.
duplicate_timestamp_df = (
    timestamp_df
    .groupBy("timestamp")
    .count()
    .filter(F.col("count") > 1)
    .orderBy("timestamp")
)


duplicate_timestamp_count = (
    duplicate_timestamp_df.count()
)


# ============================================================
# 8. TỔNG SỐ DÒNG BỊ ẢNH HƯỞNG BỞI DUPLICATE
# ============================================================
# Ví dụ:
#
# timestamp A xuất hiện 2 lần
# -> có 2 dòng bị duplicate.
#
# timestamp B xuất hiện 3 lần
# -> có 3 dòng bị duplicate.
#
# Ta tính tổng số dòng thuộc các nhóm timestamp trùng.
duplicate_rows = (
    duplicate_timestamp_df
    .agg(
        F.sum("count").alias("duplicate_rows")
    )
    .collect()[0]["duplicate_rows"]
)

if duplicate_rows is None:
    duplicate_rows = 0


# ============================================================
# 9. HIỂN THỊ KẾT QUẢ
# ============================================================
print(
    "\n================ KIỂM TRA DUPLICATE TIMESTAMP ================\n"
)

print(
    f"Tổng số dòng timestamp: "
    f"{total_timestamps:,}"
)

print(
    f"Số timestamp khác nhau: "
    f"{distinct_timestamps:,}"
)

print(
    f"Số timestamp bị trùng: "
    f"{duplicate_timestamp_count:,}"
)

print(
    f"Tổng số dòng thuộc timestamp bị trùng: "
    f"{duplicate_rows:,}"
)


# ============================================================
# 10. HIỂN THỊ CÁC TIMESTAMP TRÙNG NẾU CÓ
# ============================================================
if duplicate_timestamp_count > 0:

    print(
        "\n================ DANH SÁCH TIMESTAMP BỊ TRÙNG ================\n"
    )

    duplicate_timestamp_df.show(
        30,
        truncate=False
    )

else:

    print(
        "\nKhông phát hiện timestamp bị trùng."
    )


# ============================================================
# 11. KẾT LUẬN
# ============================================================
print(
    "\n================ KẾT LUẬN ================\n"
)

if (
    len(duplicate_building_names) == 0
    and duplicate_timestamp_count == 0
):

    print(
        "Dữ liệu không phát hiện duplicate ở khóa building_id "
        "và timestamp."
    )

elif len(duplicate_building_names) > 0:

    print(
        "Có duplicate tên building_id cần xử lý."
    )

elif duplicate_timestamp_count > 0:

    print(
        "Có duplicate timestamp cần xử lý."
    )


# ============================================================
# 12. KẾT THÚC SPARK
# ============================================================
spark.stop()