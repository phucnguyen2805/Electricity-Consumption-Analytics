from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Spark được sử dụng để đọc và kiểm tra trực tiếp dữ liệu CSV gốc.
# Ta chưa thực hiện làm sạch hay thay đổi dữ liệu ở bước này.
spark = (
    SparkSession.builder
    .appName("CheckFullMissingBuildings")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN FILE ELECTRICITY GỐC
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 3. ĐỌC DỮ LIỆU GỐC
# ============================================================
# Không dùng inferSchema=True ở bước này.
#
# Mục đích:
# - Giữ nguyên giá trị gốc trong CSV.
# - Đặc biệt quan trọng đối với Rat_public_Ulysses,
#   vì trước đó Spark đã nhận diện cột này là String.
#
# Tất cả dữ liệu sẽ được đọc dưới dạng String để quan sát
# chính xác giá trị đang tồn tại trong file gốc.
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv(electricity_path)
)


# ============================================================
# 4. HAI BUILDING CÓ MISSING 100%
# ============================================================
target_buildings = [
    "Eagle_lodging_Garland",
    "Rat_public_Ulysses"
]


# ============================================================
# 5. KIỂM TRA SCHEMA CỦA 2 BUILDING
# ============================================================
print("\n================ SCHEMA CỦA 2 BUILDING ================\n")

for building in target_buildings:
    print(
        f"{building}: "
        f"{df.schema[building].dataType}"
    )


# ============================================================
# 6. KIỂM TRA SỐ GIÁ TRỊ KHÁC NULL
# ============================================================
print("\n================ KIỂM TRA GIÁ TRỊ ================\n")

for building in target_buildings:

    total_rows = df.count()

    non_null_rows = (
        df.filter(
            F.col(building).isNotNull()
            & (F.trim(F.col(building)) != "")
        )
        .count()
    )

    empty_rows = (
        df.filter(
            F.col(building).isNull()
            | (F.trim(F.col(building)) == "")
        )
        .count()
    )

    print(f"\nBuilding: {building}")
    print(f"Tổng số dòng: {total_rows:,}")
    print(f"Giá trị có dữ liệu: {non_null_rows:,}")
    print(f"Giá trị NULL/rỗng: {empty_rows:,}")


# ============================================================
# 7. HIỂN THỊ MỘT SỐ GIÁ TRỊ GỐC
# ============================================================
print("\n================ MỘT SỐ GIÁ TRỊ GỐC ================\n")

for building in target_buildings:

    print(f"\n--- {building} ---")

    (
        df.select(
            "timestamp",
            building
        )
        .where(
            F.col(building).isNotNull()
            & (F.trim(F.col(building)) != "")
        )
        .limit(20)
        .show(
            20,
            truncate=False
        )
    )


# ============================================================
# 8. KIỂM TRA CÁC CHUỖI KHÔNG PHẢI SỐ
# ============================================================
# Với building Rat_public_Ulysses, đây là phần đặc biệt quan trọng.
#
# Nếu dữ liệu tồn tại nhưng chứa các giá trị như:
# - "NA"
# - "null"
# - "None"
# - ký hiệu khác
#
# thì try_cast(... AS DOUBLE) sẽ biến chúng thành NULL.
#
# Ta sẽ lấy các giá trị khác rỗng và hiển thị để xác định
# nguyên nhân thực sự.

print("\n================ KIỂM TRA GIÁ TRỊ KHÔNG RỖNG ================\n")

for building in target_buildings:

    print(f"\n--- {building} ---")

    (
        df.select(
            F.col(building).alias("raw_value")
        )
        .where(
            F.col(building).isNotNull()
            & (F.trim(F.col(building)) != "")
        )
        .groupBy("raw_value")
        .count()
        .orderBy(F.desc("count"))
        .limit(20)
        .show(
            20,
            truncate=False
        )
    )


# ============================================================
# 9. KẾT THÚC SPARK
# ============================================================
spark.stop()