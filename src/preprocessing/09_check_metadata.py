from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
spark = (
    SparkSession.builder
    .appName("CheckMetadata")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)

electricity_path = (
    r"D:\BigData_Electricity\data\processed\electricity_cleaned"
)


# ============================================================
# 3. ĐỌC METADATA
# ============================================================
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
)


print(
    "\n================ THÔNG TIN METADATA ================\n"
)

print(
    f"Số dòng metadata: {metadata_df.count():,}"
)

print(
    f"Số cột metadata: {len(metadata_df.columns):,}"
)

print("\nCác cột metadata:")

for column in metadata_df.columns:
    print(f"- {column}")


# ============================================================
# 4. KIỂM TRA SCHEMA
# ============================================================
print(
    "\n================ SCHEMA METADATA ================\n"
)

metadata_df.printSchema()


# ============================================================
# 5. KIỂM TRA DUPLICATE BUILDING_ID
# ============================================================
duplicate_metadata = (
    metadata_df
    .groupBy("building_id")
    .count()
    .filter(F.col("count") > 1)
)


duplicate_metadata_count = duplicate_metadata.count()


print(
    "\n================ DUPLICATE BUILDING_ID ================\n"
)

print(
    f"Số building_id bị trùng trong metadata: "
    f"{duplicate_metadata_count:,}"
)

if duplicate_metadata_count > 0:

    duplicate_metadata.show(
        20,
        truncate=False
    )

else:

    print(
        "Không phát hiện building_id bị trùng."
    )


# ============================================================
# 6. ĐỌC DATASET ELECTRICITY ĐÃ LÀM SẠCH
# ============================================================
electricity_df = (
    spark.read
    .parquet(electricity_path)
)


# ============================================================
# 7. LẤY DANH SÁCH BUILDING TRONG ELECTRICITY
# ============================================================
electricity_buildings = (
    electricity_df
    .select("building_id")
    .distinct()
)


electricity_building_count = (
    electricity_buildings.count()
)


print(
    "\n================ BUILDING TRONG ELECTRICITY ================\n"
)

print(
    f"Số building trong electricity_cleaned: "
    f"{electricity_building_count:,}"
)


# ============================================================
# 8. BUILDING ĐIỆN CÓ TRONG METADATA
# ============================================================
matched_buildings = (
    electricity_buildings
    .join(
        metadata_df.select("building_id"),
        on="building_id",
        how="inner"
    )
)


matched_count = matched_buildings.count()


# ============================================================
# 9. BUILDING KHÔNG CÓ METADATA
# ============================================================
missing_metadata = (
    electricity_buildings
    .join(
        metadata_df.select("building_id"),
        on="building_id",
        how="left_anti"
    )
)


missing_metadata_count = (
    missing_metadata.count()
)


print(
    "\n================ KIỂM TRA GHÉP ELECTRICITY - METADATA ================\n"
)

print(
    f"Building có trong cả hai: "
    f"{matched_count:,}"
)

print(
    f"Building có electricity nhưng không có metadata: "
    f"{missing_metadata_count:,}"
)


if missing_metadata_count > 0:

    print(
        "\nDanh sách một số building không có metadata:"
    )

    missing_metadata.show(
        30,
        truncate=False
    )

else:

    print(
        "Tất cả building trong electricity đều có metadata."
    )


# ============================================================
# 10. KIỂM TRA METADATA BỊ THỪA
# ============================================================
# Một số building có thể xuất hiện trong metadata nhưng không
# có dữ liệu electricity.
#
# Trường hợp này không phải lỗi, nhưng cần biết để giải thích
# trong báo cáo.
metadata_without_electricity = (
    metadata_df
    .select("building_id")
    .join(
        electricity_buildings,
        on="building_id",
        how="left_anti"
    )
)


metadata_without_electricity_count = (
    metadata_without_electricity.count()
)


print(
    "\n================ METADATA KHÔNG CÓ ELECTRICITY ================\n"
)

print(
    f"Số building có metadata nhưng không có electricity: "
    f"{metadata_without_electricity_count:,}"
)


# ============================================================
# 11. KIỂM TRA MỘT SỐ DÒNG METADATA
# ============================================================
print(
    "\n================ SAMPLE METADATA ================\n"
)

metadata_df.show(
    10,
    truncate=False
)


# ============================================================
# 12. KIỂM TRA MISSING Ở CÁC TRƯỜNG METADATA QUAN TRỌNG
# ============================================================
important_columns = [
    "site_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "lat",
    "lng",
    "timezone",
    "yearbuilt",
    "numberoffloors",
    "occupants",
    "eui",
    "site_eui"
]


print(
    "\n================ MISSING Ở METADATA QUAN TRỌNG ================\n"
)

available_columns = [
    column
    for column in important_columns
    if column in metadata_df.columns
]


missing_expr = []

for column in available_columns:

    missing_expr.append(
        F.sum(
            F.when(
                F.col(column).isNull(),
                1
            ).otherwise(0)
        ).alias(column)
    )


metadata_missing_summary = (
    metadata_df
    .agg(*missing_expr)
)


metadata_missing_summary.show(
    truncate=False
)


# ============================================================
# 13. KẾT LUẬN
# ============================================================
print(
    "\n================ KẾT LUẬN ================\n"
)

if (
    duplicate_metadata_count == 0
    and missing_metadata_count == 0
):

    print(
        "Metadata có thể ghép với electricity_cleaned "
        "theo building_id."
    )

else:

    print(
        "Cần xử lý vấn đề metadata trước khi ghép."
    )


# ============================================================
# 14. KẾT THÚC
# ============================================================
spark.stop()