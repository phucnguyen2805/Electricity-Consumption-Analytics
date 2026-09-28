from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Giảm số partition shuffle cho các phép aggregate nhỏ.
# Không cần đặt quá cao trên máy chạy Spark local.
spark = (
    SparkSession.builder
    .appName("JoinElectricityMetadataOptimized")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\data\processed\electricity_cleaned"
)

metadata_path = (
    r"D:\BigData_Electricity\bdg2\data\metadata\metadata.csv"
)

output_path = (
    r"D:\BigData_Electricity\data\processed\electricity_with_metadata"
)


# ============================================================
# 3. ĐỌC ELECTRICITY CLEANED
# ============================================================
electricity_df = (
    spark.read
    .parquet(electricity_path)
)

print(
    "\n================ ELECTRICITY CLEANED ================\n"
)

electricity_count = electricity_df.count()

electricity_buildings = (
    electricity_df
    .select("building_id")
    .distinct()
)

electricity_building_count = (
    electricity_buildings.count()
)

print(
    f"Số bản ghi: {electricity_count:,}"
)

print(
    f"Số building: {electricity_building_count:,}"
)


# ============================================================
# 4. ĐỌC METADATA
# ============================================================
metadata_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(metadata_path)
)


# ============================================================
# 5. CHỌN CÁC CỘT CẦN THIẾT
# ============================================================
metadata_selected = (
    metadata_df
    .select(
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "sqft",
        "lat",
        "lng",
        "timezone",
        "yearbuilt",
        "numberoffloors",
        "occupants",
        "eui",
        "site_eui",
        "source_eui",
        "leed_level",
        "rating"
    )
)


# ============================================================
# 6. CHUẨN HÓA CÁC TRƯỜNG SỐ
# ============================================================
# Một số trường EUI được đọc dưới dạng String.
# Chuyển sang DOUBLE để phục vụ phân tích về sau.
metadata_selected = (
    metadata_selected
    .withColumn(
        "eui",
        F.expr("try_cast(eui AS DOUBLE)")
    )
    .withColumn(
        "site_eui",
        F.expr("try_cast(site_eui AS DOUBLE)")
    )
    .withColumn(
        "source_eui",
        F.expr("try_cast(source_eui AS DOUBLE)")
    )
)


# ============================================================
# 7. KIỂM TRA DUPLICATE TRÊN METADATA
# ============================================================
# Metadata chỉ có 1.636 dòng nên phép kiểm tra này rất nhẹ.
metadata_duplicate_count = (
    metadata_selected
    .groupBy("building_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

print(
    "\n================ KIỂM TRA METADATA ================\n"
)

print(
    f"Building_id bị trùng trong metadata: "
    f"{metadata_duplicate_count:,}"
)


# ============================================================
# 8. CHỈ KIỂM TRA BUILDING THỰC SỰ CÓ ELECTRICITY
# ============================================================
# Đây là điểm tối ưu quan trọng.
#
# Không JOIN metadata với 25,9 triệu dòng.
#
# Ta chỉ lấy danh sách 1.514 building rồi ghép với metadata.
#
# Kết quả chỉ khoảng 1.514 dòng.
metadata_for_electricity = (
    electricity_buildings
    .join(
        F.broadcast(metadata_selected),
        on="building_id",
        how="left"
    )
)


metadata_check_count = (
    metadata_for_electricity.count()
)


print(
    "\n================ KIỂM TRA GHÉP BUILDING ================\n"
)

print(
    f"Building trong electricity: "
    f"{electricity_building_count:,}"
)

print(
    f"Building sau kiểm tra metadata: "
    f"{metadata_check_count:,}"
)


# ============================================================
# 9. KIỂM TRA BUILDING KHÔNG CÓ METADATA
# ============================================================
missing_metadata_buildings = (
    electricity_buildings
    .join(
        metadata_selected.select("building_id"),
        on="building_id",
        how="left_anti"
    )
)

missing_metadata_count = (
    missing_metadata_buildings.count()
)

print(
    f"Building có electricity nhưng không có metadata: "
    f"{missing_metadata_count:,}"
)


if missing_metadata_count == 0:

    print(
        "Tất cả building có electricity đều có metadata."
    )

else:

    print(
        "\nMột số building không có metadata:"
    )

    missing_metadata_buildings.show(
        20,
        truncate=False
    )


# ============================================================
# 10. KIỂM TRA BUILDING KHÔNG CÓ SITE_ID
# ============================================================
missing_site_id_count = (
    metadata_for_electricity
    .filter(F.col("site_id").isNull())
    .count()
)

print(
    "\n================ KIỂM TRA SITE_ID ================\n"
)

print(
    f"Building không có site_id: "
    f"{missing_site_id_count:,}"
)


# ============================================================
# 11. MISSING METADATA
# ============================================================
# Chỉ thực hiện trên 1.514 building metadata.
#
# Tuyệt đối không aggregate trên toàn bộ 25,9 triệu dòng
# electricity chỉ để kiểm tra metadata.
important_metadata_columns = [
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
    "site_eui",
    "source_eui"
]


missing_metadata_expressions = []

for column in important_metadata_columns:

    missing_metadata_expressions.append(
        F.sum(
            F.when(
                F.col(column).isNull(),
                1
            ).otherwise(0)
        ).alias(column)
    )


missing_metadata_summary = (
    metadata_for_electricity
    .agg(*missing_metadata_expressions)
)


print(
    "\n================ MISSING METADATA TRÊN 1.514 BUILDING ================\n"
)

missing_metadata_summary.show(
    truncate=False
)


# ============================================================
# 12. JOIN THỰC SỰ VỚI 25,9 TRIỆU BẢN GHI
# ============================================================
# Sau khi xác nhận metadata:
#
# - building_id không duplicate
# - tất cả 1.514 building đều có metadata
#
# ta mới thực hiện LEFT JOIN với electricity.
#
# Broadcast metadata vì bảng này chỉ có khoảng 1.636 dòng.
print(
    "\n================ ĐANG JOIN ELECTRICITY + METADATA ================\n"
)

joined_df = (
    electricity_df
    .join(
        F.broadcast(metadata_selected),
        on="building_id",
        how="left"
    )
)


# ============================================================
# 13. HIỂN THỊ MỘT SỐ DÒNG
# ============================================================
# Chỉ show 20 dòng để kiểm tra schema và dữ liệu.
print(
    "\n================ SAMPLE DATA SAU JOIN ================\n"
)

joined_df.show(
    20,
    truncate=False
)


# ============================================================
# 14. GHI RA PARQUET
# ============================================================
# Không gọi count() trên joined_df trước khi write.
#
# Nếu gọi count() rồi sau đó write(), Spark phải thực hiện
# lại toàn bộ JOIN thêm một lần nữa.
#
# Ở đây ta để phép JOIN được thực hiện trong lúc write.
print(
    "\n================ ĐANG LƯU DATASET ================\n"
)

(
    joined_df
    .write
    .mode("overwrite")
    .parquet(output_path)
)


# ============================================================
# 15. ĐỌC LẠI OUTPUT ĐỂ KIỂM TRA
# ============================================================
# Đọc lại Parquet giúp xác nhận dataset thực sự đã được tạo.
verification_df = (
    spark.read
    .parquet(output_path)
)


verification_count = (
    verification_df.count()
)


print(
    "\n================ XÁC NHẬN OUTPUT ================\n"
)

print(
    f"Bản ghi gốc: "
    f"{electricity_count:,}"
)

print(
    f"Bản ghi sau JOIN: "
    f"{verification_count:,}"
)


# ============================================================
# 16. KIỂM TRA KẾT QUẢ
# ============================================================
if verification_count == electricity_count:

    print(
        "Kiểm tra thành công: số bản ghi không thay đổi sau JOIN."
    )

else:

    print(
        "CẢNH BÁO: số bản ghi sau JOIN không khớp."
    )


print(
    "\n================ HOÀN TẤT ================\n"
)

print(
    f"Dataset đã lưu tại:\n{output_path}"
)


# ============================================================
# 17. KẾT THÚC SPARK
# ============================================================
spark.stop()