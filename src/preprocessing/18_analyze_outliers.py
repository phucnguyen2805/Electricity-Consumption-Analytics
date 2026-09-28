from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. KHỞI TẠO SPARK
# ============================================================
# Chúng ta sử dụng dataset site-hourly đã được tổng hợp.
#
# Dataset này chỉ còn khoảng 313 nghìn dòng, nhỏ hơn rất nhiều
# so với dataset electricity gốc 25,9 triệu dòng.
spark = (
    SparkSession.builder
    .appName("AnalyzeElectricityOutliers")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================
input_path = (
    r"D:\BigData_Electricity\data\processed\site_hourly_features"
)


# ============================================================
# 3. ĐỌC DATASET SITE-HOURLY
# ============================================================
df = (
    spark.read
    .parquet(input_path)
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings"
    )
)


# ============================================================
# 4. THÔNG TIN TỔNG QUAN
# ============================================================
total_rows = df.count()

site_count = (
    df
    .select("site_id")
    .distinct()
    .count()
)


print(
    "\n================ DATASET OUTLIER ================\n"
)

print(
    f"Số bản ghi: {total_rows:,}"
)

print(
    f"Số site: {site_count:,}"
)


# ============================================================
# 5. KIỂM TRA GIÁ TRỊ ÂM
# ============================================================
# Trong bước này KHÔNG tự động xóa giá trị âm.
#
# Ta chỉ thống kê trước để biết dataset có xuất hiện
# consumption < 0 hay không.
negative_rows = (
    df
    .filter(
        F.col("total_consumption") < 0
    )
    .count()
)


print(
    "\n================ KIỂM TRA GIÁ TRỊ ÂM ================\n"
)

print(
    f"Số site-hour có total_consumption < 0: "
    f"{negative_rows:,}"
)


# ============================================================
# 6. KIỂM TRA GIÁ TRỊ BẰNG 0
# ============================================================
# 0 không được xem là missing.
#
# Ta cũng không xem 0 là outlier một cách tự động.
# Chỉ thống kê để hiểu đặc điểm dữ liệu.
zero_rows = (
    df
    .filter(
        F.col("total_consumption") == 0
    )
    .count()
)


zero_rate = (
    zero_rows
    / total_rows
    * 100
)


print(
    "\n================ KIỂM TRA GIÁ TRỊ ZERO ================\n"
)

print(
    f"Số site-hour có total_consumption = 0: "
    f"{zero_rows:,}"
)

print(
    f"Tỷ lệ zero: "
    f"{zero_rate:.4f}%"
)


# ============================================================
# 7. THỐNG KÊ MIN / MAX TOÀN BỘ
# ============================================================
global_stats = (
    df
    .agg(
        F.min(
            "total_consumption"
        ).alias("min_consumption"),

        F.max(
            "total_consumption"
        ).alias("max_consumption"),

        F.avg(
            "total_consumption"
        ).alias("avg_consumption")
    )
    .collect()[0]
)


print(
    "\n================ THỐNG KÊ TOÀN BỘ ================\n"
)

print(
    f"Min: {global_stats['min_consumption']}"
)

print(
    f"Max: {global_stats['max_consumption']}"
)

print(
    f"Average: {global_stats['avg_consumption']}"
)


# ============================================================
# 8. THỐNG KÊ THEO SITE
# ============================================================
# Các site có quy mô rất khác nhau.
#
# Ví dụ:
# một site có 5 building không nên dùng cùng một ngưỡng
# consumption với site có 280 building.
#
# Vì vậy outlier sẽ được xác định riêng cho từng site.
site_basic_stats = (
    df
    .groupBy("site_id")
    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "timestamp"
        ).alias(
            "timestamps"
        ),

        F.min(
            "total_consumption"
        ).alias(
            "min_consumption"
        ),

        F.max(
            "total_consumption"
        ).alias(
            "max_consumption"
        ),

        F.avg(
            "total_consumption"
        ).alias(
            "avg_consumption"
        )
    )
    .orderBy("site_id")
)


print(
    "\n================ MIN / MAX THEO SITE ================\n"
)

site_basic_stats.show(
    30,
    truncate=False
)


# ============================================================
# 9. TÍNH Q1, MEDIAN, Q3 THEO SITE
# ============================================================
# Dùng percentile_approx vì đây là cách phù hợp với Spark
# khi tính percentile trên dữ liệu phân tán.
#
# Q1  = 25th percentile
# Q2  = median
# Q3  = 75th percentile
#
# IQR = Q3 - Q1
site_percentiles = (
    df
    .groupBy("site_id")
    .agg(
        F.percentile_approx(
            "total_consumption",
            0.25,
            10000
        ).alias(
            "Q1"
        ),

        F.percentile_approx(
            "total_consumption",
            0.50,
            10000
        ).alias(
            "median"
        ),

        F.percentile_approx(
            "total_consumption",
            0.75,
            10000
        ).alias(
            "Q3"
        )
    )
    .withColumn(
        "IQR",
        F.col("Q3") - F.col("Q1")
    )
)


# ============================================================
# 10. TẠO LOWER / UPPER BOUND
# ============================================================
# Quy tắc IQR:
#
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#
# Một giá trị nằm ngoài khoảng này được đánh dấu
# là statistical outlier.
#
# Lưu ý:
# Đây chỉ là bước PHÁT HIỆN outlier.
# Chưa xóa hay thay thế bất kỳ dữ liệu nào.
site_bounds = (
    site_percentiles
    .withColumn(
        "lower_bound",
        F.col("Q1") - 1.5 * F.col("IQR")
    )
    .withColumn(
        "upper_bound",
        F.col("Q3") + 1.5 * F.col("IQR")
    )
)


# ============================================================
# 11. JOIN BOUND VÀO DATASET
# ============================================================
df_with_bounds = (
    df
    .join(
        F.broadcast(site_bounds),
        on="site_id",
        how="left"
    )
)


# ============================================================
# 12. ĐÁNH DẤU OUTLIER
# ============================================================
# Chỉ xem là outlier khi:
#
#     total_consumption < lower_bound
#     hoặc
#     total_consumption > upper_bound
#
# Không dùng điều kiện total_consumption == 0 vì 0 là giá trị
# hợp lệ và đã được phân biệt khỏi missing ở các bước trước.
df_outliers = (
    df_with_bounds
    .withColumn(
        "is_outlier",
        F.when(
            (
                F.col("total_consumption")
                < F.col("lower_bound")
            )
            |
            (
                F.col("total_consumption")
                > F.col("upper_bound")
            ),
            1
        )
        .otherwise(0)
    )
)


# ============================================================
# 13. THỐNG KÊ OUTLIER
# ============================================================
outlier_summary = (
    df_outliers
    .groupBy("site_id")
    .agg(
        F.count("*").alias(
            "total_rows"
        ),

        F.sum(
            "is_outlier"
        ).alias(
            "outlier_rows"
        )
    )
    .withColumn(
        "outlier_rate",
        F.col("outlier_rows")
        / F.col("total_rows")
        * 100
    )
    .orderBy(
        F.desc("outlier_rate")
    )
)


print(
    "\n================ OUTLIER THEO SITE ================\n"
)

outlier_summary.show(
    30,
    truncate=False
)


# ============================================================
# 14. TỔNG OUTLIER
# ============================================================
global_outlier = (
    df_outliers
    .agg(
        F.sum(
            "is_outlier"
        ).alias(
            "outlier_rows"
        )
    )
    .collect()[0]
)


total_outliers = (
    global_outlier["outlier_rows"]
)

if total_outliers is None:
    total_outliers = 0


global_outlier_rate = (
    total_outliers
    / total_rows
    * 100
)


print(
    "\n================ TỔNG OUTLIER ================\n"
)

print(
    f"Tổng số outlier: "
    f"{total_outliers:,}"
)

print(
    f"Tỷ lệ outlier: "
    f"{global_outlier_rate:.4f}%"
)


# ============================================================
# 15. 30 OUTLIER CAO NHẤT
# ============================================================
print(
    "\n================ 30 GIÁ TRỊ CAO NHẤT ================\n"
)

(
    df
    .orderBy(
        F.desc("total_consumption")
    )
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings"
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 16. 30 GIÁ TRỊ THẤP NHẤT
# ============================================================
print(
    "\n================ 30 GIÁ TRỊ THẤP NHẤT ================\n"
)

(
    df
    .orderBy(
        F.asc("total_consumption")
    )
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings"
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 17. 30 OUTLIER THEO IQR RÕ NHẤT
# ============================================================
print(
    "\n================ 30 OUTLIER THEO IQR ================\n"
)

(
    df_outliers
    .filter(
        F.col("is_outlier") == 1
    )
    .orderBy(
        F.desc("total_consumption")
    )
    .select(
        "site_id",
        "timestamp",
        "total_consumption",
        "active_buildings",
        "Q1",
        "median",
        "Q3",
        "IQR",
        "lower_bound",
        "upper_bound"
    )
    .show(
        30,
        truncate=False
    )
)


# ============================================================
# 18. PHÂN BỐ OUTLIER THEO QUY MÔ
# ============================================================
# Phân loại để dễ đánh giá:
#
# 0%           -> không có outlier
# >0 - 1%      -> rất ít
# >1 - 5%      -> vừa phải
# >5 - 10%     -> khá nhiều
# >10%         -> cần xem xét kỹ
outlier_distribution = (
    outlier_summary
    .withColumn(
        "outlier_group",
        F.when(
            F.col("outlier_rate") == 0,
            "0%"
        )
        .when(
            F.col("outlier_rate") <= 1,
            ">0% - 1%"
        )
        .when(
            F.col("outlier_rate") <= 5,
            ">1% - 5%"
        )
        .when(
            F.col("outlier_rate") <= 10,
            ">5% - 10%"
        )
        .otherwise(
            ">10%"
        )
    )
    .groupBy("outlier_group")
    .agg(
        F.count("*").alias(
            "number_of_sites"
        )
    )
)


print(
    "\n================ PHÂN BỐ SITE THEO OUTLIER ================\n"
)

outlier_distribution.show(
    truncate=False
)


# ============================================================
# 19. KẾT THÚC
# ============================================================
print(
    "\n================ HOÀN TẤT ================\n"
)

print(
    "Đã hoàn thành phân tích outlier."
)

print(
    "Chưa xóa hoặc thay đổi bất kỳ giá trị outlier nào."
)


spark.stop()