import os

import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

INPUT_PATH = (
    r"D:\BigData_Electricity\data\processed"
    r"\final_predictions"
)

OUTPUT_DIR = (
    r"D:\BigData_Electricity\data\processed"
    r"\influx_export"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "electricity_forecast.parquet"
)


# ============================================================
# 2. TẠO THƯ MỤC OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. ĐỌC SPARK PARQUET DATASET
# ============================================================

print("\n================ READ PARQUET DATASET ================")

dataset = ds.dataset(
    INPUT_PATH,
    format="parquet"
)

table = dataset.to_table()

print("Rows :", table.num_rows)
print("Cols :", table.num_columns)

print("Columns:")
for name in table.column_names:
    print(" -", name)


# ============================================================
# 4. KIỂM TRA SỐ DÒNG
# ============================================================

expected_rows = 153016

if table.num_rows != expected_rows:
    raise ValueError(
        f"Unexpected row count: {table.num_rows}, "
        f"expected {expected_rows}"
    )


# ============================================================
# 5. GHI THÀNH MỘT FILE PARQUET
# ============================================================

print("\n================ WRITE PARQUET FILE ================")

pq.write_table(
    table,
    OUTPUT_FILE,
    compression="snappy"
)

print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# 6. KIỂM TRA LẠI
# ============================================================

check = pq.read_table(
    OUTPUT_FILE
)

print("\n================ FINAL CHECK ================")

print("Rows:", check.num_rows)
print("Columns:", check.num_columns)

if (
    check.num_rows == expected_rows
    and check.column_names == table.column_names
):
    print(
        "PASS: InfluxDB export Parquet created successfully."
    )
else:
    print(
        "FAIL: Export file verification failed."
    )