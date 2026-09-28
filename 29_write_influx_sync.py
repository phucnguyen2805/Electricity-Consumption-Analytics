import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from influxdb_client_3 import InfluxDBClient3


# ============================================================
# 1. CONFIG
# ============================================================

HOST = "http://127.0.0.1:8181"

DATABASE = "electricity"

MEASUREMENT = "electricity_forecast"

INPUT_FILE = (
    Path(r"D:\BigData_Electricity\data\processed")
    / "influx_export"
    / "electricity_forecast.parquet"
)

CHUNK_DIR = (
    Path(r"D:\BigData_Electricity\data\processed")
    / "influx_chunks"
)

CHUNK_SIZE = 500

EXPECTED_ROWS = 153016

TOKEN = os.getenv("INFLUXDB3_AUTH_TOKEN")


# ============================================================
# 2. KIỂM TRA
# ============================================================

if not TOKEN:
    raise RuntimeError(
        "INFLUXDB3_AUTH_TOKEN is missing."
    )

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


# ============================================================
# 3. XÓA CHUNK CŨ
#
# Script có thể được chạy lại.
# Chunk tạm cũ sẽ được tạo lại từ file nguồn chuẩn.
# ============================================================

if CHUNK_DIR.exists():
    shutil.rmtree(CHUNK_DIR)

CHUNK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\n================ CONFIG ================")
print("Host       :", HOST)
print("Database   :", DATABASE)
print("Measurement:", MEASUREMENT)
print("Input      :", INPUT_FILE)
print("Chunk size :", CHUNK_SIZE)


# ============================================================
# 4. ĐỌC PARQUET
# ============================================================

print("\n================ READ SOURCE ================")

table = pq.read_table(
    INPUT_FILE
)

print("Rows:", table.num_rows)
print("Cols:", table.num_columns)

if table.num_rows != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"but source contains {table.num_rows}"
    )


# ============================================================
# 5. CHIA THÀNH CÁC FILE NHỎ
# ============================================================

print("\n================ CREATE CHUNKS ================")

chunk_files = []

total_rows = table.num_rows

chunk_number = 0

for start in range(
    0,
    total_rows,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        total_rows
    )

    chunk_table = table.slice(
        start,
        end - start
    )

    chunk_file = (
        CHUNK_DIR /
        f"chunk_{chunk_number:04d}.parquet"
    )

    pq.write_table(
        chunk_table,
        chunk_file,
        compression="snappy"
    )

    chunk_files.append(
        chunk_file
    )

    chunk_number += 1


print(
    "Chunks created:",
    len(chunk_files)
)

print(
    "Expected chunks:",
    (
        total_rows + CHUNK_SIZE - 1
    ) // CHUNK_SIZE
)


# ============================================================
# 6. SYNCHRONOUS WRITE
#
# Không sử dụng WriteOptions batch mode.
#
# Mỗi write_file():
#   - đọc đúng 1 chunk
#   - gửi request ngay
#   - hoàn tất mới chuyển sang chunk tiếp theo.
#
# Điều này tránh queue batch và lỗi KeyboardInterrupt
# lúc client.close().
# ============================================================

print("\n================ WRITING TO INFLUXDB ================")

client = InfluxDBClient3(
    host=HOST,
    database=DATABASE,
    token=TOKEN
)


written_rows = 0

try:

    for index, chunk_file in enumerate(
        chunk_files,
        start=1
    ):

        chunk_rows = pq.read_metadata(
            chunk_file
        ).num_rows

        print(
            f"[{index}/{len(chunk_files)}] "
            f"Writing {chunk_rows} rows..."
        )

        client.write_file(
            file=str(chunk_file),
            measurement_name=MEASUREMENT,
            timestamp_column="timestamp",
            tag_columns=[
                "site_id",
                "is_outlier"
            ]
        )

        written_rows += chunk_rows

        print(
            f"    OK - cumulative rows: "
            f"{written_rows}"
        )


finally:

    # --------------------------------------------------------
    # Trong synchronous mode không còn batch queue cần flush.
    # close() chỉ giải phóng client.
    # --------------------------------------------------------

    client.close()


print("\n================ WRITE RESULT ================")

print(
    "Expected rows:",
    EXPECTED_ROWS
)

print(
    "Written rows  :",
    written_rows
)


if written_rows == EXPECTED_ROWS:
    print(
        "PASS: All source rows were sent to InfluxDB."
    )
else:
    print(
        "FAIL: Written row count does not match source."
    )