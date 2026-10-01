import os
import shutil
from pathlib import Path

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
    / "electricity_forecast_utc.parquet"
)

CHUNK_DIR = (
    Path(r"D:\BigData_Electricity\data\processed")
    / "influx_chunks_utc"
)

CHUNK_SIZE = 10000

EXPECTED_ROWS = 153016

TOKEN = os.getenv(
    "INFLUXDB3_AUTH_TOKEN"
)


# ============================================================
# 2. CHECK TOKEN
# ============================================================

if not TOKEN:
    raise RuntimeError(
        "INFLUXDB3_AUTH_TOKEN is missing."
    )


# ============================================================
# 3. CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


# ============================================================
# 4. ĐỌC SOURCE
# ============================================================

print("\n================ READ SOURCE ================")

table = pq.read_table(
    INPUT_FILE
)

print("Rows:", table.num_rows)
print("Columns:", table.num_columns)

if table.num_rows != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"got {table.num_rows}"
    )


# ============================================================
# 5. XÓA CHUNK CŨ
# ============================================================

if CHUNK_DIR.exists():
    shutil.rmtree(CHUNK_DIR)

CHUNK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. CHIA CHUNK
# ============================================================

print("\n================ CREATE CHUNKS ================")

chunk_files = []

for start in range(
    0,
    table.num_rows,
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        table.num_rows
    )

    chunk_table = table.slice(
        start,
        end - start
    )

    chunk_file = (
        CHUNK_DIR
        / f"chunk_{len(chunk_files):04d}.parquet"
    )

    pq.write_table(
        chunk_table,
        chunk_file,
        compression="snappy"
    )

    chunk_files.append(
        chunk_file
    )


print(
    "Chunks created:",
    len(chunk_files)
)


# ============================================================
# 7. SYNCHRONOUS WRITE
# ============================================================

print(
    "\n================ WRITE TO INFLUXDB ================"
)

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
            f"    cumulative rows: "
            f"{written_rows}"
        )

finally:

    client.close()


# ============================================================
# 8. FINAL CHECK
# ============================================================

print(
    "\n================ WRITE RESULT ================"
)

print(
    "Expected rows:",
    EXPECTED_ROWS
)

print(
    "Written rows :",
    written_rows
)

if written_rows == EXPECTED_ROWS:
    print(
        "PASS: All UTC rows written to InfluxDB."
    )
else:
    print(
        "FAIL: Written row count does not match."
    )