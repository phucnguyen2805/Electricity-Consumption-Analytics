import os
from pathlib import Path

import pyarrow.parquet as pq


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"D:\BigData_Electricity"
)


PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# ============================================================
# HÀM ĐẾM ROW PARQUET
#
# Không đọc toàn bộ dữ liệu vào RAM.
# Chỉ đọc metadata của từng file Parquet.
# ============================================================

def parquet_row_count(path: Path) -> int:

    if not path.exists():
        return -1

    total = 0

    parquet_files = list(
        path.rglob("*.parquet")
    )

    if not parquet_files:
        return -1

    for file_path in parquet_files:

        parquet_file = pq.ParquetFile(
            file_path
        )

        total += (
            parquet_file
            .metadata
            .num_rows
        )

    return total


# ============================================================
# HÀM KIỂM TRA DATASET
# ============================================================

def print_dataset(
    name: str,
    path: Path
):

    rows = parquet_row_count(path)

    print(
        f"{name:<32}: "
        f"{rows:,}"
        if rows >= 0
        else
        f"{name:<32}: NOT FOUND"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 65)
print(" ELECTRICITY CONSUMPTION ANALYTICS")
print(" FINAL PROJECT CHECK")
print("=" * 65)


# ============================================================
# 1. RAW DATA
#
# Raw electricity:
# 17,544 timestamps
# 1,578 buildings
# 27,684,432 expected long records
# ============================================================

print()
print("================ DATA PREPROCESSING ================")

print(
    "Raw electricity long records       : 27,684,432"
)

print(
    "Raw building columns               : 1,578"
)

print(
    "Raw timestamps                     : 17,544"
)

print(
    "Clean electricity records          :"
)

print_dataset(
    "electricity_final",
    PROCESSED_DIR / "electricity_final"
)


# ============================================================
# 2. SITE HOURLY
# ============================================================

print()
print("================ FEATURE ENGINEERING ================")

print_dataset(
    "site_hourly_features_final",
    PROCESSED_DIR
    / "site_hourly_features_final"
)


print_dataset(
    "ML lag dataset",
    PROCESSED_DIR
    / "site_hourly_lag_features"
)


print_dataset(
    "ML rolling/weather dataset",
    PROCESSED_DIR
    / "electricity_ml_dataset"
)


# ============================================================
# 3. TRAIN / TEST
# ============================================================

print()
print("================ TRAIN / TEST ================")

train_rows = parquet_row_count(
    PROCESSED_DIR
    / "ml_train_2016"
)

test_rows = parquet_row_count(
    PROCESSED_DIR
    / "ml_test_2017"
)


print(
    f"Train 2016 rows                  : "
    f"{train_rows:,}"
)

print(
    f"Test 2017 rows                   : "
    f"{test_rows:,}"
)

print(
    f"Total ML rows                    : "
    f"{train_rows + test_rows:,}"
    if train_rows >= 0 and test_rows >= 0
    else
    "Total ML rows                    : UNKNOWN"
)


# ============================================================
# 4. PREDICTIONS
# ============================================================

print()
print("================ MODEL OUTPUT ================")

print_dataset(
    "Linear Regression predictions",
    PROCESSED_DIR
    / "predictions_linear_baseline"
)

print_dataset(
    "Random Forest predictions",
    PROCESSED_DIR
    / "predictions_random_forest"
)

print_dataset(
    "Ridge predictions",
    PROCESSED_DIR
    / "predictions_ridge_regression"
)

print_dataset(
    "Final predictions",
    PROCESSED_DIR
    / "final_predictions"
)


# ============================================================
# 5. DASHBOARD DATASET
# ============================================================

print()
print("================ DASHBOARD DATA ================")

hourly_rows = parquet_row_count(
    PROCESSED_DIR
    / "dashboard_hourly"
)

daily_rows = parquet_row_count(
    PROCESSED_DIR
    / "dashboard_daily"
)

site_rows = parquet_row_count(
    PROCESSED_DIR
    / "dashboard_site"
)


print(
    f"Hourly dashboard rows             : "
    f"{hourly_rows:,}"
)

print(
    f"Daily dashboard rows              : "
    f"{daily_rows:,}"
)

print(
    f"Site dashboard rows               : "
    f"{site_rows:,}"
)


# ============================================================
# 6. INFLUXDB EXPECTED
# ============================================================

print()
print("================ INFLUXDB ================")

print(
    "Database                          : electricity"
)

print(
    "Measurement                       : electricity_forecast"
)

print(
    "Expected points                   : 153,016"
)

print(
    "Expected sites                    : 18"
)

print(
    "Expected min time                 : "
    "2017-01-01T00:00:00"
)

print(
    "Expected max time                 : "
    "2017-12-31T23:00:00"
)


# ============================================================
# 7. MODEL METRICS
# ============================================================

print()
print("================ MODEL RESULTS ================")

print(
    "Naive lag-24h"
)

print(
    "  RMSE = 2171.919018"
)

print(
    "  MAE  = 768.596259"
)

print(
    "  R2   = 0.958002"
)


print()
print(
    "Linear Regression"
)

print(
    "  RMSE = 1296.931754"
)

print(
    "  MAE  = 377.429933"
)

print(
    "  R2   = 0.985025"
)


print()
print(
    "Random Forest"
)

print(
    "  RMSE = 1357.076116"
)

print(
    "  MAE  = 537.137581"
)

print(
    "  R2   = 0.983604"
)


print()
print(
    "Ridge Regression"
)

print(
    "  regParam = 10"
)

print(
    "  RMSE = 1295.986751"
)

print(
    "  MAE  = 383.172457"
)

print(
    "  R2   = 0.985047"
)


# ============================================================
# 8. OUTLIER
# ============================================================

print()
print("================ OUTLIER ================")

print(
    "Test outlier rows                 : 2,549"
)

print(
    "Test outlier rate                 : "
    "≈ 1.67%"
)

print(
    "Ridge outlier MAE                 : "
    "1,473.61"
)

print(
    "Ridge normal MAE                  : "
    "364.70"
)

print(
    "Outlier SSE contribution          : "
    "60.84%"
)


# ============================================================
# 9. FINAL CHECK
# ============================================================

print()
print("================ FINAL CHECK ================")


expected_checks = {

    "Clean electricity":
        parquet_row_count(
            PROCESSED_DIR
            / "electricity_final"
        ) == 25949330,

    "Site hourly":
        parquet_row_count(
            PROCESSED_DIR
            / "site_hourly_features_final"
        ) == 313370,

    "Train":
        train_rows == 151654,

    "Test":
        test_rows == 153016,

    "Final predictions":
        parquet_row_count(
            PROCESSED_DIR
            / "final_predictions"
        ) == 153016,

    "Hourly dashboard":
        hourly_rows == 24,

    "Daily dashboard":
        daily_rows == 365,

    "Site dashboard":
        site_rows == 18
}


all_pass = True


for name, result in expected_checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{name:<32}: {status}"
    )

    if not result:
        all_pass = False


print()

if all_pass:

    print(
        "================================================="
    )

    print(
        " ALL LOCAL PROJECT CHECKS PASSED"
    )

    print(
        "================================================="
    )

else:

    print(
        "================================================="
    )

    print(
        " SOME PROJECT CHECKS FAILED"
    )

    print(
        "================================================="
    )