import csv
import math
import heapq

from collections import Counter
from datetime import datetime, timedelta


# ============================================================
# 1. ĐƯỜNG DẪN FILE DỮ LIỆU
# ============================================================
electricity_path = (
    r"D:\BigData_Electricity\bdg2\data\meters\raw\electricity.csv"
)


# ============================================================
# 2. CÁC GIÁ TRỊ ĐƯỢC XEM LÀ MISSING
# ============================================================
# File CSV gốc có thể biểu diễn missing dưới dạng:
# - ô trống
# - null
# - none
# - na
# - nan
#
# Ngoài ra, nếu một giá trị không thể chuyển thành số,
# ta cũng xem đó là dữ liệu không hợp lệ/missing.
MISSING_TEXT_VALUES = {
    "",
    "null",
    "none",
    "na",
    "nan",
    "n/a",
}


def is_missing(value):
    """
    Xác định một ô dữ liệu có phải missing hay không.

    Hàm này làm việc trực tiếp với chuỗi đọc từ CSV.
    """

    if value is None:
        return True

    value = value.strip()

    if value.lower() in MISSING_TEXT_VALUES:
        return True

    try:
        number = float(value)

        if math.isnan(number):
            return True

    except ValueError:
        # Nếu không thể chuyển thành số thì xem là dữ liệu
        # không hợp lệ và xử lý tương tự missing.
        return True

    return False


def parse_timestamp(value):
    """
    Chuyển timestamp trong CSV thành datetime.

    Hàm hỗ trợ một số dạng timestamp phổ biến:
    - 2016-01-01 00:00:00
    - 2016-01-01T00:00:00
    """

    value = value.strip()

    try:
        return datetime.fromisoformat(value)

    except ValueError:
        # Một số dữ liệu có thể dùng khoảng trắng thay vì T.
        return datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S"
        )


# ============================================================
# 3. PASS 1 - TÍNH MISSING CHO TỪNG BUILDING
# ============================================================
#
# Quan trọng:
#
# Không dùng Spark.
# Không unpivot.
# Không Window.
# Không collect() toàn bộ dữ liệu.
#
# Python chỉ đọc từng dòng CSV một.
#
# Vì vậy bộ nhớ sử dụng chủ yếu phụ thuộc vào 1 dòng dữ liệu
# thay vì 26+ triệu dòng.
print(
    "\n================ PASS 1: PHÂN TÍCH MISSING ================\n"
)

missing_counts = []
header = None
total_rows = 0

with open(
    electricity_path,
    mode="r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.reader(file)

    # Đọc header.
    header = next(reader)

    # Tất cả cột sau timestamp là building.
    building_names = header[1:]

    # Khởi tạo bộ đếm missing cho từng building.
    missing_counts = [
        0 for _ in building_names
    ]

    for row in reader:

        # Bỏ qua dòng lỗi nếu số cột không đúng.
        if len(row) != len(header):
            continue

        total_rows += 1

        # Bắt đầu từ index 1 vì index 0 là timestamp.
        for i, value in enumerate(row[1:]):

            if is_missing(value):
                missing_counts[i] += 1


# ============================================================
# 4. XÁC ĐỊNH BUILDING GIỮ LẠI
# ============================================================
#
# Quy tắc đã xác định từ bước phân tích trước:
#
# missing <= 50%  -> giữ
# missing > 50%   -> loại
#
# Tỷ lệ được tính dựa trên tổng 17.544 timestamp.
valid_buildings = []
removed_buildings = []

valid_indices = []

for i, building in enumerate(building_names):

    missing_rate = (
        missing_counts[i]
        / total_rows
        * 100
    )

    if missing_rate <= 50:
        valid_buildings.append(building)
        valid_indices.append(i + 1)

    else:
        removed_buildings.append(building)


total_missing_remaining = sum(
    missing_counts[i]
    for i in range(len(building_names))
    if i + 1 in valid_indices
)


print(
    "================ SAU KHI LOẠI BUILDING MISSING > 50% ================\n"
)

print(
    f"Số mốc thời gian: {total_rows:,}"
)

print(
    f"Số building ban đầu: {len(building_names):,}"
)

print(
    f"Số building được giữ lại: {len(valid_buildings):,}"
)

print(
    f"Số building bị loại: {len(removed_buildings):,}"
)

print(
    f"Tổng số bản ghi sau khi lọc: "
    f"{len(valid_buildings) * total_rows:,}"
)

print(
    f"Tổng số missing còn lại: "
    f"{total_missing_remaining:,}"
)


# ============================================================
# 5. PASS 2 - KIỂM TRA TIMESTAMP + MISSING GAP
# ============================================================
#
# Ta đọc file lại lần thứ hai.
#
# Chỉ xử lý các building có missing <= 50%.
#
# Không lưu toàn bộ dữ liệu vào RAM.
print(
    "\n================ PASS 2: PHÂN TÍCH GAP ================\n"
)

valid_index_set = set(valid_indices)

# Trạng thái gap hiện tại của từng building.
current_gap = {
    building: 0
    for building in valid_buildings
}

gap_start_time = {
    building: None
    for building in valid_buildings
}

gap_last_missing_time = {
    building: None
    for building in valid_buildings
}


# Tổng số gap của từng building.
building_gap_count = {
    building: 0
    for building in valid_buildings
}


# Gap dài nhất của từng building.
building_max_gap = {
    building: 0
    for building in valid_buildings
}


# Phân bố số gap theo độ dài.
global_gap_distribution = Counter()


# Chỉ giữ lại 30 gap dài nhất.
#
# Mỗi phần tử:
# (
#     gap_length,
#     building_id,
#     start_time,
#     end_time
# )
top_gaps = []


# ============================================================
# 6. HÀM GHI NHẬN GAP
# ============================================================
def register_gap(
    building,
    gap_length,
    start_time,
    end_time
):
    """
    Lưu thông tin một đoạn missing liên tiếp.
    """

    if gap_length <= 0:
        return

    global_gap_distribution[gap_length] += 1

    building_gap_count[building] += 1

    if gap_length > building_max_gap[building]:
        building_max_gap[building] = gap_length

    item = (
        gap_length,
        building,
        start_time,
        end_time
    )

    # Chỉ giữ 30 gap dài nhất.
    if len(top_gaps) < 30:

        heapq.heappush(
            top_gaps,
            item
        )

    elif gap_length > top_gaps[0][0]:

        heapq.heapreplace(
            top_gaps,
            item
        )


# ============================================================
# 7. KIỂM TRA TÍNH LIÊN TỤC CỦA TIMESTAMP
# ============================================================
timestamp_count = 0

first_timestamp = None
last_timestamp = None

non_hour_intervals = 0
sample_non_hour_intervals = []


# ============================================================
# 8. ĐỌC CSV THEO TỪNG DÒNG
# ============================================================
with open(
    electricity_path,
    mode="r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.reader(file)

    # Bỏ header.
    next(reader)

    previous_timestamp = None

    for row in reader:

        if len(row) != len(header):
            continue

        timestamp = parse_timestamp(row[0])

        timestamp_count += 1

        if first_timestamp is None:
            first_timestamp = timestamp

        last_timestamp = timestamp


        # ----------------------------------------------------
        # KIỂM TRA INTERVAL GIỮA HAI TIMESTAMP
        # ----------------------------------------------------
        if previous_timestamp is not None:

            interval = (
                timestamp - previous_timestamp
            )

            if interval != timedelta(hours=1):

                non_hour_intervals += 1

                if len(sample_non_hour_intervals) < 10:

                    sample_non_hour_intervals.append(
                        (
                            previous_timestamp,
                            timestamp,
                            interval
                        )
                    )

        previous_timestamp = timestamp


        # ----------------------------------------------------
        # KIỂM TRA MISSING CHO CÁC BUILDING HỢP LỆ
        # ----------------------------------------------------
        for index in valid_indices:

            building_position = index - 1
            building = building_names[building_position]

            value = row[index]

            if is_missing(value):

                # Nếu đây là missing đầu tiên của một gap,
                # lưu lại thời điểm bắt đầu.
                if current_gap[building] == 0:

                    gap_start_time[building] = timestamp

                current_gap[building] += 1

                gap_last_missing_time[building] = timestamp

            else:

                # Nếu trước đó có một đoạn missing,
                # thì đoạn đó vừa kết thúc.
                if current_gap[building] > 0:

                    register_gap(
                        building=building,
                        gap_length=current_gap[building],
                        start_time=gap_start_time[building],
                        end_time=gap_last_missing_time[building]
                    )

                    current_gap[building] = 0
                    gap_start_time[building] = None
                    gap_last_missing_time[building] = None


# ============================================================
# 9. XỬ LÝ GAP KẾT THÚC Ở CUỐI FILE
# ============================================================
for building in valid_buildings:

    if current_gap[building] > 0:

        register_gap(
            building=building,
            gap_length=current_gap[building],
            start_time=gap_start_time[building],
            end_time=gap_last_missing_time[building]
        )


# ============================================================
# 10. KẾT QUẢ TIMESTAMP
# ============================================================
print(
    "\n================ KIỂM TRA TIMESTAMP ================\n"
)

print(
    f"Số timestamp: {timestamp_count:,}"
)

print(
    f"Timestamp đầu tiên: {first_timestamp}"
)

print(
    f"Timestamp cuối cùng: {last_timestamp}"
)

print(
    f"Số khoảng thời gian khác 1 giờ: "
    f"{non_hour_intervals:,}"
)

if non_hour_intervals == 0:

    print(
        "Kết luận: Các timestamp liên tiếp cách nhau đúng 1 giờ."
    )

else:

    print(
        "Có timestamp không cách nhau đúng 1 giờ."
    )

    print(
        "\nMột số ví dụ:"
    )

    for (
        previous_ts,
        current_ts,
        interval
    ) in sample_non_hour_intervals:

        print(
            f"{previous_ts} -> {current_ts} "
            f"({interval})"
        )


# ============================================================
# 11. PHÂN BỐ ĐỘ DÀI GAP
# ============================================================
print(
    "\n================ PHÂN BỐ ĐỘ DÀI KHOẢNG MISSING ================\n"
)

for gap_length in sorted(
    global_gap_distribution
):

    number_of_gaps = (
        global_gap_distribution[gap_length]
    )

    print(
        f"{gap_length:>6} giờ : "
        f"{number_of_gaps:,} gap"
    )


# ============================================================
# 12. NHÓM GAP
# ============================================================
group_counter = Counter()

for (
    gap_length,
    number_of_gaps
) in global_gap_distribution.items():

    if gap_length == 1:

        group = "1 giờ"

    elif gap_length <= 3:

        group = "2-3 giờ"

    elif gap_length <= 6:

        group = "4-6 giờ"

    elif gap_length <= 24:

        group = "7-24 giờ"

    elif gap_length <= 72:

        group = "25-72 giờ"

    else:

        group = ">72 giờ"

    group_counter[group] += number_of_gaps


print(
    "\n================ NHÓM ĐỘ DÀI MISSING ================\n"
)

group_order = [
    "1 giờ",
    "2-3 giờ",
    "4-6 giờ",
    "7-24 giờ",
    "25-72 giờ",
    ">72 giờ"
]

for group in group_order:

    print(
        f"{group:<10} : "
        f"{group_counter[group]:,} gap"
    )


# ============================================================
# 13. 30 GAP DÀI NHẤT
# ============================================================
print(
    "\n================ 30 KHOẢNG MISSING DÀI NHẤT ================\n"
)

sorted_top_gaps = sorted(
    top_gaps,
    key=lambda item: item[0],
    reverse=True
)

for rank, item in enumerate(
    sorted_top_gaps,
    start=1
):

    (
        gap_length,
        building,
        start_time,
        end_time
    ) = item

    print(
        f"{rank:>2}. "
        f"{building:<30} | "
        f"{gap_length:>5} giờ | "
        f"{start_time} -> {end_time}"
    )


# ============================================================
# 14. BUILDING CÓ GAP DÀI NHẤT
# ============================================================
print(
    "\n================ BUILDING CÓ GAP DÀI NHẤT ================\n"
)

building_max_sorted = sorted(
    building_max_gap.items(),
    key=lambda item: item[1],
    reverse=True
)

for rank, (
    building,
    max_gap
) in enumerate(
    building_max_sorted[:30],
    start=1
):

    print(
        f"{rank:>2}. "
        f"{building:<30} | "
        f"Gap dài nhất: {max_gap:,} giờ | "
        f"Tổng số gap: "
        f"{building_gap_count[building]:,}"
    )


print(
    "\n================ HOÀN TẤT ================\n"
)