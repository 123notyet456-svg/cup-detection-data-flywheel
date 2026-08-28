from pathlib import Path
import cv2
import csv
import random

PREVIEW_DIR = Path(
    r"D:\cup_dataset_project\data\prelabel_preview"
)

REPORT_PATH = Path(
    r"D:\cup_dataset_project\reports\prelabel_qc.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# 获取所有预标注可视化图片
image_paths = sorted([
    p for p in PREVIEW_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
])

# 固定随机种子，保证以后重新运行还是同一批样本
random.seed(42)

sample_size = min(20, len(image_paths))

sample_paths = random.sample(
    image_paths,
    sample_size
)

# =========================
# 读取已有QC记录
# =========================

qc_results = {}

if REPORT_PATH.exists():

    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            filename = row.get("filename")
            status = row.get("qc_status")

            if filename and status:
                qc_results[filename] = status


# =========================
# 保存函数
# =========================

def save_results():

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "qc_status"
            ]
        )

        writer.writeheader()

        for filename, status in qc_results.items():

            writer.writerow({
                "filename": filename,
                "qc_status": status
            })


# =========================
# 开始QC
# =========================

print("=" * 60)
print("Pre-label QC")
print("=" * 60)

print("""
按键：

A = Accept
    标签可以直接使用

M = Minor Fix
    目标正确，只需要微调 Bounding Box

J = Major Fix
    漏标 / 错标 / 多标 / 严重框错

Q = Quit
    保存并退出
""")

for index, path in enumerate(sample_paths):

    if path.name in qc_results:
        continue

    img = cv2.imread(str(path))

    if img is None:
        continue

    display_img = img.copy()

    progress = (
        f"{index + 1}/{sample_size}"
    )

    instructions = (
        "A:accept  M:minor_fix  "
        "J:major_fix  Q:quit"
    )

    cv2.putText(
        display_img,
        progress,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    height = display_img.shape[0]

    cv2.putText(
        display_img,
        instructions,
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )

    cv2.imshow(
        "Pre-label QC",
        display_img
    )

    while True:

        key = cv2.waitKey(0) & 0xFF

        if key == ord("a"):
            status = "accept"
            break

        elif key == ord("m"):
            status = "minor_fix"
            break

        elif key == ord("j"):
            status = "major_fix"
            break

        elif key == ord("q"):
            save_results()
            cv2.destroyAllWindows()
            print("\n结果已保存。")
            exit()

    qc_results[path.name] = status

    save_results()

    print(
        f"{path.name} -> {status}"
    )


cv2.destroyAllWindows()

# =========================
# 统计
# =========================

accept = sum(
    x == "accept"
    for x in qc_results.values()
)

minor = sum(
    x == "minor_fix"
    for x in qc_results.values()
)

major = sum(
    x == "major_fix"
    for x in qc_results.values()
)

total = len(qc_results)

print("\n===== QC Summary =====")

print(f"QC数量：{total}")

print(f"Accept：{accept}")
print(f"Minor Fix：{minor}")
print(f"Major Fix：{major}")

if total > 0:

    print(
        f"Accept Rate："
        f"{accept / total * 100:.1f}%"
    )

    print(
        f"Minor Fix Rate："
        f"{minor / total * 100:.1f}%"
    )

    print(
        f"Major Fix Rate："
        f"{major / total * 100:.1f}%"
    )

print("\n报告：")
print(REPORT_PATH)

print("\nQC完成 ✅")