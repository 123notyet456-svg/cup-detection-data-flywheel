from pathlib import Path
from collections import Counter, defaultdict
import cv2
import hashlib
import csv

# 原始数据目录
RAW_DIR = Path(r"D:\cup_dataset_project\data\raw")

# 报告输出目录
REPORT_DIR = Path(r"D:\cup_dataset_project\reports")
REPORT_DIR.mkdir(exist_ok=True)

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 递归寻找所有图片
image_paths = [
    p for p in RAW_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print("=" * 50)
print("开始扫描原始图片")
print("=" * 50)

print(f"\n找到图片数量：{len(image_paths)}")

records = []
format_counter = Counter()
hash_dict = defaultdict(list)

valid_count = 0
invalid_count = 0

widths = []
heights = []

small_images = []

for path in image_paths:

    extension = path.suffix.lower()
    format_counter[extension] += 1

    # 文件大小
    file_size_kb = path.stat().st_size / 1024

    # MD5，用于检测完全重复图片
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    hash_dict[file_hash].append(path.name)

    # OpenCV读取图片
    img = cv2.imread(str(path))

    if img is None:

        invalid_count += 1

        records.append({
            "filename": path.name,
            "path": str(path),
            "status": "unreadable",
            "width": "",
            "height": "",
            "file_size_kb": round(file_size_kb, 2),
            "small_image": ""
        })

        continue

    valid_count += 1

    height, width = img.shape[:2]

    widths.append(width)
    heights.append(height)

    # 小尺寸图片候选
    is_small = width < 300 or height < 300

    if is_small:
        small_images.append(path.name)

    records.append({
        "filename": path.name,
        "path": str(path),
        "status": "ok",
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "small_image": is_small
    })


# 找完全重复图片
duplicate_groups = [
    files
    for files in hash_dict.values()
    if len(files) > 1
]


# 输出CSV报告
report_path = REPORT_DIR / "data_quality_report.csv"

with open(report_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "path",
            "status",
            "width",
            "height",
            "file_size_kb",
            "small_image"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


# 打印扫描结果
print("\n===== 数据质量报告 =====")

print(f"\n总图片数：{len(image_paths)}")
print(f"可正常读取：{valid_count}")
print(f"无法读取：{invalid_count}")

print("\n图片格式：")

for fmt, count in format_counter.items():
    print(f"{fmt}: {count}")


if widths:

    print("\n尺寸统计：")

    print(f"最小宽度：{min(widths)}")
    print(f"最大宽度：{max(widths)}")
    print(f"平均宽度：{sum(widths) / len(widths):.1f}")

    print(f"最小高度：{min(heights)}")
    print(f"最大高度：{max(heights)}")
    print(f"平均高度：{sum(heights) / len(heights):.1f}")


print(f"\n小尺寸图片候选：{len(small_images)}")

print(f"完全重复图片组数：{len(duplicate_groups)}")

if duplicate_groups:

    print("\n重复图片：")

    for group in duplicate_groups:
        print(group)


print("\nCSV报告已保存到：")
print(report_path)

print("\n扫描完成 ✅")