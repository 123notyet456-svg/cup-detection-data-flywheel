from pathlib import Path
import hashlib
import shutil
import csv

RAW_DIR = Path(r"D:\cup_dataset_project\data\raw")
CLEANED_DIR = Path(r"D:\cup_dataset_project\data\cleaned")
REPORT_DIR = Path(r"D:\cup_dataset_project\reports")

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    p for p in RAW_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

seen_hashes = {}
duplicate_records = []

kept_count = 0
duplicate_count = 0

for path in image_paths:

    # 计算文件 MD5
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash not in seen_hashes:

        # 第一次出现，保留
        seen_hashes[file_hash] = path.name

        target_path = CLEANED_DIR / path.name

        shutil.copy2(path, target_path)

        kept_count += 1

    else:

        # 已经出现过，是重复图片
        duplicate_count += 1

        duplicate_records.append({
            "duplicate_file": path.name,
            "kept_file": seen_hashes[file_hash],
            "md5": file_hash
        })


# 输出重复图片报告
report_path = REPORT_DIR / "duplicate_report.csv"

with open(report_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "duplicate_file",
            "kept_file",
            "md5"
        ]
    )

    writer.writeheader()
    writer.writerows(duplicate_records)


print("=" * 50)
print("图片去重完成")
print("=" * 50)

print(f"原始图片：{len(image_paths)}")
print(f"保留图片：{kept_count}")
print(f"重复图片：{duplicate_count}")

print("\n清洗后的图片目录：")
print(CLEANED_DIR)

print("\n重复图片报告：")
print(report_path)

print("\n完成 ✅")