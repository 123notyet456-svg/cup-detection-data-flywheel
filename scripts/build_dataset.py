from pathlib import Path
import random
import shutil
import csv

PROJECT_DIR = Path(r"D:\cup_dataset_project")

# 最终有效图片
IMAGE_DIR = PROJECT_DIR / "prelabel_source" / "images"

# Label Studio人工验收后的最终标签
LABEL_DIR = PROJECT_DIR / "final_export" / "labels"

# 最终YOLO数据集
OUTPUT_DIR = PROJECT_DIR / "dataset_v1"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================
# 1. 找图片和标签
# ==========================================

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

labels = [
    p for p in LABEL_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() == ".txt"
]

print("=" * 60)
print("Final Dataset Pairing Check")
print("=" * 60)

print(f"找到图片：{len(images)}")
print(f"找到TXT标签：{len(labels)}")

label_map = {
    p.stem: p
    for p in labels
}

image_stems = {
    p.stem
    for p in images
}

# ==========================================
# 2. 配对
# ==========================================

paired = []
missing_labels = []

for image in images:

    label = label_map.get(image.stem)

    if label:
        paired.append((image, label))

    else:
        missing_labels.append(image)


orphan_labels = [
    label
    for label in labels
    if label.stem not in image_stems
]


print(f"\n成功配对：{len(paired)}")
print(f"图片缺少标签：{len(missing_labels)}")
print(f"孤立标签：{len(orphan_labels)}")

if missing_labels:

    print("\n缺少标签的图片：")

    for p in missing_labels:
        print(p.name)


if orphan_labels:

    print("\n没有对应图片的标签：")

    for p in orphan_labels:
        print(p.name)


if len(paired) == 0:

    raise RuntimeError(
        "没有找到有效的图片-标签配对"
    )


# ==========================================
# 3. 划分 train / val / test
# ==========================================

random.seed(42)
random.shuffle(paired)

total = len(paired)

train_count = int(total * 0.8)
val_count = int(total * 0.1)

train_data = paired[:train_count]

val_data = paired[
    train_count:
    train_count + val_count
]

test_data = paired[
    train_count + val_count:
]

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}


# ==========================================
# 4. 如果dataset_v1已经存在，先清掉
# ==========================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)


for split in splits:

    (
        OUTPUT_DIR
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================
# 5. 复制图片和标签
# ==========================================

split_records = []

for split_name, split_data in splits.items():

    for image_path, label_path in split_data:

        shutil.copy2(
            image_path,
            OUTPUT_DIR
            / "images"
            / split_name
            / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_DIR
            / "labels"
            / split_name
            / label_path.name
        )

        split_records.append({
            "filename": image_path.name,
            "split": split_name
        })


# ==========================================
# 6. 生成划分记录
# ==========================================

report_path = (
    REPORT_DIR
    / "dataset_split.csv"
)

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "split"
        ]
    )

    writer.writeheader()
    writer.writerows(split_records)


# ==========================================
# 7. 生成 YOLO data.yaml
# ==========================================

yaml_path = OUTPUT_DIR / "data.yaml"

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: cup
"""

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_content)


# ==========================================
# 8. 最终统计
# ==========================================

print("\n===== Dataset Split =====")

print(f"Total：{total}")
print(f"Train：{len(train_data)}")
print(f"Val：{len(val_data)}")
print(f"Test：{len(test_data)}")

print("\nDataset目录：")
print(OUTPUT_DIR)

print("\ndata.yaml：")
print(yaml_path)

print("\nDataset Construction 完成 ✅")