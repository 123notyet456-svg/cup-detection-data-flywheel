from pathlib import Path
import shutil

PROJECT_DIR = Path(r"D:\cup_dataset_project")

IMAGE_SOURCE = PROJECT_DIR / "data" / "cleaned"

TRAIN_LABELS = PROJECT_DIR / "cup.v1i.yolov8" / "train" / "labels"
VALID_LABELS = PROJECT_DIR / "cup.v1i.yolov8" / "valid" / "labels"

OUTPUT_DIR = PROJECT_DIR / "prelabel_source"
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_SOURCE.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

copied = 0
missing = []

for image_path in images:

    # 复制清洗后的图片
    shutil.copy2(
        image_path,
        OUTPUT_IMAGES / image_path.name
    )

    label_name = image_path.stem + ".txt"

    train_label = TRAIN_LABELS / label_name
    valid_label = VALID_LABELS / label_name

    if train_label.exists():

        shutil.copy2(
            train_label,
            OUTPUT_LABELS / label_name
        )

        copied += 1

    elif valid_label.exists():

        shutil.copy2(
            valid_label,
            OUTPUT_LABELS / label_name
        )

        copied += 1

    else:

        missing.append(image_path.name)


# YOLO类别文件
with open(
    OUTPUT_DIR / "classes.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("cup\n")


print("=" * 50)
print("Pre-label Source 准备完成")
print("=" * 50)

print(f"Cleaned 图片：{len(images)}")
print(f"成功匹配标签：{copied}")
print(f"缺少标签：{len(missing)}")

if missing:

    print("\n缺少标签的图片：")

    for name in missing:
        print(name)

print("\n输出目录：")
print(OUTPUT_DIR)