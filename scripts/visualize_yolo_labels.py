from pathlib import Path
import cv2

IMAGE_DIR = Path(
    r"D:\cup_dataset_project\cup.v1i.yolov8\train\images"
)

LABEL_DIR = Path(
    r"D:\cup_dataset_project\cup.v1i.yolov8\train\labels"
)

OUTPUT_DIR = Path(
    r"D:\cup_dataset_project\data\prelabel_preview"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

processed = 0
missing_labels = 0
box_count = 0

for image_path in image_paths:

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"无法读取图片：{image_path.name}")
        continue

    height, width = img.shape[:2]

    # YOLO 标签文件与图片同名，只是后缀改成 txt
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():

        missing_labels += 1

        print(
            f"没有找到标签：{image_path.name}"
        )

        continue

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            print(
                f"标签格式异常：{label_path.name}"
            )
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # YOLO归一化坐标 → 像素坐标
        x_center_px = x_center * width
        y_center_px = y_center * height

        box_width_px = box_width * width
        box_height_px = box_height * height

        x1 = int(
            x_center_px - box_width_px / 2
        )

        y1 = int(
            y_center_px - box_height_px / 2
        )

        x2 = int(
            x_center_px + box_width_px / 2
        )

        y2 = int(
            y_center_px + box_height_px / 2
        )

        # 防止坐标跑出图片范围
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # 画 Bounding Box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 类别名称
        cv2.putText(
            img,
            "cup",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        box_count += 1

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        img
    )

    processed += 1


print("=" * 50)

print("YOLO标签可视化完成")

print("=" * 50)

print(f"图片总数：{len(image_paths)}")

print(f"成功生成预览：{processed}")

print(f"缺少标签：{missing_labels}")

print(f"Bounding Box 总数：{box_count}")

print("\n输出目录：")

print(OUTPUT_DIR)

print("\n完成 ✅")