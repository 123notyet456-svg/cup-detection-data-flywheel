from pathlib import Path
import hashlib
import cv2
import csv

DATASET_DIR = Path(r"D:\cup_dataset_project\dataset_v1")
REPORT_PATH = Path(r"D:\cup_dataset_project\reports\leakage_report.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dhash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (9, 8))

    diff = img[:, 1:] > img[:, :-1]

    value = 0

    for bit in diff.flatten():
        value = (value << 1) | int(bit)

    return value


def hamming_distance(a, b):
    return (a ^ b).bit_count()


splits = {}

for split in ["train", "val", "test"]:

    folder = DATASET_DIR / "images" / split

    splits[split] = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


print("=" * 60)
print("Dataset Leakage Check")
print("=" * 60)

for split, paths in splits.items():
    print(f"{split}: {len(paths)}")


# ==========================================
# 1. 完全重复检查
# ==========================================

hash_records = {}

for split, paths in splits.items():

    for path in paths:

        file_hash = md5(path)

        hash_records.setdefault(
            file_hash, []
        ).append(
            (split, path.name)
        )


exact_leaks = []

for records in hash_records.values():

    split_names = {
        split
        for split, _
        in records
    }

    if len(split_names) > 1:
        exact_leaks.append(records)


print(
    f"\n跨 Split 完全重复组："
    f"{len(exact_leaks)}"
)


# ==========================================
# 2. 近似重复检查
# ==========================================

all_images = []

for split, paths in splits.items():

    for path in paths:

        image_hash = dhash(path)

        if image_hash is not None:

            all_images.append(
                (
                    split,
                    path.name,
                    image_hash
                )
            )


near_leaks = []

# 数字越小越严格
THRESHOLD = 5

for i in range(len(all_images)):

    split_a, name_a, hash_a = all_images[i]

    for j in range(i + 1, len(all_images)):

        split_b, name_b, hash_b = all_images[j]

        # 同一个 split 不属于 train/val/test 泄漏
        if split_a == split_b:
            continue

        distance = hamming_distance(
            hash_a,
            hash_b
        )

        if distance <= THRESHOLD:

            near_leaks.append({
                "split_a": split_a,
                "file_a": name_a,
                "split_b": split_b,
                "file_b": name_b,
                "distance": distance
            })


print(
    f"跨 Split 近似重复候选："
    f"{len(near_leaks)}"
)


# ==========================================
# 3. 保存近似重复报告
# ==========================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split_a",
            "file_a",
            "split_b",
            "file_b",
            "distance"
        ]
    )

    writer.writeheader()
    writer.writerows(near_leaks)


# ==========================================
# 4. Summary
# ==========================================

print("\n===== Check Summary =====")

if len(exact_leaks) == 0:
    print("Exact Duplicate Leakage：PASS ✅")
else:
    print("Exact Duplicate Leakage：FAIL ❌")

if len(near_leaks) == 0:
    print("Near Duplicate：PASS ✅")
else:
    print("Near Duplicate：需要人工复核 ⚠️")

print("\n报告：")
print(REPORT_PATH)

print("\nLeakage Check 完成 ✅")