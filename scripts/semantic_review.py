from pathlib import Path
import cv2
import csv

# =========================
# 路径配置
# =========================

CLEANED_DIR = Path(r"D:\cup_dataset_project\data\cleaned")
REPORT_PATH = Path(
    r"D:\cup_dataset_project\reports\semantic_review.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================
# 获取全部图片
# =========================

image_paths = sorted([
    p for p in CLEANED_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
])


# =========================
# 读取已有审核记录
# =========================

review_results = {}

if REPORT_PATH.exists():

    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            filename = row.get("filename")
            status = row.get("status")

            if filename and status:
                review_results[filename] = status


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
                "status"
            ]
        )

        writer.writeheader()

        for filename, status in review_results.items():

            writer.writerow({
                "filename": filename,
                "status": status
            })


# =========================
# 显示当前统计
# =========================

def print_summary():

    positive = sum(
        status == "positive"
        for status in review_results.values()
    )

    negative = sum(
        status == "negative"
        for status in review_results.values()
    )

    review = sum(
        status == "review"
        for status in review_results.values()
    )

    reject = sum(
        status == "reject"
        for status in review_results.values()
    )

    print("\n===== Review Summary =====")

    print(f"Positive：{positive}")
    print(f"Negative：{negative}")
    print(f"Review：{review}")
    print(f"Reject：{reject}")

    print(
        f"已审核：{len(review_results)} / "
        f"{len(image_paths)}"
    )


# =========================
# 程序开始
# =========================

print("=" * 60)
print("Semantic Quality Review")
print("=" * 60)

print("""
按键说明：

P = Positive
    有杯子，可用于训练

N = Negative
    没有杯子，但可作为负样本

R = Review
    暂时不确定，需要后续检查

X = Reject
    明显无效数据

Q = Quit
    保存并退出
""")

print(
    f"总图片数：{len(image_paths)}"
)

print(
    f"已审核：{len(review_results)}"
)

print(
    f"剩余：{len(image_paths) - len(review_results)}"
)

if len(review_results) > 0:

    print("\n检测到已有审核记录。")
    print("将自动跳过已审核图片并继续。")


# =========================
# 开始审核
# =========================

quit_requested = False

for index, path in enumerate(image_paths):

    # 已经审核过的图片自动跳过
    if path.name in review_results:
        continue

    img = cv2.imread(str(path))

    if img is None:

        print(
            f"无法读取：{path.name}"
        )

        review_results[path.name] = "reject"

        save_results()

        continue


    display_img = img.copy()


    # 当前整体位置
    progress_text = (
        f"{index + 1}/{len(image_paths)}"
    )

    # 已审核数量
    reviewed_text = (
        f"Reviewed: "
        f"{len(review_results)}/"
        f"{len(image_paths)}"
    )

    key_text = (
        "P:positive  "
        "N:negative  "
        "R:review  "
        "X:reject  "
        "Q:quit"
    )


    cv2.putText(
        display_img,
        progress_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display_img,
        reviewed_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    height = display_img.shape[0]

    cv2.putText(
        display_img,
        key_text,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


    cv2.imshow(
        "Semantic Review",
        display_img
    )


    while True:

        key = cv2.waitKey(0) & 0xFF


        if key == ord("p"):

            status = "positive"
            break


        elif key == ord("n"):

            status = "negative"
            break


        elif key == ord("r"):

            status = "review"
            break


        elif key == ord("x"):

            status = "reject"
            break


        elif key == ord("q"):

            quit_requested = True
            break


    if quit_requested:

        print("\n收到退出指令。")
        print("审核结果已经保存。")

        break


    # 保存当前判断
    review_results[path.name] = status

    # 每判断一张就立即保存
    save_results()

    print(
        f"{len(review_results)}/"
        f"{len(image_paths)} "
        f"{path.name} "
        f"-> {status}"
    )


# =========================
# 程序结束
# =========================

cv2.destroyAllWindows()

save_results()

print_summary()

print("\n报告保存到：")
print(REPORT_PATH)

if len(review_results) == len(image_paths):

    print("\n全部图片审核完成 ✅")

else:

    print("\n本次审核已保存，可下次继续 ✅")