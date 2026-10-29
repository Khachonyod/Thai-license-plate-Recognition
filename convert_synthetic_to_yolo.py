# บันทึกเป็น convert_synthetic_to_yolo.py แล้วสั่งรัน
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path(r"D:\dev\license-plate-generater\synthetic_plates")
DEST_DIR = Path(r"D:\dev\TLPR\datasets\synthetic_yolo")

CHAR_TO_CLASS = {
    # 10 Digits
    "0": 10, "1": 4, "2": 9, "3": 8, "4": 2, "5": 1, "6": 7, "7": 6, "8": 0, "9": 3,
    # 44 Thai Consonants
    "ก": 14, "ข": 18, "ฃ": 19, "ค": 20, "ฅ": 54, "ฆ": 21, "ง": 24, "จ": 17, "ฉ": 41,
    "ช": 12, "ซ": 47, "ฌ": 13, "ญ": 40, "ฎ": 51, "ฏ": 52, "ฐ": 35, "ฑ": 53, "ฒ": 33,
    "ณ": 50, "ด": 42, "ต": 37, "ถ": 45, "ท": 34, "ธ": 36, "น": 25, "บ": 11, "ป": 43,
    "ผ": 27, "ฝ": 49, "พ": 26, "ฟ": 48, "ภ": 44, "ม": 23, "ย": 39, "ร": 29, "ล": 22,
    "ว": 38, "ศ": 31, "ษ": 30, "ส": 32, "ห": 15, "ฬ": 28, "อ": 46, "ฮ": 16,
}
PROVINCE_CLASS = 5

for split in ["train", "val"]:
    (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

lines = (SRC_DIR / "annotations.jsonl").read_text(encoding="utf-8").strip().splitlines()
random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * 0.9)  # 90% Train, 10% Val

for idx, line in enumerate(tqdm(lines, desc="Converting")):
    data = json.loads(line)
    img_name = data["image"]
    img_w, img_h = data["width"], data["height"]
    split = "train" if idx < split_idx else "val"

    # Copy Image
    src_img = SRC_DIR / img_name
    dest_img = DEST_DIR / "images" / split / img_name
    shutil.copy2(src_img, dest_img)

    # Process Labels
    yolo_lines = []
    prov_boxes = []

    for ann in data.get("annotations", []):
        field = ann.get("field")
        ch = ann.get("char")
        bx1, by1, bx2, by2 = ann["bbox_xyxy"]

        if field == "province":
            prov_boxes.append((bx1, by1, bx2, by2))
        elif field == "registration" and ch in CHAR_TO_CLASS:
            cls_id = CHAR_TO_CLASS[ch]
            # Convert to YOLO normalized xywh
            cx = ((bx1 + bx2) / 2.0) / img_w
            cy = ((by1 + by2) / 2.0) / img_h
            w = (bx2 - bx1) / img_w
            h = (by2 - by1) / img_h
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Merge individual province chars into 1 Province bounding box
    if prov_boxes:
        px1 = min(b[0] for b in prov_boxes)
        py1 = min(b[1] for b in prov_boxes)
        px2 = max(b[2] for b in prov_boxes)
        py2 = max(b[3] for b in prov_boxes)
        cx = ((px1 + px2) / 2.0) / img_w
        cy = ((py1 + py2) / 2.0) / img_h
        w = (px2 - px1) / img_w
        h = (py2 - py1) / img_h
        yolo_lines.append(f"{PROVINCE_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_file = DEST_DIR / "labels" / split / (Path(img_name).stem + ".txt")
    label_file.write_text("\n".join(yolo_lines), encoding="utf-8")

print("Conversion complete!")