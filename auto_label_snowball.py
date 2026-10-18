import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
import easyocr
from tqdm import tqdm


CHAR_TO_CLASS = {
    # 10 Digits
    "0": 10,  # Zero
    "1": 4,   # One
    "2": 9,   # Two
    "3": 8,   # Three
    "4": 2,   # Four
    "5": 1,   # Five
    "6": 7,   # Six
    "7": 6,   # Seven
    "8": 0,   # Eight
    "9": 3,   # Nine
    # All 44 Thai Consonants (100% No Fallback)
    "ก": 14,  # gor-gai
    "ข": 18,  # kor-kai
    "ฃ": 19,  # kor-kuad
    "ค": 20,  # kor-kwai
    "ฅ": 54,  # kor-khon
    "ฆ": 21,  # kor-ra-kang
    "ง": 24,  # ngor-ngu
    "จ": 17,  # jor-jarn
    "ฉ": 41,  # chor-ching
    "ช": 12,  # chor-chang
    "ซ": 47,  # sor-so
    "ฌ": 13,  # chor-ga-cher
    "ญ": 40,  # yor-ying
    "ฎ": 51,  # dor-cha-da
    "ฏ": 52,  # tor-pa-tak
    "ฐ": 35,  # thor-than
    "ฑ": 53,  # thor-mon-tho
    "ฒ": 33,  # thor-pud-thao
    "ณ": 50,  # nor-nen
    "ด": 42,  # dor-dek
    "ต": 37,  # tor-tao
    "ถ": 45,  # thor-thung
    "ท": 34,  # thor-ta-harn
    "ธ": 36,  # thor-thong
    "น": 25,  # nor-nu
    "บ": 11,  # bor-bai-mai
    "ป": 43,  # por-pla
    "ผ": 27,  # por-pung
    "ฝ": 49,  # for-fa
    "พ": 26,  # por-parn
    "ฟ": 48,  # for-fan
    "ภ": 44,  # por-sam-phao
    "ม": 23,  # mor-ma
    "ย": 39,  # yor-yak
    "ร": 29,  # ror-ruea
    "ล": 22,  # lor-ling
    "ว": 38,  # vor-van
    "ศ": 31,  # sor-sa-la
    "ษ": 30,  # sor-rue-si
    "ส": 32,  # sor-suea
    "ห": 15,  # hor-heeb
    "ฬ": 28,  # ror-ju-la
    "อ": 46,  # or-ang
    "ฮ": 16,  # hor-nok-hooke
}

PROVINCE_CLASS = 5  # Province


def main():
    parser = argparse.ArgumentParser(description="Auto-Label Snowball สำหรับ Character Detection")
    parser.add_argument(
        "--source",
        default=r"C:\Users\jrgam\OneDrive\Desktop\Character Detector",
        help="โฟลเดอร์ภาพแผ่นป้ายที่ต้องการ Auto-label",
    )
    parser.add_argument(
        "--dest",
        default="thai-char-v2",
        help="โฟลเดอร์ปลายทางสำหรับเก็บ Dataset v2",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="จำนวนรูปภาพที่ต้องการ Auto-label ในรอบนี้ (ค่าเริ่มต้น: 300)",
    )
    args = parser.parse_args()

    src_dir = Path(args.source)
    if not src_dir.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์ภาพ: {src_dir}")

    dest_dir = Path(args.dest)
    train_img_dir = dest_dir / "train" / "images"
    train_lbl_dir = dest_dir / "train" / "labels"
    val_img_dir = dest_dir / "valid" / "images"
    val_lbl_dir = dest_dir / "valid" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. คัดลอกข้อมูลเดิมจากเวอร์ชันก่อนหน้า (thai-char-v2 หรือ thai-char-detection-1) มารวมด้วย
    prev_dir = Path("thai-char-v2") if Path("thai-char-v2").is_dir() else Path("thai-char-detection-1")
    if prev_dir.is_dir() and prev_dir.resolve() != dest_dir.resolve():
        print(f"กำลังรวมข้อมูลเดิมจาก {prev_dir.name} เข้ามาในชุดใหม่...")
        for split in ["train", "valid"]:
            s_img = prev_dir / split / "images"
            s_lbl = prev_dir / split / "labels"
            t_img = dest_dir / split / "images"
            t_lbl = dest_dir / split / "labels"
            if s_img.is_dir():
                for f in s_img.glob("*.jpg"):
                    shutil.copy2(f, t_img / f.name)
            if s_lbl.is_dir():
                for f in s_lbl.glob("*.txt"):
                    shutil.copy2(f, t_lbl / f.name)

    # 2. ค้นหารายชื่อไฟล์ภาพใหม่
    all_images = sorted(glob.glob(str(src_dir / "*.jpg")))
    if not all_images:
        raise SystemExit(f"ไม่พบไฟล์ .jpg ใน {src_dir}")

    # กรองเฉพาะภาพที่ยังไม่เคยมีใน dest
    existing_names = {p.name for p in train_img_dir.glob("*.jpg")} | {p.name for p in val_img_dir.glob("*.jpg")}
    new_images = [p for p in all_images if Path(p).name not in existing_names][: args.count]

    print(f"พบภาพทั้งหมด {len(all_images)} ภาพ")
    print(f"จะทำการ Auto-label เพิ่ม {len(new_images)} ภาพ...")

    # โหลด EasyOCR ด้วย GPU
    reader = easyocr.Reader(["th", "en"], gpu=True)

    labeled_count = 0
    val_interval = 5  # ทุกๆ 5 รูป ให้เก็บเข้า valid 1 รูป (สัดส่วน 80% train / 20% valid)

    for idx, img_path_str in enumerate(tqdm(new_images, desc="Auto-labeling")):
        img_path = Path(img_path_str)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        res = reader.readtext(img)
        if not res:
            continue

        yolo_boxes = []

        for poly, text, conf in res:
            if conf < 0.25:
                continue

            pts = [[int(pt[0]), int(pt[1])] for pt in poly]
            x_min = max(0, min(p[0] for p in pts))
            x_max = min(w, max(p[0] for p in pts))
            y_min = max(0, min(p[1] for p in pts))
            y_max = min(h, max(p[1] for p in pts))

            box_y_mid = (y_min + y_max) / 2.0
            clean_text = [c for c in text if c in CHAR_TO_CLASS]

            # ถ้าอยู่ครึ่งล่างของป้าย -> นับเป็น Province
            if box_y_mid > h * 0.50:
                xc = ((x_min + x_max) / 2.0) / w
                yc = ((y_min + y_max) / 2.0) / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h
                yolo_boxes.append(f"{PROVINCE_CLASS} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            else:
                # ถ้าอยู่ครึ่งบน -> แยกเป็นตัวเลขและพยัญชนะ
                if not clean_text:
                    continue
                n_chars = len(clean_text)
                char_w = (x_max - x_min) / n_chars
                for i, char in enumerate(clean_text):
                    cls_id = CHAR_TO_CLASS[char]
                    c_xmin = x_min + i * char_w
                    c_xmax = c_xmin + char_w
                    xc = ((c_xmin + c_xmax) / 2.0) / w
                    yc = ((y_min + y_max) / 2.0) / h
                    bw = (char_w * 0.95) / w
                    bh = (y_max - y_min) / h
                    yolo_boxes.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_boxes:
            continue

        # เลือกว่าจะลง train หรือ valid
        is_val = (idx % val_interval == 0)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # บันทึกภาพและไฟล์ label .txt
        target_img_file = target_img_dir / img_path.name
        target_lbl_file = target_lbl_dir / (img_path.stem + ".txt")

        shutil.copy2(img_path, target_img_file)
        with open(target_lbl_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_boxes) + "\n")

        labeled_count += 1

    # 3. คัดลอก data.yaml มาใส่ในโฟลเดอร์ใหม่
    src_yaml = Path("thai-char-v2") / "data.yaml" if (Path("thai-char-v2") / "data.yaml").is_file() else Path("thai-char-detection-1") / "data.yaml"
    if src_yaml.is_file() and src_yaml.resolve() != (dest_dir / "data.yaml").resolve():
        dest_yaml = dest_dir / "data.yaml"
        shutil.copy2(src_yaml, dest_yaml)

    print(f"\n Auto-label สำเร็จเรียบร้อย: {labeled_count} ภาพ")
    print(f" ภาพ Train ทั้งหมด: {len(list(train_img_dir.glob('*.jpg')))} ภาพ")
    print(f" ภาพ Valid ทั้งหมด: {len(list(val_img_dir.glob('*.jpg')))} ภาพ")
    print(f" Dataset พร้อมเทรนอยู่ที่: {dest_dir.resolve()}\\data.yaml")


if __name__ == "__main__":
    main()
