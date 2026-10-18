import argparse
import difflib
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_PLATE_MODEL = (
    PROJECT_DIR / "weights" / "plate_detector.pt"
    if (PROJECT_DIR / "weights" / "plate_detector.pt").exists()
    else PROJECT_DIR / "runs" / "detect" / "plate-gpu-v1" / "weights" / "best.pt"
)
DEFAULT_CHAR_MODEL = (
    PROJECT_DIR / "weights" / "char_detector.pt"
    if (PROJECT_DIR / "weights" / "char_detector.pt").exists()
    else PROJECT_DIR / "runs" / "detect" / "char-v5" / "weights" / "best.pt"
)

DIGIT_NAMES = {
    "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine",
}

CLASS_NAME_TO_CHAR = {
    # Digits
    "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4",
    "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9",
    # Thai Consonants (44 characters)
    "gor-gai": "ก", "kor-kai": "ข", "kor-kuad": "ฃ", "kor-kwai": "ค", "kor-khon": "ฅ",
    "kor-ra-kang": "ฆ", "ngor-ngu": "ง", "jor-jarn": "จ", "chor-ching": "ฉ", "chor-chang": "ช",
    "sor-so": "ซ", "chor-ga-cher": "ฌ", "yor-ying": "ญ", "dor-cha-da": "ฎ", "tor-pa-tak": "ฏ",
    "thor-than": "ฐ", "thor-mon-tho": "ฑ", "thor-pud-thao": "ฒ", "nor-nen": "ณ", "dor-dek": "ด",
    "tor-tao": "ต", "thor-thung": "ถ", "thor-ta-harn": "ท", "thor-thong": "ธ", "nor-nu": "น",
    "bor-bai-mai": "บ", "por-pla": "ป", "por-pung": "ผ", "for-fa": "ฝ", "por-parn": "พ",
    "for-fan": "ฟ", "por-sam-phao": "ภ", "mor-ma": "ม", "yor-yak": "ย", "ror-ruea": "ร",
    "lor-ling": "ล", "vor-van": "ว", "sor-sa-la": "ศ", "sor-rue-si": "ษ", "sor-suea": "ส",
    "hor-heeb": "ห", "ror-ju-la": "ฬ", "or-ang": "อ", "hor-nok-hooke": "ฮ",
}

PROVINCES_77 = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา",
    "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน",
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต",
    "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี",
    "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี",
    "อุบลราชธานี",
]


def match_province(raw_text: str) -> str:
    if not raw_text.strip():
        return "อ่านชื่อจังหวัดไม่ได้"
    matches = difflib.get_close_matches(raw_text.strip(), PROVINCES_77, n=1, cutoff=0.3)
    return matches[0] if matches else raw_text.strip()


def parse_thai_license_syntax(char_model, plate_crop, conf_thresh: float = 0.08, device: str = "0"):
    """
    ตรวจจับและถอดรหัสตัวอักษรบนแผ่นป้ายทะเบียนไทยตามหลักไวยากรณ์ (Syntax Validation)
    - Pattern A: [1 เลข] [2 พยัญชนะ] [1-4 เลข] เช่น 7กล 9108
    - Pattern B: [2 พยัญชนะ] [1-4 เลข] เช่น ฒก 8534, กก 1234
    """
    crop_h, crop_w = plate_crop.shape[:2]
    char_results = char_model.predict(
        source=plate_crop, device=device, conf=conf_thresh, verbose=False
    )[0]

    top_candidates = []
    province_crop = None
    province_conf = 0.0

    for cb in char_results.boxes:
        cls_id = int(cb.cls[0])
        cls_name = char_model.names[cls_id]
        conf = float(cb.conf[0])
        cx, cy, bw, bh = map(float, cb.xywh[0].tolist())

        if cls_name == "Province" or cy > crop_h * 0.58:
            px1, py1, px2, py2 = map(int, cb.xyxy[0].tolist())
            px1 = max(0, px1 - 3)
            py1 = max(0, py1 - 2)
            px2 = min(crop_w, px2 + 3)
            py2 = min(crop_h, py2 + 2)
            province_crop = plate_crop[py1:py2, px1:px2]
            province_conf = max(province_conf, conf)
        else:
            top_candidates.append((cx, cy, bw, bh, cls_name, conf))

    # จัดเรียงจากซ้ายไปขวา
    top_candidates.sort(key=lambda b: b[0])

    # จัดกลุ่มตัวอักษรที่อยู่ในตำแหน่งเดียวกัน (Slot Clustering)
    clusters = []
    for b in top_candidates:
        if not clusters or abs(b[0] - clusters[-1][0][0]) > (crop_w * 0.055):
            clusters.append([b])
        else:
            clusters[-1].append(b)

    if not clusters:
        return "", province_crop, province_conf, char_results

    # วิเคราะห์ว่าเป็น Pattern A (มีเลขนำหน้า เช่น 7กล) หรือ Pattern B (พยัญชนะนำหน้า เช่น ฒก)
    slot0_cands = clusters[0]
    best_s0 = max(slot0_cands, key=lambda x: x[5])
    is_pattern_a = (best_s0[4] in DIGIT_NAMES and len(clusters) >= 6)

    result_chars = []
    for idx, cl in enumerate(clusters):
        # กำหนดชนิดที่คาดหวังตาม Syntax กฎหมายไทย
        if is_pattern_a:
            # Pattern A: ช่อง 0=เลข, ช่อง 1=พยัญชนะ, ช่อง 2=พยัญชนะ, ช่อง 3 ขึ้นไป=เลข
            expect_digit = (idx == 0 or idx >= 3)
        else:
            # Pattern B: ช่อง 0=พยัญชนะ, ช่อง 1=พยัญชนะ, ช่อง 2 ขึ้นไป=เลข
            expect_digit = (idx >= 2)

        # กรองตัวเลือกที่ตรงตาม Syntax ของช่องนั้น
        filtered = [c for c in cl if (c[4] in DIGIT_NAMES) == expect_digit]
        chosen = max(filtered, key=lambda x: x[5]) if filtered else max(cl, key=lambda x: x[5])

        # ป้องกันความสับสนระหว่างเลข 1 กับเลข 7 สำหรับช่องตัวเลข
        if expect_digit and idx > 0:
            cand_names = [c[4] for c in cl]
            if "One" in cand_names and "Seven" in cand_names and chosen[4] == "Seven":
                bw, bh = chosen[2], chosen[3]
                if (bw / bh) < 0.48:  # เลข 1 มีทรงสลิมแคบกว่าเลข 7
                    chosen = next(c for c in cl if c[4] == "One")

        result_chars.append(CLASS_NAME_TO_CHAR.get(chosen[4], chosen[4]))

    plate_text = "".join(result_chars)
    return plate_text, province_crop, province_conf, char_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-Stage End-to-End YOLO Thai License Plate Recognition System (Syntax Validated)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(PROJECT_DIR / "truck.png"),
        help="Path to image file (Default: truck.png)",
    )
    parser.add_argument(
        "--plate-model",
        default=str(DEFAULT_PLATE_MODEL),
        help="Path to Plate Detection weights (Stage 1)",
    )
    parser.add_argument(
        "--char-model",
        default=str(DEFAULT_CHAR_MODEL),
        help="Path to Character Detection weights (Stage 2)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.08,
        help="Base confidence threshold for candidate detection (Default: 0.08)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device to run inference on (Default: 0 for GPU, or 'cpu')",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    plate_model_path = Path(args.plate_model)
    char_model_path = Path(args.char_model)

    if not image_path.is_file():
        parser.error(f"Not Found Image File: {image_path}")
    if not plate_model_path.is_file():
        parser.error(f"Not Found Plate Model File: {plate_model_path}")
    if not char_model_path.is_file():
        parser.error(f"Not Found Character Model File: {char_model_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        parser.error(f"Can't Read Image File: {image_path}")

    height, width = image.shape[:2]
    output_dir = PROJECT_DIR / "runs" / "lpr_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 Thai License Plate Recognition (Syntax Validated Pipeline)")
    print(f"📷 Image: {image_path.name} ({width}x{height})")
    print(f"🎯 Plate Model: {plate_model_path.name}")
    print(f"🔤 Char Model:  {char_model_path.name}")
    print(f"⚙️ Device:      GPU ({args.device})")
    print("=" * 60)

    # Initialize models
    plate_model = YOLO(str(plate_model_path))
    char_model = YOLO(str(char_model_path))
    ocr_reader = easyocr.Reader(["th"], gpu=(args.device != "cpu"))

    # Stage 1: Plate Detection
    plate_results = plate_model.predict(
        source=image, device=args.device, verbose=False
    )[0]

    if plate_results.boxes is None or len(plate_results.boxes) == 0:
        print("\n❌ Not Found License Plate in this image.")
        return

    annotated_vehicle = image.copy()

    for idx, p_box in enumerate(plate_results.boxes, start=1):
        x1, y1, x2, y2 = map(int, p_box.xyxy[0].tolist())
        plate_conf = float(p_box.conf[0])

        pad_x = int((x2 - x1) * 0.03)
        pad_y = int((y2 - y1) * 0.05)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(height, y2 + pad_y)

        plate_crop = image[cy1:cy2, cx1:cx2]

        # Stage 2: Character Detection with Thai Syntax Validation
        plate_text, province_crop, province_conf, char_results = parse_thai_license_syntax(
            char_model=char_model,
            plate_crop=plate_crop,
            conf_thresh=args.conf,
            device=args.device,
        )

        # Read actual Province Name using OCR on the exact Province Crop
        province_name = "ไม่พบกล่องจังหวัด"
        if province_crop is not None and province_crop.size > 0:
            ph, pw = province_crop.shape[:2]
            p_resized = cv2.resize(province_crop, (pw * 2, ph * 2), interpolation=cv2.INTER_CUBIC)
            res = ocr_reader.readtext(p_resized)
            raw_prov = "".join(r[1] for r in res)
            province_name = match_province(raw_prov)

        print(f"\n🏷️  Plate #{idx} (Detection Confidence: {plate_conf:.0%}):")
        print(f"   ► หมวดเลขทะเบียน: {plate_text if plate_text else '(อ่านตัวเลข/พยัญชนะไม่ได้)'}")
        print(f"   ► จังหวัด:         {province_name}")

        # Save annotated crop
        crop_annotated = char_results.plot()
        crop_path = output_dir / f"plate_{idx}_annotated.jpg"
        cv2.imwrite(str(crop_path), crop_annotated)
        print(f"   ► Crop ผลลัพธ์:     {crop_path}")

        # Draw on main image
        cv2.rectangle(annotated_vehicle, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_vehicle,
            f"{plate_text} ({plate_conf:.0%})",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    result_image_path = output_dir / "detections.jpg"
    cv2.imwrite(str(result_image_path), annotated_vehicle)
    print(f"\n🖼️  ภาพรวมผลตรวจจับทั้งหมด: {result_image_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
