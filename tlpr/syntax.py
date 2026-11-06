import difflib
from typing import Any
import numpy as np

from tlpr.constants import (
    CLASS_NAME_TO_CHAR,
    DIGIT_NAMES,
    DIGIT_TO_THAI_CONFUSION,
    PROVINCES_77,
    THAI_TO_DIGIT_CONFUSION,
)


def match_province(raw_text: str) -> str:
    """จับคู่ชื่อจังหวัดที่อ่านได้จาก OCR กับรายชื่อ 77 จังหวัดทางการของไทยด้วย Fuzzy Matching"""
    if not raw_text.strip():
        return "อ่านชื่อจังหวัดไม่ได้"
    matches = difflib.get_close_matches(raw_text.strip(), PROVINCES_77, n=1, cutoff=0.3)
    return matches[0] if matches else raw_text.strip()


def parse_thai_license_syntax(
    char_model: Any,
    plate_crop: np.ndarray,
    conf_thresh: float = 0.05,
    device: str = "0",
) -> tuple[str, np.ndarray | None, float, Any, list]:
    """
    ตรวจจับและถอดรหัสตัวอักษรบนแผ่นป้ายทะเบียนไทยตามหลักไวยากรณ์ (Syntax Validation & Slot Clustering)
    
    ขั้นตอนการทำงาน:
    1. ตรวจจับตำแหน่งตัวอักษรและแยกกล่องจังหวัด (Province)
    2. จัดกลุ่ม Candidate Bounding Boxes ที่ตำแหน่งเดียวกันเข้าเป็น 'Slot' ด้วย Leader Overlap Clustering
    3. กรอง Slot ผีหรือสัญญาณรบกวน (Noise / Screws / Outliers)
    4. วิเคราะห์หาจุดแบ่ง (Split Point) ระหว่าง 'หมวดอักษร' กับ 'เลขทะเบียน' ด้วย Gap Distance & Syntax Fit
    5. บังคับใช้กฎไวยากรณ์ (Grammar Constraints):
       - โซนหมวดอักษร: ตัวแรกเป็น [1-9] หรือ [ก-ฮ], ตัวถัดไปต้องเป็น [ก-ฮ] เท่านั้น
       - โซนเลขทะเบียน: บังคับเลือกเฉพาะตัวเลข [0-9] เท่านั้น (พยัญชนะไม่มีสิทธิ์โผล่ในโซนนี้)

    Returns:
        tuple: (plate_text, province_crop, province_conf, char_results, kept_boxes)
    """
    crop_h, crop_w = plate_crop.shape[:2]
    char_results = char_model.predict(
        source=plate_crop, device=device, conf=conf_thresh, verbose=False
    )[0]

    raw_candidates = []
    province_crop = None
    province_conf = 0.0

    for cb in char_results.boxes:
        cls_id = int(cb.cls[0])
        cls_name = char_model.names[cls_id]
        conf = float(cb.conf[0])
        cx, cy, bw, bh = map(float, cb.xywh[0].tolist())
        bx1, by1, bx2, by2 = map(float, cb.xyxy[0].tolist())

        # 1. แยกกล่องจังหวัด (อยู่ครึ่งล่าง หรือมีชื่อคลาสเป็น Province)
        if cls_name == "Province" or cy > crop_h * 0.58:
            px1 = max(0, int(bx1) - 3)
            py1 = max(0, int(by1) - 2)
            px2 = min(crop_w, int(bx2) + 3)
            py2 = min(crop_h, int(by2) + 2)
            if py2 > py1 and px2 > px1:
                province_crop = plate_crop[py1:py2, px1:px2]
                province_conf = max(province_conf, conf)
            continue

        # กรองสัญญาณรบกวนขอบป้ายหรือหัวน็อต (ขนาดเล็กหรือสูงผิดปกติ)
        if bh < crop_h * 0.22 or bh > crop_h * 0.98:
            continue
        if (bx1 < crop_w * 0.01 or bx2 > crop_w * 0.99) and conf < 0.15:
            continue

        raw_candidates.append({
            "box": (bx1, by1, bx2, by2),
            "cx": cx,
            "cy": cy,
            "bw": bw,
            "bh": bh,
            "cls": cls_name,
            "conf": conf,
        })

    if not raw_candidates:
        return "", province_crop, province_conf, char_results, []

    med_w = float(np.median([r["bw"] for r in raw_candidates]))
    med_h = float(np.median([r["bh"] for r in raw_candidates]))

    # 2. Leader-Box Overlap Clustering เพื่อรวม Bounding Boxes ที่ทับซ้อนกันในตำแหน่งเดียวกัน
    def calc_1d_x_overlap(b1, b2):
        inter = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        min_w = min(b1[2] - b1[0], b2[2] - b2[0])
        return inter / min_w if min_w > 0 else 0.0

    sorted_raw = sorted(raw_candidates, key=lambda x: x["conf"], reverse=True)
    slots = []

    for r in sorted_raw:
        matched = None
        for s in slots:
            leader = s["items"][0]
            x_ov = calc_1d_x_overlap(leader["box"], r["box"])
            c_dist = abs(leader["cx"] - r["cx"])
            min_width = min(leader["bw"], r["bw"])
            # ถือว่าเป็นตัวอักษรเดียวกันหาก Overlap สูง หรือจุดศูนย์กลางใกล้กันมาก
            if x_ov > 0.50 or (c_dist < min_width * 0.40 and x_ov > 0.30):
                matched = s
                break
        if matched is not None:
            matched["items"].append(r)
        else:
            slots.append({"items": [r]})

    # 3. กรอง Slot สัญญาณรบกวน
    valid_slots = []
    for s in slots:
        max_c = max(item["conf"] for item in s["items"])
        slot_h = float(np.median([item["bh"] for item in s["items"]]))
        if max_c < 0.07 or slot_h < med_h * 0.55:
            continue
        valid_slots.append(s)

    if not valid_slots:
        return "", province_crop, province_conf, char_results, []

    # เรียง Slot จากซ้ายไปขวาตามพิกัด X
    valid_slots.sort(key=lambda s: s["items"][0]["cx"])

    # ยุบ Slot ที่อยู่ชิดกันเกินไป (Center ห่างกันน้อยกว่า 0.55 ของความกว้างตัวอักษร)
    merged_slots = []
    for s in valid_slots:
        if not merged_slots:
            merged_slots.append(s)
            continue
        prev = merged_slots[-1]
        prev_cx = prev["items"][0]["cx"]
        curr_cx = s["items"][0]["cx"]
        if abs(curr_cx - prev_cx) < med_w * 0.55:
            prev["items"].extend(s["items"])
        else:
            merged_slots.append(s)

    # ป้ายทะเบียนไทยปกติมีตัวอักษรแถวบนไม่เกิน 7-8 ตัว
    if len(merged_slots) > 7:
        merged_slots.sort(key=lambda s: max(item["conf"] for item in s["items"]), reverse=True)
        merged_slots = merged_slots[:7]
        merged_slots.sort(key=lambda s: s["items"][0]["cx"])

    K = len(merged_slots)
    slot_centers = [s["items"][0]["cx"] for s in merged_slots]
    gaps = [slot_centers[i + 1] - slot_centers[i] for i in range(K - 1)]

    # 4. หาจุดแบ่ง (Split Point) ระหว่าง 'หมวดอักษร' กับ 'เลขทะเบียน'
    # หมวดอักษร (M) มีความยาว 1-3 ตัว, เลขทะเบียน (K - M) มีความยาว 1-4 ตัว
    possible_splits = [m for m in [1, 2, 3] if 1 <= (K - m) <= 4]
    if not possible_splits:
        possible_splits = [max(1, min(K - 1, 2))]

    best_split = possible_splits[0]
    best_score = -1e9

    for m in possible_splits:
        score = 0.0
        # ประเมินขนาด Gap ระหว่างหมวดอักษรกับเลขทะเบียน
        if m - 1 < len(gaps):
            gap_at_m = gaps[m - 1]
            avg_other = (
                float(np.mean([gaps[j] for j in range(len(gaps)) if j != m - 1]))
                if len(gaps) > 1
                else gap_at_m
            )
            score += (gap_at_m / (avg_other + 1e-5)) * 3.0

        # ประเมินความสอดคล้องกับไวยากรณ์ (Syntax Fit Score)
        for i, s in enumerate(merged_slots):
            has_digit = any(item["cls"] in DIGIT_NAMES for item in s["items"])
            has_thai = any(item["cls"] not in DIGIT_NAMES for item in s["items"])
            top_conf = max(item["conf"] for item in s["items"])

            if i < m:
                # โซนหมวดอักษร
                if i == 0:
                    score += 1.0 * top_conf
                else:
                    score += (2.0 * top_conf) if has_thai else -1.0
            else:
                # โซนเลขทะเบียน (ต้องมีตัวเลข)
                score += (2.5 * top_conf) if has_digit else -1.5

        # ให้น้ำหนักล่วงหน้ากับโครงสร้างมาตรฐาน (หมวด 2 หรือ 3 ตัว)
        if m in (2, 3):
            score += 0.8

        if score > best_score:
            best_score = score
            best_split = m

    # 5. คัดเลือกตัวอักษรที่ถูกต้องตามตำแหน่ง (Enforced Character Selection)
    result_chars = []
    kept_boxes = []

    for i, s in enumerate(merged_slots):
        s_sorted = sorted(s["items"], key=lambda x: x["conf"], reverse=True)
        slot_bw = float(np.median([x["bw"] for x in s["items"]]))
        slot_bh = float(np.median([x["bh"] for x in s["items"]]))
        rep_box = s_sorted[0]["box"]

        if i < best_split:
            # --- โซนหมวดอักษร (Category Prefix) ---
            if i == 0:
                # ตำแหน่งแรก: เป็นได้ทั้งเลข 1-9 หรือพยัญชนะไทย (ห้ามเป็นเลข 0)
                cand = next((c for c in s_sorted if c["cls"] != "Zero"), s_sorted[0])
                cls_name = cand["cls"]
                char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
                conf = cand["conf"]
            else:
                # ตำแหน่งถัดมา: บังคับเป็นพยัญชนะไทยเท่านั้น (ห้ามเป็นตัวเลข)
                thai_cands = [c for c in s_sorted if c["cls"] not in DIGIT_NAMES]
                if thai_cands:
                    cand = thai_cands[0]
                    cls_name = cand["cls"]
                    char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
                    conf = cand["conf"]
                else:
                    # Visual Confusion Fallback
                    top_cls = s_sorted[0]["cls"]
                    char_text = DIGIT_TO_THAI_CONFUSION.get(
                        top_cls, CLASS_NAME_TO_CHAR.get(top_cls, top_cls)
                    )
                    conf = s_sorted[0]["conf"]
        else:
            # --- โซนเลขทะเบียน (Registration Numbers) ---
            # บังคับเป็นตัวเลข 0-9 เท่านั้น
            digit_cands = [c for c in s_sorted if c["cls"] in DIGIT_NAMES]
            if digit_cands:
                cand = digit_cands[0]
                cls_name = cand["cls"]
                char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
                conf = cand["conf"]
            else:
                # แปลงพยัญชนะกลับเป็นตัวเลขที่หน้าตาคล้ายกัน
                top_cls = s_sorted[0]["cls"]
                char_text = THAI_TO_DIGIT_CONFUSION.get(top_cls, "0")
                conf = s_sorted[0]["conf"]

        result_chars.append(char_text)
        kept_boxes.append((rep_box[0], rep_box[1], rep_box[2], rep_box[3], char_text, conf))

    # รวมผลลัพธ์โดยมีเว้นวรรคระหว่างหมวดอักษรกับเลขทะเบียน
    plate_text = "".join(result_chars[:best_split]) + " " + "".join(result_chars[best_split:])
    return plate_text, province_crop, province_conf, char_results, kept_boxes

