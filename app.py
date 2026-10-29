import argparse
import difflib
import os
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_PLATE_MODEL = PROJECT_DIR / "weights" / "plate_detector.pt"
DEFAULT_CHAR_MODEL = PROJECT_DIR / "weights" / "char_detector.pt"

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


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    จัดเรียงจุด 4 จุดของสี่เหลี่ยมตามลำดับ:
    0: บน-ซ้าย (Top-Left)
    1: บน-ขวา (Top-Right)
    2: ล่าง-ขวา (Bottom-Right)
    3: ล่าง-ซ้าย (Bottom-Left)
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # ผลรวม x + y: ค่าน้อยสุด = บน-ซ้าย, ค่ามากสุด = ล่าง-ขวา
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # ผลต่าง y - x: บน-ขวา มีค่าน้อยสุด, ล่าง-ซ้าย มีค่ามากสุด
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def enhance_clahe_denoise(
    image_bgr,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    denoise_method: str = "bilateral",
):
    """
    ปรับปรุงคุณภาพภาพแผ่นป้ายทะเบียนด้วยเทคนิค CLAHE และ Denoising
    เพื่อเพิ่มคอนทราสต์ของตัวอักษร ลดเงา/แสงสะท้อน และลบสัญญาณรบกวน (Noise) โดยรักษาขอบภาพให้คมชัด

    Args:
        image_bgr: ภาพ BGR ของแผ่นป้าย
        clip_limit: ขีดจำกัดคอนทราสต์ของ CLAHE (ป้องกันการขยาย noise เกินไป)
        tile_grid_size: ขนาดบล็อกย่อย (Grid) สำหรับทำ Adaptive Equalization
        denoise_method: วิธีการ Denoising:
            - 'bilateral': Bilateral Filter (รักษาเส้นขอบตัวอักษรได้ยอดเยี่ยมและรวดเร็ว)
            - 'nlmeans': Non-Local Means Denoising (ลดสัญญาณรบกวนขั้นสูงสำหรับภาพสี)
            - 'gaussian': Gaussian Blur แบบนุ่มนวล
            - 'none': ข้ามขั้นตอน Denoising

    Returns:
        np.ndarray: ภาพ BGR ที่ผ่านการปรับปรุงคอนทราสต์และลดสัญญาณรบกวนแล้ว
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. เทคนิค Denoising: กรองสัญญาณรบกวนแต่รักษาเส้นขอบตัวอักษร (Edge-Preserving Denoising)
    if denoise_method == "bilateral":
        # Bilateral Filter ในระบบสี BGR (พิจารณาทั้งระยะห่างพิกเซลและความต่างของสี)
        denoised = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlmeans":
        # Fast Non-Local Means Denoising สำหรับภาพสี
        denoised = cv2.fastNlMeansDenoisingColored(
            image_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
    elif denoise_method == "gaussian":
        denoised = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    else:
        denoised = image_bgr

    # 2. เทคนิค CLAHE (Contrast Limited Adaptive Histogram Equalization):
    # เพื่อรักษาข้อมูลสีเดิมและไม่ทำให้สีเพี้ยน เราจะแปลงเป็นระบบสี LAB
    # และทำ CLAHE เฉพาะช่องความสว่าง (L - Lightness Channel) เท่านั้น
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # รวมช่องสี LAB กลับคืนและแปลงกลับเป็น BGR
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def rectify_license_plate(
    plate_crop,
    method: str = "homography",
):
    """
    ปรับมุมมองและดึงภาพแผ่นป้ายทะเบียนให้กลับมาเป็นหน้าตรงและตั้งตรง (Plate Rectification / Deskewing)
    โดยใช้เทคนิค Homography (Perspective Transformation) หรือ Affine Transformation

    Args:
        plate_crop: ภาพ BGR ของแผ่นป้ายที่ถูกตัดออกมา (Cropped Plate)
        method:
            - 'homography': คำนวณ Homography Matrix 3x3 (Perspective Transform)
              แก้ปัญหามุมมอง 3 มิติ (กล้องเอียงข้าง/ก้มเงย)
            - 'affine': คำนวณ Affine Matrix 2x3 (Rotation Deskew)
              หมุนปรับระดับองศาในระนาบ 2D ให้ตั้งตรง

    Returns:
        tuple[np.ndarray, np.ndarray | None, float]:
            (ภาพที่ถูกปรับหน้าตรงแล้ว, Transformation Matrix, องศาการเอียงที่ตรวจพบ)
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, None, 0.0

    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)

    # เชื่อมเส้นขอบแผ่นป้ายที่อาจขาดตอน
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, None, 0.0

    # เลือกคอนทัวร์ที่มีพื้นที่ใหญ่ที่สุดซึ่งน่าจะเป็นกรอบแผ่นป้าย
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (w * h * 0.08):
        return plate_crop, None, 0.0

    rect = cv2.minAreaRect(cnt)
    center, (rw, rh), angle = rect

    # ปรับมุมองศาการเอียงของ minAreaRect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0

    if method == "homography":
        # 1. หา 4 จุดมุมด้วย Polygon Approximation หรือ minAreaRect box points
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        if len(approx) == 4:
            src_pts = approx.reshape(4, 2).astype(np.float32)
            src_pts = order_quad_points(src_pts)
        else:
            box = cv2.boxPoints(rect)
            src_pts = order_quad_points(box)

        (tl, tr, br, bl) = src_pts

        # คำนวณความกว้างและความสูงเป้าหมาย
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        max_w = int(max(w_top, w_bot))

        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        max_h = int(max(h_left, h_right))

        if max_w < 30 or max_h < 30:
            return plate_crop, None, angle

        # พิกัดสี่เหลี่ยมหน้าตรงเป้าหมาย (Destination Points)
        dst_pts = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        # คำนวณ Homography Matrix (Perspective Transform 3x3)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            plate_crop,
            H,
            (max_w, max_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # หากขนาดกลายเป็นแนวตั้ง ให้หมุนปรับเป็นแนวนอน
        if max_h > max_w * 1.3:
            rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

        return rectified, H, angle

    elif method == "affine":
        # คำนวณ Affine Transformation Matrix 2x3 เพื่อหมุนแก้การเอียง (Rotation Deskew)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        rectified = cv2.warpAffine(
            plate_crop,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rectified, M, angle

    else:
        raise ValueError(
            f"Unknown rectify method: '{method}'. Choose 'homography' or 'affine'."
        )


def binarize_license_plate(
    plate_crop,
    method: str = "adaptive",
    clean_noise: bool = True,
    invert: bool = False,
):
    """
    แปลงภาพแผ่นป้ายทะเบียนเป็นภาพขาว-ดำ (Binarization)
    เพื่อตัดรายละเอียดของพื้นหลัง (แสงสะท้อน, ลายกราฟิกป้ายประมูล, เงา)
    ให้เหลือเฉพาะตัวอักษรและตัวเลขที่คมชัด

    Args:
        plate_crop: ภาพ BGR ของแผ่นป้ายที่ถูกตัดออกมา (Cropped Plate)
        method: วิธีการ Binarization:
            - 'adaptive': Adaptive Gaussian Thresholding + CLAHE + Bilateral Filter (แนะนำ: รองรับแสงเงาไม่สม่ำเสมอได้ดีที่สุด)
            - 'blackhat': Morphological Black-Hat + Otsu (เหมาะมากสำหรับป้ายทะเบียนลายประมูล/พื้นหลังมีกราฟิก)
            - 'otsu': Otsu Thresholding แบบคลาสสิก
        clean_noise: ทำ Morphological Opening เพื่อกำจัดจุด Noise เม็ดทราย/หัวน็อตเล็กๆ
        invert: สลับสี (True: พื้นหลังดำ ตัวอักษรขาว / False: พื้นหลังขาว ตัวอักษรดำ)

    Returns:
        np.ndarray: ภาพขาว-ดำ (Single channel, uint8)
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    # 1. แปลงเป็นภาพระดับสีเทา (Grayscale)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. ปรับปรุงคอนทราสต์เฉพาะจุดด้วย CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ช่วยดึงรายละเอียดตัวอักษรในบริเวณที่มืดหรือสว่างจ้าเกินไป
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. กรองสัญญาณรบกวนด้วย Bilateral Filter
    # รักษาขอบตัวอักษรให้คมชัด โดยไม่ทำให้เส้นตัวหนังสือเบลอเหมือน Gaussian ทั่วไป
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    if method == "adaptive":
        # Adaptive Gaussian Threshold คำนวณขีดแบ่งแสงแยกตามพื้นที่ย่อย (Local Neighborhood)
        # ตัวอักษรสีดำ (0) บนพื้นขาว (255)
        binarized = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=19,
            C=8,
        )

    elif method == "blackhat":
        # Morphological Black-Hat: Closing(I) - I
        # สกัดเฉพาะองค์ประกอบที่เข้มกว่าพื้นหลังรอบข้าง (ตัวอักษร) ตัดลายกราฟิกพื้นหลังออก
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(
            blackhat_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binarized = cv2.bitwise_not(thresh)  # พื้นหลังขาว ตัวอักษรดำ

    elif method == "otsu":
        # Otsu's Global Thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    else:
        raise ValueError(
            f"Unknown binarization method: '{method}'. Choose from: 'adaptive', 'blackhat', 'otsu'."
        )

    # 4. กำจัดเม็ด Noise เล็กๆ ด้วย Morphological Opening
    if clean_noise and method in ["adaptive", "otsu"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)

    # 5. สลับสีหากต้องการตัวอักษรขาวบนพื้นหลังดำ
    if invert:
        binarized = cv2.bitwise_not(binarized)

    return binarized


def extract_character_edges(
    plate_crop,
    method: str = "canny",
    low_thresh: int = 50,
    high_thresh: int = 150,
):
    """
    สกัดเฉพาะ 'ขอบของตัวอักษร' (Character Edges / Outlines) ออกมาเป็นภาพขาว-ดำ
    ตัดพื้นผิวและรายละเอียดพื้นหลังทั้งหมดออก เหลือเพียงเส้นขอบโครงสร้างตัวหนังสือ

    Args:
        plate_crop: ภาพ BGR ของแผ่นป้ายทะเบียน
        method: 'canny' หรือ 'morph_gradient'
        low_thresh: Hysteresis threshold ล่างสำหรับ Canny
        high_thresh: Hysteresis threshold บนสำหรับ Canny

    Returns:
        np.ndarray: ภาพขาว-ดำ แสดงเส้นขอบตัวอักษร (ขอบสีดำบนพื้นหลังสีขาว)
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    if method == "canny":
        edges = cv2.Canny(denoised, low_thresh, high_thresh)
        return cv2.bitwise_not(edges)  # ขอบสีดำ บนพื้นขาว

    elif method == "morph_gradient":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(thresh)

    else:
        raise ValueError(
            f"Unknown edge method: '{method}'. Choose 'canny' or 'morph_gradient'."
        )


def compute_iou(box1: tuple, box2: tuple) -> float:
    """คำนวณพื้นที่ทับซ้อน 2 มิติ (2D Intersection over Union) ระหว่าง 2 กล่อง (x1, y1, x2, y2)"""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_x_overlap(box1: tuple, box2: tuple) -> float:
    """คำนวณสัดส่วนการทับซ้อนในแนวนอน (1D Horizontal Overlap) เทียบกับกล่องที่แคบกว่า"""
    ix1, ix2 = max(box1[0], box2[0]), min(box1[2], box2[2])
    iw = max(0.0, ix2 - ix1)
    min_w = min(box1[2] - box1[0], box2[2] - box2[0])
    return iw / min_w if min_w > 0 else 0.0


def inter_class_nms(candidates: list, iou_thresh: float = 0.30, x_overlap_thresh: float = 0.45) -> list:
    """
    Inter-class Non-Maximum Suppression (IoU + 1D Horizontal Overlap Suppression)
    ตัดกล่องที่ทำนายทับซ้อนกันในตำแหน่งตัวอักษรเดียวกันข้ามคลาส โดยเลือกกล่องที่ Confidence สูงสุด
    """
    sorted_cands = sorted(candidates, key=lambda c: c[9], reverse=True)
    kept = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        kept.append(best)
        b_box = (best[0], best[1], best[2], best[3])
        rem = []
        for cand in sorted_cands:
            c_box = (cand[0], cand[1], cand[2], cand[3])
            iou = compute_iou(b_box, c_box)
            x_ov = compute_x_overlap(b_box, c_box)
            if iou > iou_thresh or x_ov > x_overlap_thresh:
                continue  # ถูก Suppress ตัดทิ้ง
            rem.append(cand)
        sorted_cands = rem
    return kept


def draw_annotated_chars(image_bgr, kept_boxes: list, thai_font_path=None):
    """วาด Bounding Box ผลลัพธ์เฉพาะตัวอักษรที่ผ่าน Inter-class NMS แล้ว"""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    if thai_font_path and Path(thai_font_path).is_file():
        try:
            font = ImageFont.truetype(str(thai_font_path), 18)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    for b in kept_boxes:
        bx1, by1, bx2, by2, char_text, conf = b
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 0), width=2)
        # แถบพื้นหลังข้อความ
        label_h = 22
        draw.rectangle([bx1, max(0, by1 - label_h), bx1 + 22, max(0, by1)], fill=(0, 255, 0))
        draw.text((bx1 + 3, max(0, by1 - label_h)), char_text, fill=(0, 0, 0), font=font)

    return pil_img


def parse_thai_license_syntax(char_model, plate_crop, conf_thresh: float = 0.08, device: str = "0"):
    """
    ตรวจจับและถอดรหัสตัวอักษรบนแผ่นป้ายทะเบียนไทยตามหลักไวยากรณ์ (Syntax Validation)
    ผสาน Inter-class NMS, IoU Suppression และ Slot Clustering
    - Pattern A: [1 เลข] [1-2 พยัญชนะ] [1-4 เลข] เช่น 2ขต 921, 7กล 9108, 4ต 444
    - Pattern B: [2 พยัญชนะ] [1-4 เลข] เช่น ฒก 8534, กก 1234, ดษ 4999, สส 6588
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

        # แยกกล่องจังหวัด (อยู่ครึ่งล่าง หรือมีชื่อคลาสเป็น Province)
        if cls_name == "Province" or cy > crop_h * 0.58:
            px1 = max(0, int(bx1) - 3)
            py1 = max(0, int(by1) - 2)
            px2 = min(crop_w, int(bx2) + 3)
            py2 = min(crop_h, int(by2) + 2)
            if py2 > py1 and px2 > px1:
                province_crop = plate_crop[py1:py2, px1:px2]
                province_conf = max(province_conf, conf)
            continue

        # กรองสัญญาณรบกวนขอบป้ายหรือหัวน็อต (ขนาดเล็ก/สูงผิดปกติ)
        if bh < crop_h * 0.28 or bh > crop_h * 0.95:
            continue
        # กรองเส้นขอบกรอบป้ายด้านนอก (ชิดขอบซ้าย/ขวามากเกินไป)
        if (bx1 < crop_w * 0.02 or bx2 > crop_w * 0.98) and (bw / bh < 0.25 or conf < 0.20):
            continue

        raw_candidates.append((bx1, by1, bx2, by2, cx, cy, bw, bh, cls_name, conf))

    if not raw_candidates:
        return "", province_crop, province_conf, char_results, []

    # 1. ทำ Inter-class NMS ตัดกล่องซ้อนทับกันข้ามคลาส
    nms_candidates = inter_class_nms(raw_candidates, iou_thresh=0.30, x_overlap_thresh=0.45)

    # 2. จัดเรียงตัวอักษรจากซ้ายไปขวาตามแกน X
    nms_candidates.sort(key=lambda b: b[4])

    result_chars = []
    kept_boxes = []

    for idx, c in enumerate(nms_candidates):
        cls_name = c[8]
        bw, bh = c[6], c[7]

        # ตรวจสอบความสับสนระหว่างเลข 1 กับ 7
        if cls_name == "Seven" and (bw / bh) < 0.48:
            cls_name = "One"

        char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
        result_chars.append(char_text)
        kept_boxes.append((c[0], c[1], c[2], c[3], char_text, c[9]))

    plate_text = "".join(result_chars)
    return plate_text, province_crop, province_conf, char_results, kept_boxes


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
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="Contrast limit for CLAHE (Default: 2.0)",
    )
    parser.add_argument(
        "--denoise-method",
        choices=["bilateral", "nlmeans", "gaussian", "none"],
        default="bilateral",
        help="Denoising algorithm for plate image ('bilateral', 'nlmeans', 'gaussian', 'none', Default: bilateral)",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable CLAHE and Denoising enhancement preprocessing",
    )
    parser.add_argument(
        "--binarize-method",
        choices=["adaptive", "blackhat", "otsu"],
        default="adaptive",
        help="Binarization method to eliminate background details ('adaptive', 'blackhat', 'otsu', Default: adaptive)",
    )
    parser.add_argument(
        "--no-binarize",
        action="store_true",
        help="Disable license plate binarization",
    )
    parser.add_argument(
        "--rectify-method",
        choices=["homography", "affine"],
        default="homography",
        help="Rectification method to frontalize and deskew license plate ('homography' 3x3, 'affine' 2x3, Default: homography)",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Disable license plate rectification (frontal deskewing)",
    )
    parser.add_argument(
        "--no-rectified-for-char",
        action="store_true",
        help="Do not pass rectified/enhanced plate into character detection stage (Default: False, pass rectified)",
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
    print(f"📷 Image:           {image_path.name} ({width}x{height})")
    print(f"🎯 Plate Model:     {plate_model_path.name}")
    print(f"🔤 Char Model:      {char_model_path.name}")
    print(f"⚙️ Device:          GPU ({args.device})")
    print(f"✨ Enhancement:     {'Disabled' if args.no_enhance else f'CLAHE(clip={args.clahe_clip}) + Denoise({args.denoise_method})'}")
    print(f"📐 Rectification:   {'Disabled' if args.no_rectify else args.rectify_method}")
    print(f"🌓 Binarization:    {'Disabled' if args.no_binarize else args.binarize_method}")
    print("=" * 60)

    # Initialize models
    plate_model = YOLO(str(plate_model_path))
    char_model = YOLO(str(char_model_path))
    
    ocr_reader = None
    try:
        easyocr_dir = PROJECT_DIR / "weights" / "easyocr"
        easyocr_dir.mkdir(parents=True, exist_ok=True)
        ocr_reader = easyocr.Reader(
            ["th"],
            gpu=(args.device != "cpu"),
            model_storage_directory=str(easyocr_dir),
            user_network_directory=str(easyocr_dir),
        )
    except Exception:
        ocr_reader = None

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

        # 1. บันทึกภาพตัดแผ่นป้ายต้นฉบับ
        crop_raw_path = output_dir / f"plate_{idx}_crop.jpg"
        cv2.imwrite(str(crop_raw_path), plate_crop)

        # 2. เทคนิค CLAHE + Denoising: ปรับปรุงคอนทราสต์เฉพาะจุดและลดสัญญาณรบกวน
        enhanced_plate = plate_crop
        enhanced_path = None
        if not args.no_enhance:
            enhanced_plate = enhance_clahe_denoise(
                plate_crop,
                clip_limit=args.clahe_clip,
                denoise_method=args.denoise_method,
            )
            enhanced_path = output_dir / f"plate_{idx}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), enhanced_plate)

        # 3. เทคนิค Rectification: ดึงภาพป้ายให้กลับมาเป็นหน้าตรงและตั้งตรง (Homography / Affine)
        rectified_plate = enhanced_plate
        rect_path = None
        detect_angle = 0.0
        if not args.no_rectify:
            rectified_plate, transform_mat, detect_angle = rectify_license_plate(
                enhanced_plate, method=args.rectify_method
            )
            rect_path = output_dir / f"plate_{idx}_rectified.jpg"
            cv2.imwrite(str(rect_path), rectified_plate)

        # 4. เทคนิค Binarization ตัดรายละเอียดพื้นหลัง (ทำบนภาพที่ผ่าน Enhancement และ Rectification)
        base_for_bin = rectified_plate if not args.no_rectify else enhanced_plate
        bin_path = None
        edge_path = None
        if not args.no_binarize:
            binarized_plate = binarize_license_plate(base_for_bin, method=args.binarize_method)
            bin_path = output_dir / f"plate_{idx}_binarized.jpg"
            cv2.imwrite(str(bin_path), binarized_plate)

            character_edges = extract_character_edges(base_for_bin, method="canny")
            edge_path = output_dir / f"plate_{idx}_edges.jpg"
            cv2.imwrite(str(edge_path), character_edges)

        # Stage 2: Character Detection with Thai Syntax Validation & Inter-class NMS
        if args.no_rectified_for_char:
            char_input = plate_crop
        else:
            char_input = rectified_plate if not args.no_rectify else (enhanced_plate if not args.no_enhance else plate_crop)

        plate_text, province_crop, province_conf, char_results, kept_boxes = parse_thai_license_syntax(
            char_model=char_model,
            plate_crop=char_input,
            conf_thresh=args.conf,
            device=args.device,
        )

        # Read actual Province Name using OCR on the exact Province Crop
        province_name = "ไม่พบกล่องจังหวัด"
        if province_crop is not None and province_crop.size > 0:
            if ocr_reader is not None:
                try:
                    ph, pw = province_crop.shape[:2]
                    p_resized = cv2.resize(province_crop, (pw * 2, ph * 2), interpolation=cv2.INTER_CUBIC)
                    res = ocr_reader.readtext(p_resized)
                    raw_prov = "".join(r[1] for r in res)
                    province_name = match_province(raw_prov)
                except Exception:
                    province_name = "เกิดข้อผิดพลาดในการอ่าน OCR"
            else:
                province_name = "(EasyOCR โมเดลยังไม่ได้ดาวน์โหลด)"

        print(f"\n🏷️  Plate #{idx} (Detection Confidence: {plate_conf:.0%}):")
        print(f"   ► ภาพตัดป้าย (Raw):       {crop_raw_path}")
        if enhanced_path:
            print(f"   ► CLAHE + Denoise:        {enhanced_path} ({args.denoise_method}, Clip: {args.clahe_clip})")
        if rect_path:
            m_name = "Homography (Perspective 3x3)" if args.rectify_method == "homography" else "Affine (Rotation 2x3)"
            print(f"   ► ปรับหน้าตรง (Rectified): {rect_path} ({m_name}, Tilt: {detect_angle:+.1f}°)")
        if bin_path:
            print(f"   ► ขาว-ดำ (Binarized):     {bin_path} (Method: {args.binarize_method})")
            print(f"   ► ขอบตัวอักษร (Edges):     {edge_path}")
        print(f"   ► หมวดเลขทะเบียน:         {plate_text if plate_text else '(อ่านตัวเลข/พยัญชนะไม่ได้)'}")
        print(f"   ► จังหวัด:                 {province_name}")

        # Save annotated crop with clean NMS kept boxes
        thai_font = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "tahoma.ttf"
        crop_annotated = draw_annotated_chars(
            char_input, kept_boxes, thai_font_path=str(thai_font) if thai_font.is_file() else None
        )
        crop_path = output_dir / f"plate_{idx}_annotated.jpg"
        crop_annotated.save(str(crop_path))
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
