import cv2
import numpy as np


def enhance_clahe_denoise(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    denoise_method: str = "bilateral",
) -> np.ndarray:
    """
    ปรับปรุงคุณภาพภาพแผ่นป้ายทะเบียนด้วยเทคนิค CLAHE และ Denoising
    เพื่อเพิ่มคอนทราสต์ของตัวอักษร ลดเงา/แสงสะท้อน และลบสัญญาณรบกวน (Noise) โดยรักษาขอบภาพให้คมชัด

    Args:
        image_bgr: ภาพ BGR ของแผ่นป้าย
        clip_limit: ขีดจำกัดคอนทราสต์ของ CLAHE (ป้องกันการขยาย noise เกินไป)
        tile_grid_size: ขนาดบล็อกย่อย (Grid) สำหรับทำ Adaptive Equalization
        denoise_method: วิธีการ Denoising ('bilateral', 'nlmeans', 'gaussian', 'none')

    Returns:
        np.ndarray: ภาพ BGR ที่ผ่านการปรับปรุงคอนทราสต์และลดสัญญาณรบกวนแล้ว
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. Edge-Preserving Denoising
    if denoise_method == "bilateral":
        denoised = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlmeans":
        denoised = cv2.fastNlMeansDenoisingColored(
            image_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
    elif denoise_method == "gaussian":
        denoised = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    else:
        denoised = image_bgr

    # 2. CLAHE on Luminance (L) channel in LAB color space
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def binarize_license_plate(
    plate_crop: np.ndarray,
    method: str = "adaptive",
    clean_noise: bool = True,
    invert: bool = False,
) -> np.ndarray:
    """
    แปลงภาพแผ่นป้ายทะเบียนเป็นภาพขาว-ดำ (Binarization)
    เพื่อตัดรายละเอียดของพื้นหลัง (แสงสะท้อน, ลายกราฟิกป้ายประมูล, เงา)
    ให้เหลือเฉพาะตัวอักษรและตัวเลขที่คมชัด

    Args:
        plate_crop: ภาพ BGR ของแผ่นป้ายที่ถูกตัดออกมา (Cropped Plate)
        method: วิธีการ Binarization:
            - 'adaptive': Adaptive Gaussian Thresholding + CLAHE + Bilateral Filter
            - 'blackhat': Morphological Black-Hat + Otsu (เหมาะสำหรับป้ายกราฟิกประมูล)
            - 'otsu': Otsu Thresholding แบบคลาสสิก
        clean_noise: ทำ Morphological Opening เพื่อกำจัดจุด Noise เม็ดทราย/หัวน็อตเล็กๆ
        invert: สลับสี (True: พื้นหลังดำ ตัวอักษรขาว / False: พื้นหลังขาว ตัวอักษรดำ)

    Returns:
        np.ndarray: ภาพขาว-ดำ (Single channel, uint8)
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    # 1. Grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. Local contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Bilateral filter for edge-preserving noise removal
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    if method == "adaptive":
        binarized = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=19,
            C=8,
        )
    elif method == "blackhat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(
            blackhat_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binarized = cv2.bitwise_not(thresh)
    elif method == "otsu":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(
            f"Unknown binarization method: '{method}'. Choose from: 'adaptive', 'blackhat', 'otsu'."
        )

    # 4. Clean noise with opening
    if clean_noise and method in ["adaptive", "otsu"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)

    # 5. Invert if requested
    if invert:
        binarized = cv2.bitwise_not(binarized)

    return binarized


def extract_character_edges(
    plate_crop: np.ndarray,
    method: str = "canny",
    low_thresh: int = 50,
    high_thresh: int = 150,
) -> np.ndarray:
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
        return cv2.bitwise_not(edges)  # Black edges on white background
    elif method == "morph_gradient":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(thresh)
    else:
        raise ValueError(
            f"Unknown edge method: '{method}'. Choose 'canny' or 'morph_gradient'."
        )

