from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from tlpr.constants import (
    DEFAULT_CHAR_MODEL,
    DEFAULT_PLATE_MODEL,
    PROJECT_DIR,
)
from tlpr.enhancement import (
    binarize_license_plate,
    enhance_clahe_denoise,
    extract_character_edges,
)
from tlpr.rectification import rectify_license_plate
from tlpr.syntax import match_province, parse_thai_license_syntax
from tlpr.visualization import draw_annotated_chars


@dataclass
class PlateDetection:
    """ผลลัพธ์ของแต่ละแผ่นป้ายที่ตรวจจับได้"""
    plate_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    plate_text: str
    province_name: str
    kept_boxes: list = field(default_factory=list)
    raw_crop: Optional[np.ndarray] = None
    enhanced_crop: Optional[np.ndarray] = None
    rectified_crop: Optional[np.ndarray] = None
    binarized_crop: Optional[np.ndarray] = None
    edges_crop: Optional[np.ndarray] = None
    annotated_crop: Optional[Image.Image] = None
    tilt_angle: float = 0.0


@dataclass
class TLPRResult:
    """ผลลัพธ์รวมของภาพยานพาหนะ"""
    image_shape: tuple[int, int]
    plates: list[PlateDetection] = field(default_factory=list)
    annotated_vehicle: Optional[np.ndarray] = None


class TLPRPipeline:
    """
    Two-Stage Thai License Plate Recognition Pipeline:
    Stage 1: Plate Detection (YOLO)
    Preprocessing: CLAHE Enhancement, Edge-Preserving Denoising, Rectification, Binarization
    Stage 2: Character Detection (YOLO) + Thai Syntax Validation & Slot Clustering
    Stage 3: Province OCR (EasyOCR + Fuzzy Matching)
    """

    def __init__(
        self,
        plate_model_path: Optional[Union[str, Path]] = None,
        char_model_path: Optional[Union[str, Path]] = None,
        device: str = "0",
        enable_ocr: bool = True,
    ):
        self.device = str(device)
        self.plate_model_path = Path(plate_model_path or DEFAULT_PLATE_MODEL)
        self.char_model_path = Path(char_model_path or DEFAULT_CHAR_MODEL)

        if not self.plate_model_path.is_file():
            raise FileNotFoundError(f"Plate model file not found: {self.plate_model_path}")
        if not self.char_model_path.is_file():
            raise FileNotFoundError(f"Character model file not found: {self.char_model_path}")

        # Load models
        self.plate_model = YOLO(str(self.plate_model_path))
        self.char_model = YOLO(str(self.char_model_path))

        # EasyOCR reader for province name extraction
        self.ocr_reader = None
        if enable_ocr:
            try:
                import easyocr
                easyocr_dir = PROJECT_DIR / "weights" / "easyocr"
                easyocr_dir.mkdir(parents=True, exist_ok=True)
                self.ocr_reader = easyocr.Reader(
                    ["th"],
                    gpu=(self.device != "cpu"),
                    model_storage_directory=str(easyocr_dir),
                    user_network_directory=str(easyocr_dir),
                    verbose=False,
                )
            except Exception:
                self.ocr_reader = None

    def read_province_ocr(self, province_crop: Optional[np.ndarray]) -> str:
        """ทำ OCR ข้อความในกล่องจังหวัดและจับคู่ Fuzzy Matching กับ 77 จังหวัด"""
        if province_crop is None or province_crop.size == 0:
            return "ไม่พบกล่องจังหวัด"
        if self.ocr_reader is None:
            return "(ไม่ได้เปิดใช้งาน EasyOCR หรือยังไม่ดาวน์โหลดโมเดล)"

        try:
            ph, pw = province_crop.shape[:2]
            p_resized = cv2.resize(province_crop, (pw * 2, ph * 2), interpolation=cv2.INTER_CUBIC)
            res = self.ocr_reader.readtext(p_resized)
            raw_prov = "".join(r[1] for r in res)
            return match_province(raw_prov)
        except Exception:
            return "เกิดข้อผิดพลาดในการอ่าน OCR"

    def process(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        conf: float = 0.08,
        enable_enhancement: bool = True,
        clahe_clip: float = 2.0,
        denoise_method: str = "bilateral",
        enable_rectification: bool = True,
        rectify_method: str = "homography",
        enable_binarization: bool = True,
        binarize_method: str = "adaptive",
        use_rectified_for_char: bool = False,
    ) -> TLPRResult:
        """
        ประมวลผลอ่านป้ายทะเบียนแบบครบวงจร
        """
        if isinstance(image_or_path, (str, Path)):
            img_path = Path(image_or_path)
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
        else:
            image = image_or_path

        height, width = image.shape[:2]
        result = TLPRResult(image_shape=(height, width))

        # Stage 1: Plate Detection
        plate_results = self.plate_model.predict(
            source=image, device=self.device, verbose=False
        )[0]

        if plate_results.boxes is None or len(plate_results.boxes) == 0:
            result.annotated_vehicle = image.copy()
            return result

        annotated_vehicle = image.copy()

        for idx, p_box in enumerate(plate_results.boxes, start=1):
            x1, y1, x2, y2 = map(int, p_box.xyxy[0].tolist())
            plate_conf = float(p_box.conf[0])

            # Padding crop
            pad_x = int((x2 - x1) * 0.03)
            pad_y = int((y2 - y1) * 0.05)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(width, x2 + pad_x)
            cy2 = min(height, y2 + pad_y)

            plate_crop = image[cy1:cy2, cx1:cx2]

            # 1. Image Enhancement
            enhanced_plate = plate_crop
            if enable_enhancement:
                enhanced_plate = enhance_clahe_denoise(
                    plate_crop,
                    clip_limit=clahe_clip,
                    denoise_method=denoise_method,
                )

            # 2. Rectification
            rectified_plate = enhanced_plate
            detect_angle = 0.0
            if enable_rectification:
                rectified_plate, _, detect_angle = rectify_license_plate(
                    enhanced_plate, method=rectify_method
                )

            # 3. Binarization
            binarized_crop = None
            edges_crop = None
            if enable_binarization:
                base_for_bin = rectified_plate if enable_rectification else enhanced_plate
                binarized_crop = binarize_license_plate(base_for_bin, method=binarize_method)
                edges_crop = extract_character_edges(base_for_bin, method="canny")

            # 4. Stage 2: Character Detection with Thai Syntax Validation
            char_input = (
                (rectified_plate if enable_rectification else (enhanced_plate if enable_enhancement else plate_crop))
                if use_rectified_for_char
                else plate_crop
            )

            plate_text, province_crop, province_conf, char_results, kept_boxes = parse_thai_license_syntax(
                char_model=self.char_model,
                plate_crop=char_input,
                conf_thresh=conf,
                device=self.device,
            )

            # 5. Province OCR
            province_name = self.read_province_ocr(province_crop)

            # 6. Annotated Crop
            annotated_crop = draw_annotated_chars(char_input, kept_boxes)

            # Draw on vehicle image
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

            det = PlateDetection(
                plate_id=idx,
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=plate_conf,
                plate_text=plate_text,
                province_name=province_name,
                kept_boxes=kept_boxes,
                raw_crop=plate_crop,
                enhanced_crop=enhanced_plate if enable_enhancement else None,
                rectified_crop=rectified_plate if enable_rectification else None,
                binarized_crop=binarized_crop,
                edges_crop=edges_crop,
                annotated_crop=annotated_crop,
                tilt_angle=detect_angle,
            )
            result.plates.append(det)

        result.annotated_vehicle = annotated_vehicle
        return result

