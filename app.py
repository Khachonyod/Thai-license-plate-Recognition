"""
TLPR: Thai License Plate Recognition System
CLI Application and Backward-Compatible Interface
"""

import argparse
from pathlib import Path
import cv2

# Re-export all core components for 100% backward compatibility
from tlpr.constants import (
    CLASS_NAME_TO_CHAR,
    DEFAULT_CHAR_MODEL,
    DEFAULT_PLATE_MODEL,
    DIGIT_NAMES,
    DIGIT_TO_THAI_CONFUSION,
    PROJECT_DIR,
    PROVINCES_77,
    THAI_TO_DIGIT_CONFUSION,
)
from tlpr.enhancement import (
    binarize_license_plate,
    enhance_clahe_denoise,
    extract_character_edges,
)
from tlpr.geometry import compute_iou, compute_x_overlap, inter_class_nms
from tlpr.pipeline import TLPRPipeline, TLPRResult, PlateDetection
from tlpr.rectification import order_quad_points, rectify_license_plate
from tlpr.syntax import match_province, parse_thai_license_syntax
from tlpr.visualization import draw_annotated_chars


def get_default_sample_image() -> str:
    """ค้นหาไฟล์ภาพตัวอย่างเริ่มต้น (samples/test.jpg หรือ test.jpg)"""
    candidate_paths = [
        PROJECT_DIR / "samples" / "test.jpg",
        PROJECT_DIR / "test.jpg",
        PROJECT_DIR / "samples" / "test1.jpg",
        PROJECT_DIR / "test1.jpg",
    ]
    for p in candidate_paths:
        if p.is_file():
            return str(p)
    return str(PROJECT_DIR / "test.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-Stage End-to-End YOLO Thai License Plate Recognition System (Syntax Validated)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=get_default_sample_image(),
        help="Path to image file (Default: samples/test.jpg or test.jpg)",
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
        "--use-rectified-for-char",
        action="store_true",
        help="Pass rectified plate into character detection stage (Default: False, pass raw plate crop to preserve stroke details)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable EasyOCR for province detection",
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

    # Output directory
    output_dir = PROJECT_DIR / "runs" / "lpr_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 Thai License Plate Recognition (Syntax Validated Pipeline)")
    print(f"📷 Image:           {image_path.name}")
    print(f"🎯 Plate Model:     {plate_model_path.name}")
    print(f"🔤 Char Model:      {char_model_path.name}")
    print(f"⚙️ Device:          {args.device}")
    print(f"✨ Enhancement:     {'Disabled' if args.no_enhance else f'CLAHE(clip={args.clahe_clip}) + Denoise({args.denoise_method})'}")
    print(f"📐 Rectification:   {'Disabled' if args.no_rectify else args.rectify_method}")
    print(f"🌓 Binarization:    {'Disabled' if args.no_binarize else args.binarize_method}")
    print("=" * 60)

    # Initialize Pipeline
    pipeline = TLPRPipeline(
        plate_model_path=plate_model_path,
        char_model_path=char_model_path,
        device=args.device,
        enable_ocr=(not args.no_ocr),
    )

    result = pipeline.process(
        image_or_path=image_path,
        conf=args.conf,
        enable_enhancement=(not args.no_enhance),
        clahe_clip=args.clahe_clip,
        denoise_method=args.denoise_method,
        enable_rectification=(not args.no_rectify),
        rectify_method=args.rectify_method,
        enable_binarization=(not args.no_binarize),
        binarize_method=args.binarize_method,
        use_rectified_for_char=args.use_rectified_for_char,
    )

    if not result.plates:
        print("\n❌ Not Found License Plate in this image.")
        return

    for plate in result.plates:
        idx = plate.plate_id
        # Save crops
        crop_raw_path = output_dir / f"plate_{idx}_crop.jpg"
        if plate.raw_crop is not None:
            cv2.imwrite(str(crop_raw_path), plate.raw_crop)

        enhanced_path = None
        if plate.enhanced_crop is not None and not args.no_enhance:
            enhanced_path = output_dir / f"plate_{idx}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), plate.enhanced_crop)

        rect_path = None
        if plate.rectified_crop is not None and not args.no_rectify:
            rect_path = output_dir / f"plate_{idx}_rectified.jpg"
            cv2.imwrite(str(rect_path), plate.rectified_crop)

        bin_path = None
        edge_path = None
        if plate.binarized_crop is not None and not args.no_binarize:
            bin_path = output_dir / f"plate_{idx}_binarized.jpg"
            cv2.imwrite(str(bin_path), plate.binarized_crop)

            if plate.edges_crop is not None:
                edge_path = output_dir / f"plate_{idx}_edges.jpg"
                cv2.imwrite(str(edge_path), plate.edges_crop)

        annotated_path = output_dir / f"plate_{idx}_annotated.jpg"
        if plate.annotated_crop is not None:
            plate.annotated_crop.save(str(annotated_path))

        print(f"\n🏷️  Plate #{idx} (Detection Confidence: {plate.confidence:.0%}):")
        print(f"   ► ภาพตัดป้าย (Raw):       {crop_raw_path}")
        if enhanced_path:
            print(f"   ► CLAHE + Denoise:        {enhanced_path} ({args.denoise_method}, Clip: {args.clahe_clip})")
        if rect_path:
            m_name = "Homography (Perspective 3x3)" if args.rectify_method == "homography" else "Affine (Rotation 2x3)"
            print(f"   ► ปรับหน้าตรง (Rectified): {rect_path} ({m_name}, Tilt: {plate.tilt_angle:+.1f}°)")
        if bin_path:
            print(f"   ► ขาว-ดำ (Binarized):     {bin_path} (Method: {args.binarize_method})")
            if edge_path:
                print(f"   ► ขอบตัวอักษร (Edges):     {edge_path}")
        print(f"   ► หมวดเลขทะเบียน:         {plate.plate_text if plate.plate_text else '(อ่านตัวเลข/พยัญชนะไม่ได้)'}")
        if plate.kept_boxes:
            conf_details = " | ".join(f"{ch}: {c:.1%}" for _, _, _, _, ch, c in plate.kept_boxes)
            print(f"   ► ความมั่นใจรายตัว:       {conf_details}")
        print(f"   ► จังหวัด:                 {plate.province_name}")
        print(f"   ► Crop ผลลัพธ์:     {annotated_path}")

    if result.annotated_vehicle is not None:
        result_image_path = output_dir / "detections.jpg"
        cv2.imwrite(str(result_image_path), result.annotated_vehicle)
        print(f"\n🖼️  ภาพรวมผลตรวจจับทั้งหมด: {result_image_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
