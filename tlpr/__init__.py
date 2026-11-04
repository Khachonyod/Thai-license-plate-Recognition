"""
TLPR: Thai License Plate Recognition
Modular package for 2-stage YOLO-based Thai license plate recognition.
"""

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
from tlpr.pipeline import TLPRPipeline
from tlpr.rectification import order_quad_points, rectify_license_plate
from tlpr.syntax import match_province, parse_thai_license_syntax
from tlpr.visualization import draw_annotated_chars

__version__ = "1.0.0"

__all__ = [
    "TLPRPipeline",
    "enhance_clahe_denoise",
    "binarize_license_plate",
    "extract_character_edges",
    "rectify_license_plate",
    "order_quad_points",
    "compute_iou",
    "compute_x_overlap",
    "inter_class_nms",
    "parse_thai_license_syntax",
    "match_province",
    "draw_annotated_chars",
    "CLASS_NAME_TO_CHAR",
    "DIGIT_NAMES",
    "THAI_TO_DIGIT_CONFUSION",
    "DIGIT_TO_THAI_CONFUSION",
    "PROVINCES_77",
    "DEFAULT_PLATE_MODEL",
    "DEFAULT_CHAR_MODEL",
    "PROJECT_DIR",
]

