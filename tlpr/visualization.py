import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_default_thai_font(size: int = 18) -> ImageFont.ImageFont:
    """พยายามค้นหาและโหลดฟอนต์ภาษาไทยจากระบบ (Windows/Linux/Mac) หรือโหลดฟอนต์เริ่มต้น"""
    candidate_paths = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "tahoma.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "cordia.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "angsana.ttf",
        Path("/usr/share/fonts/truetype/thai/Garuda.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Thonburi.ttc"),
    ]
    for p in candidate_paths:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_annotated_chars(
    image_bgr: np.ndarray,
    kept_boxes: list,
    thai_font_path: Optional[str] = None,
) -> Image.Image:
    """วาด Bounding Box ผลลัพธ์เฉพาะตัวอักษรที่ผ่าน Inter-class NMS แล้ว"""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    if thai_font_path and Path(thai_font_path).is_file():
        try:
            font = ImageFont.truetype(str(thai_font_path), 18)
        except Exception:
            font = get_default_thai_font(18)
    else:
        font = get_default_thai_font(18)

    for b in kept_boxes:
        bx1, by1, bx2, by2, char_text, conf = b
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 0), width=2)
        # แถบพื้นหลังข้อความแสดงตัวอักษรและค่า Confidence
        label_text = f"{char_text} {conf:.0%}"
        label_w = max(24, int(len(label_text) * 11))
        label_h = 22
        draw.rectangle([bx1, max(0, by1 - label_h), bx1 + label_w, max(0, by1)], fill=(0, 255, 0))
        draw.text((bx1 + 3, max(0, by1 - label_h)), label_text, fill=(0, 0, 0), font=font)

    return pil_img

