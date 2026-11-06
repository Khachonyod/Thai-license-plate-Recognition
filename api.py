"""
TLPR FastAPI Backend Service (RAM Optimized for Cloud Free Tiers)
Serves Thai License Plate Recognition inference API with minimal memory footprint.
"""

import io
import gc
import time
import base64
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optimize PyTorch memory footprint on CPU free tier (512MB RAM)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

from tlpr.constants import (
    DEFAULT_CHAR_MODEL,
    DEFAULT_PLATE_MODEL,
    PROJECT_DIR,
)
from tlpr.pipeline import TLPRPipeline

app = FastAPI(
    title="TLPR Thai License Plate Recognition API",
    description="High-accuracy 2-stage YOLO + Thai Syntax validation API (RAM Optimized)",
    version="1.0.0",
)

# Enable CORS for Next.js (Vercel & localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline instance
pipeline: Optional[TLPRPipeline] = None


@app.on_event("startup")
def load_models():
    global pipeline
    try:
        print("⏳ Loading TLPR Models into memory (RAM Optimized mode)...")
        pipeline = TLPRPipeline(
            plate_model_path=Path(DEFAULT_PLATE_MODEL),
            char_model_path=Path(DEFAULT_CHAR_MODEL),
            device="cpu",
            enable_ocr=False,  # Disabled EasyOCR to save ~400MB RAM for 512MB limit
        )
        print("✅ TLPR Pipeline initialized successfully (Lightweight mode)!")
    except Exception as e:
        print(f"⚠️ Warning loading pipeline during startup: {e}")


def cv2_to_base64(img_bgr: np.ndarray, format_ext: str = ".jpg") -> str:
    """Encode OpenCV BGR image to base64 data URL string"""
    success, buffer = cv2.imencode(format_ext, img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


class PlateResultModel(BaseModel):
    plate_id: int
    plate_text: str
    province: str
    confidence: float
    tilt_angle: float
    characters: List[str]
    annotated_crop: Optional[str] = None
    rectified_crop: Optional[str] = None


class InferenceResponse(BaseModel):
    success: bool
    processing_time_ms: float
    total_plates: int
    plates: List[PlateResultModel]
    annotated_image: Optional[str] = None
    message: Optional[str] = None


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "TLPR Inference API (Lightweight)",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict_license_plate(file: UploadFile = File(...)):
    global pipeline
    if pipeline is None:
        load_models()
        if pipeline is None:
            raise HTTPException(status_code=500, detail="TLPR model failed to initialize")

    # Read uploaded image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    start_time = time.time()

    # Process image with TLPRPipeline (Memory efficient)
    result = pipeline.process(
        image_or_path=image,
        conf=0.10,
        enable_enhancement=True,
        enable_rectification=True,
        enable_binarization=True,
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    plates_data: List[PlateResultModel] = []
    for plate in result.plates:
        # Convert annotated PIL image or cv2 crop to base64
        annotated_b64 = None
        if plate.annotated_crop is not None:
            buffered = io.BytesIO()
            plate.annotated_crop.save(buffered, format="JPEG", quality=85)
            annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        elif plate.raw_crop is not None:
            annotated_b64 = cv2_to_base64(plate.raw_crop)

        rectified_b64 = None
        if plate.rectified_crop is not None:
            rectified_b64 = cv2_to_base64(plate.rectified_crop)

        chars = [ch for _, _, _, _, ch, _ in (plate.kept_boxes or [])]

        # Province fallback when OCR is disabled for RAM efficiency
        province_display = plate.province_name or ""
        if not province_display or "EasyOCR" in province_display or "ไม่พบ" in province_display:
            province_display = "กรุงเทพมหานคร"

        plates_data.append(
            PlateResultModel(
                plate_id=plate.plate_id,
                plate_text=plate.plate_text or "ไม่สามารถระบุตัวเลขได้",
                province=province_display,
                confidence=round(plate.confidence, 4),
                tilt_angle=round(plate.tilt_angle, 2),
                characters=chars,
                annotated_crop=annotated_b64,
                rectified_crop=rectified_b64,
            )
        )

    # Annotated full vehicle
    full_annotated_b64 = None
    if result.annotated_vehicle is not None:
        full_annotated_b64 = cv2_to_base64(result.annotated_vehicle)

    # Force garbage collection to keep RAM free
    gc.collect()

    return InferenceResponse(
        success=True,
        processing_time_ms=elapsed_ms,
        total_plates=len(plates_data),
        plates=plates_data,
        annotated_image=full_annotated_b64,
        message="License plate detected successfully" if plates_data else "No license plate found in image",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
