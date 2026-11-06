# TLPR: Thai License Plate Recognition

ระบบตรวจจับและอ่านป้ายทะเบียนรถยนต์ไทยแบบ End-to-End อัตโนมัติ (Two-Stage YOLO Architecture + Syntax Validator)

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **Stage 1: Plate Detector (`weights/plate_detector.pt`)**
   - ตรวจจับตำแหน่งแผ่นป้ายทะเบียนบนตัวรถอย่างแม่นยำ พร้อมทำ Crop อัตโนมัติ
2. **Image Enhancement: CLAHE & Edge-Preserving Denoising**
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** ปรับคอนทราสต์เฉพาะจุดบนระบบสี LAB (Luminance Channel) ดึงความคมชัดของตัวอักษรแม้มีแสงสะท้อนหรือเงาตกกระทบโดยไม่ทำให้สีเพี้ยน
   - **Denoising:** กรองและลบสัญญาณรบกวน (Noise) รองรับ `bilateral` (รักษาขอบตัวอักษรได้คมชัดสูง), `nlmeans` (Non-Local Means ลด Noise ขั้นสูงสำหรับภาพสี), และ `gaussian`
3. **Plate Rectification: ดึงภาพป้ายให้เป็นหน้าตรงและตั้งตรง (Homography & Affine)**
   - **Homography Matrix (3x3 Perspective Transform):** หา 4 จุดมุมของกรอบป้ายและดึงภาพมุมกล้องเฉียง/ก้มเงย ให้กลับมาเป็นมุมมองหน้าตรง (Frontal View)
   - **Affine Transformation (2x3 Rotation Deskew):** หมุนปรับระดับองศาการเอียงของแผ่นป้ายให้อยู่ในแนวระนาบตั้งตรง
4. **License Plate Binarization & Character Edge Extraction (เทคนิคประมวลผลภาพขาว-ดำ)**
   - แปลงภาพ Crop ของแผ่นป้ายเป็นขาว-ดำ (Binarization) เพื่อตัดลายกราฟิก (เช่น ป้ายประมูล), แสงสะท้อน และเงา
   - รองรับ 3 วิธี: `adaptive` (Adaptive Gaussian + CLAHE + Bilateral Filter), `blackhat` (Morphological Black-Hat + Otsu สำหรับป้ายกราฟิก), และ `otsu`
   - สกัดเส้นขอบตัวอักษรเฉพาะจุด (Character Edges) บันทึกเป็นภาพแยกเพื่อให้เห็นโครงสร้างลายเส้นชัดเจน
5. **Stage 2: Character Detector (`weights/cChar_detector.pt`)**
   - ตรวจจับตัวอักษรและตัวเลข 55 คลาส (พยัญชนะไทย 44 ตัว, ตัวเลข 0-9, และกรอบข้อความจังหวัด)
6. **Thai Syntax Validator & Slot Clustering**
   - ระบบจัดกลุ่ม Slot ตามโครงสร้างไวยากรณ์ป้ายทะเบียนไทย (เช่น หมวด `1กข 1234` หรือ `กข 1234`)
   - แก้ปัญหาความกำกวมของตัวอักษร เช่น แยกแยะเลข `1` กับเลข `7` ตาม Slot ตำแหน่ง และ Confusion Fallback
7. **Province Recognition**
   - ทำ OCR เฉพาะกรอบกล่องจังหวัด และจับคู่ความใกล้เคียงกับฐานข้อมูล 77 จังหวัด (Fuzzy String Matching)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
TLPR/
├── tlpr/                             # Core Library Module
│   ├── __init__.py                   # Package exports
│   ├── constants.py                  # ค่าคงที่คลาสอักษรไทย, ตัวเลข, รายชื่อ 77 จังหวัด
│   ├── enhancement.py                # CLAHE, Denoising, Binarization, Edge Extraction
│   ├── rectification.py              # Homography & Affine Perspective Transform
│   ├── geometry.py                   # IoU, X-overlap, Inter-class NMS
│   ├── syntax.py                     # Syntax Validator, Slot Clustering, Province Matcher
│   ├── visualization.py              # วาด Bounding Box ตัวอักษรพร้อมฟอนต์ภาษาไทย
│   └── pipeline.py                   # End-to-End TLPRPipeline Class
├── samples/                          # ตัวอย่างภาพทดสอบ (test.jpg - test5.jpg)
├── weights/                          # Model Weights พร้อมใช้งาน
│   ├── plate_detector.pt             # โมเดลตรวจจับป้ายทะเบียน (Stage 1)
│   └── cChar_detector.pt             # โมเดลอ่านตัวอักษร/ตัวเลข (Stage 2)
├── scripts/                          # สคริปต์ช่วยงาน Dataset และ Labeling
│   ├── auto_label_snowball.py        # สคริปต์ทำ Auto-labeling สำหรับเพิ่มข้อมูลภาพ
│   └── download_dataset.py           # สคริปต์ดาวน์โหลดชุดข้อมูลจาก Roboflow
├── app.py                            # CLI หลักสำหรับรันระบบ LPR Pipeline (Backward-Compatible)
├── run_test.py                       # CLI Benchmark ประเมินความถูกต้องของโมเดล
├── run_test_gui.py                   # แอป GUI สำหรับตรวจทานความถูกต้องรายภาพ
├── requirements.txt                  # รายการ Library ที่ต้องติดตั้ง
├── .gitignore                        # กรองขยะและ Dataset ขนาดใหญ่
└── README.md                         # เอกสารอธิบายการใช้งาน
```

---

## 🚀 การติดตั้งและใช้งาน (Installation & Quick Start)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันโปรแกรมทดสอบอ่านป้ายทะเบียน

#### 🖥️ ใช้งานผ่านโปรแกรม GUI (TLPR Studio) - รองรับทั้งภาพและวิดีโอคลิป
```bash
# เปิดใช้งานหน้าต่าง GUI Studio แบบเต็มรูปแบบ
python tlpr_gui.py

# หรือเปิดผ่าน app.py ด้วย flag --gui
python app.py --gui
```
> **ฟังก์ชันใน TLPR Studio GUI:**
> - นำเข้าภาพนิ่ง (`.jpg`, `.png`, `.webp`, ฯลฯ) หรือคลิปวิดีโอ (`.mp4`, `.avi`, `.mov`, ฯลฯ)
> - ควบคุมการเล่นวิดีโอ (Play / Pause / Stop / Seek Bar / ข้ามเฟรม Frame Skip)
> - ปรับแต่ง Pre-processing แบบ Real-Time (CLAHE, Denoising, Rectification, Binarization)
> - แสดงผลขั้นตอน Pre-processing แต่ละสเตจ (Raw Crop, Enhanced, Rectified, Binarized, Chars)
> - บันทึกภาพ Snapshot หรืออัดคลิปวิดีโอผลลัพธ์เป็นไฟล์ MP4

#### ⚙️ ใช้งานผ่าน Command Line (CLI)
```bash
# ทดสอบอ่านป้ายทะเบียนจากภาพตัวอย่าง (ใช้ samples/test.jpg เป็นค่าเริ่มต้น)
python app.py

# ระบุภาพที่ต้องการทดสอบ
python app.py samples/test.jpg

# กำหนดเทคนิค CLAHE และ Denoising
python app.py samples/test.jpg --denoise-method bilateral --clahe-clip 2.5
python app.py samples/test.jpg --denoise-method nlmeans
python app.py samples/test.jpg --no-enhance

# เลือกเทคนิคการดึงภาพหน้าตรง (homography หรือ affine) หรือปิดการ Rectify
python app.py samples/test.jpg --rectify-method homography
python app.py samples/test.jpg --rectify-method affine
python app.py samples/test.jpg --no-rectify

# กำหนดวิธี Binarization (adaptive, blackhat, otsu) หรือปิดการแปลง
python app.py samples/test.jpg --binarize-method blackhat
python app.py samples/test.jpg --no-binarize

# กำหนดระดับ Confidence Threshold และ Device
python app.py samples/test.jpg --conf 0.10 --device 0
```

### 3. เรียกใช้งานผ่าน Python Code (Modular SDK)
```python
from tlpr import TLPRPipeline

# โหลด Pipeline
pipeline = TLPRPipeline(device="0")

# ประมวลผลภาพ
result = pipeline.process("samples/test.jpg")

for plate in result.plates:
    print(f"ป้ายทะเบียน: {plate.plate_text}")
    print(f"จังหวัด:     {plate.province_name}")
    print(f"ความมั่นใจ:   {plate.confidence:.1%}")
```

### 4. ทดสอบและประเมินผลโมเดล (Model Testing & Verification)

```bash
# 🖥️ เปิดแอปพลิเคชัน GUI สำหรับตรวจทานด้วยตนเองทีละภาพ (Human-in-the-loop Verification)
python run_test_gui.py

# ⚡ รัน Benchmark อัตโนมัติทาง Command-line พร้อมบันทึก Log ลง run_test_log/
python run_test.py --count 50
python run_test.py --model weights/cChar_detector.pt --count 50 --random-sample
```

### 5. ผลลัพธ์
- ผลการตรวจจับจะถูกบันทึกไว้ที่โฟลเดอร์ `runs/lpr_results/` ประกอบด้วย:
  - `plate_{id}_crop.jpg`: ภาพ Crop แผ่นป้ายทะเบียนต้นฉบับ
  - `plate_{id}_enhanced.jpg`: ภาพที่ผ่านการปรับคอนทราสต์ CLAHE และกรอง Denoising
  - `plate_{id}_rectified.jpg`: ภาพป้ายที่ถูกดึงกลับมาเป็นหน้าตรงและตั้งตรง (Homography/Affine)
  - `plate_{id}_binarized.jpg`: ภาพขาว-ดำหลังทำ Binarization ตัดพื้นหลัง
  - `plate_{id}_edges.jpg`: ภาพโครงร่างเส้นขอบตัวอักษรคมชัด
  - `plate_{id}_annotated.jpg`: ภาพผลการตรวจจับตัวอักษรและ Slot
  - `detections.jpg`: ภาพรวมกรอบทะเบียนบนตัวรถ
- หน้าจอ Terminal จะแสดงผลหมวดเลขทะเบียน และจังหวัดที่อ่านได้ทันที

---

## 🛠️ รายละเอียดโมเดล (Trained Models)

| โมเดล | หน้าที่ | Backbone | Input Size |
| :--- | :--- | :--- | :--- |
| `weights/plate_detector.pt` | ตรวจจับกรอบป้ายทะเบียน (Stage 1) | YOLO | Auto (640x640) |
| `weights/cChar_detector.pt` | ตรวจจับตัวอักษร/ตัวเลข 55 คลาส (Stage 2) | YOLO | Auto (640x640) |
