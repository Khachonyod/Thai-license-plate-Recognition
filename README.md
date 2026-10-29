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
5. **Stage 2: Character Detector (`weights/char_detector.pt`)**
   - ตรวจจับตัวอักษรและตัวเลข 55 คลาส (พยัญชนะไทย 44 ตัว, ตัวเลข 0-9, และกรอบข้อความจังหวัด)
6. **Thai Syntax Validator & Slot Clustering**
   - ระบบจัดกลุ่ม Slot ตามโครงสร้างไวยากรณ์ป้ายทะเบียนไทย (เช่น หมวด `1กข 1234` หรือ `กข 1234`)
   - แก้ปัญหาความกำกวมของตัวอักษร เช่น แยกแยะเลข `1` กับเลข `7` ตาม Slot ตำแหน่ง
7. **Province Recognition**
   - ทำ OCR เฉพาะกรอบกล่องจังหวัด และจับคู่ความใกล้เคียงกับฐานข้อมูล 77 จังหวัด (Fuzzy String Matching)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
TLPR/
├── weights/                      # Model Weights พร้อมใช้งาน
│   ├── plate_detector.pt         # โมเดลตรวจจับป้ายทะเบียน
│   └── char_detector.pt          # โมเดลอ่านตัวอักษร/ตัวเลข (v6)
├── app.py                        # สคริปต์หลักสำหรับรันระบบ LPR Pipeline
├── auto_label_snowball.py        # สคริปต์ทำ Auto-labeling สำหรับเพิ่มข้อมูลภาพ
├── download_dataset.py           # สคริปต์ดาวน์โหลดชุดข้อมูลจาก Roboflow
├── green.jpg                     # ตัวอย่างภาพทดสอบ (รถเก๋ง)
├── truck.png                     # ตัวอย่างภาพทดสอบ (รถบรรทุก)
├── ev.jpg                        # ตัวอย่างภาพทดสอบ (รถ EV)
├── requirements.txt              # รายการ Library ที่ต้องติดตั้ง
├── .gitignore                    # ไฟล์คัดกรองขยะและ Dataset ขนาดใหญ่
└── README.md                     # เอกสารอธิบายการใช้งาน
```

---

## 🚀 การติดตั้งและใช้งาน (Installation & Quick Start)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันโปรแกรมทดสอบอ่านป้ายทะเบียน
```bash
# ทดสอบอ่านป้ายทะเบียนจากภาพ (ระบบจะ Enhance, Rectify และทำ Binarization อัตโนมัติ)
python app.py green.jpg

# กำหนดเทคนิค CLAHE และ Denoising
python app.py truck.png --denoise-method bilateral --clahe-clip 2.5
python app.py truck.png --denoise-method nlmeans
python app.py truck.png --no-enhance

# เลือกเทคนิคการดึงภาพหน้าตรง (homography หรือ affine) หรือปิดการ Rectify
python app.py truck.png --rectify-method homography
python app.py truck.png --rectify-method affine
python app.py truck.png --no-rectify

# กำหนดวิธี Binarization (adaptive, blackhat, otsu) หรือปิดการแปลง
python app.py truck.png --binarize-method blackhat
python app.py truck.png --no-binarize

# กำหนดระดับ Confidence Threshold และ Device เองได้
python app.py truck.png --conf 0.10 --device 0
```

### 3. ผลลัพธ์
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
| `weights/plate_detector.pt` | ตรวจจับกรอบป้ายทะเบียน | YOLO | Auto (640x640) |
| `weights/char_detector.pt` | ตรวจจับตัวอักษร/ตัวเลข 55 คลาส | YOLO | Auto (640x640) |
