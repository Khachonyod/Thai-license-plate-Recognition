# TLPR: Thai License Plate Recognition

ระบบตรวจจับและอ่านป้ายทะเบียนรถยนต์ไทยแบบ End-to-End อัตโนมัติ (Two-Stage YOLO Architecture + Syntax Validator)

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **Stage 1: Plate Detector (`weights/plate_detector.pt`)**
   - ตรวจจับตำแหน่งแผ่นป้ายทะเบียนบนตัวรถอย่างแม่นยำ พร้อมทำ Crop อัตโนมัติ
2. **Stage 2: Character Detector (`weights/char_detector.pt`)**
   - ตรวจจับตัวอักษรและตัวเลข 55 คลาส (พยัญชนะไทย 44 ตัว, ตัวเลข 0-9, และกรอบข้อความจังหวัด)
3. **Thai Syntax Validator & Slot Clustering**
   - ระบบจัดกลุ่ม Slot ตามโครงสร้างไวยากรณ์ป้ายทะเบียนไทย (เช่น หมวด `1กข 1234` หรือ `กข 1234`)
   - แก้ปัญหาความกำกวมของตัวอักษร เช่น แยกแยะเลข `1` กับเลข `7` ตาม Slot ตำแหน่ง
4. **Province Recognition**
   - ทำ OCR เฉพาะกรอบกล่องจังหวัด และจับคู่ความใกล้เคียงกับฐานข้อมูล 77 จังหวัด (Fuzzy String Matching)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
TLPR/
├── weights/                      # Model Weights พร้อมใช้งาน
│   ├── plate_detector.pt         # โมเดลตรวจจับป้ายทะเบียน
│   └── char_detector.pt          # โมเดลอ่านตัวอักษร/ตัวเลข (v5)
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
# ทดสอบอ่านป้ายทะเบียนจากภาพ
python app.py green.jpg

# กำหนดระดับ Confidence Threshold และ Device เองได้
python app.py truck.png --conf 0.10 --device 0
```

### 3. ผลลัพธ์
- ผลการตรวจจับและ Crop แผ่นป้ายจะถูกบันทึกไว้ที่โฟลเดอร์ `runs/lpr_results/`
- หน้าจอ Terminal จะแสดงผลหมวดเลขทะเบียน และจังหวัดที่อ่านได้ทันที

---

## 🛠️ รายละเอียดโมเดล (Trained Models)

| โมเดล | หน้าที่ | Backbone | Input Size |
| :--- | :--- | :--- | :--- |
| `weights/plate_detector.pt` | ตรวจจับกรอบป้ายทะเบียน | YOLO | Auto (640x640) |
| `weights/char_detector.pt` | ตรวจจับตัวอักษร/ตัวเลข 55 คลาส | YOLO | Auto (640x640) |
