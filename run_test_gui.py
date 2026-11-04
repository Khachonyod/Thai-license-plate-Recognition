import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

from tlpr.constants import CLASS_NAME_TO_CHAR, PROJECT_DIR
from tlpr.syntax import parse_thai_license_syntax
from tlpr.visualization import draw_annotated_chars

DEFAULT_WEIGHTS_DIR = PROJECT_DIR / "weights"
DEFAULT_DATASET = PROJECT_DIR / "thai-char-v4" / "train"
DEFAULT_LOG_DIR = PROJECT_DIR / "run_test_log"


def get_ground_truth(label_path: Path, class_names: list) -> str:
    """อ่าน Ground Truth จากไฟล์ Label .txt โดยกรองกล่องจังหวัดออก"""
    if not label_path.is_file():
        return ""
    chars = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            if cid == 5 or yc > 0.60:
                continue
            cls_name = class_names[cid]
            char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
            chars.append((xc, char_text))
    chars.sort(key=lambda item: item[0])
    return "".join(c[1] for c in chars)


class TLPRTestVerifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TLPR - Human Verification & Test GUI")
        self.geometry("1100x860")
        self.minsize(980, 750)

        # ตั้งค่าไอคอนและฟอนต์มาตรฐาน
        self.thai_font = ("Tahoma", 11)
        self.thai_font_bold = ("Tahoma", 11, "bold")
        self.title_font = ("Tahoma", 16, "bold")
        self.plate_font = ("Consolas", 24, "bold")

        # ตัวแปรสถานะ
        self.char_model = None
        self.current_model_path = ""
        self.test_items = []
        self.current_index = 0
        self.results_data = []

        self._init_ui()
        self._refresh_model_list()

    def _init_ui(self):
        # 1. แถบ Config ด้านบน
        config_frame = ttk.LabelFrame(self, text="⚙️ ตั้งค่าการทดสอบ (Test Configuration)", padding=10)
        config_frame.pack(fill=tk.X, padx=12, pady=8)

        # แถวที่ 1: เลือกรุ่นโมเดล และ โฟลเดอร์ Dataset
        row1 = ttk.Frame(config_frame)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="โมเดล:").pack(side=tk.LEFT, padx=(0, 4))
        self.model_combo = ttk.Combobox(row1, width=28, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(row1, text="เลือกไฟล์ .pt...", command=self._browse_model).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row1, text="ชุดข้อมูล (Dataset):").pack(side=tk.LEFT, padx=(0, 4))
        self.dataset_entry = ttk.Entry(row1, width=38)
        self.dataset_entry.insert(0, str(DEFAULT_DATASET))
        self.dataset_entry.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)

        ttk.Button(row1, text="เลือกโฟลเดอร์...", command=self._browse_dataset).pack(side=tk.LEFT)

        # แถวที่ 2: จำนวนภาพ, วิธีเลือกภาพ, Confidence, ปุ่มเริ่มทดสอบ
        row2 = ttk.Frame(config_frame)
        row2.pack(fill=tk.X, pady=4)

        ttk.Label(row2, text="จำนวนรูปที่ตรวจ:").pack(side=tk.LEFT, padx=(0, 4))
        self.count_spin = ttk.Spinbox(row2, from_=1, to=2000, width=6)
        self.count_spin.set(50)
        self.count_spin.pack(side=tk.LEFT, padx=(0, 15))

        self.sample_mode_var = tk.StringVar(value="sequential")
        ttk.Radiobutton(row2, text="เรียงตามชื่อไฟล์", variable=self.sample_mode_var, value="sequential").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(row2, text="สุ่มเลือก (Random)", variable=self.sample_mode_var, value="random").pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row2, text="Confidence:").pack(side=tk.LEFT, padx=(0, 4))
        self.conf_spin = ttk.Spinbox(row2, from_=0.01, to=0.90, increment=0.02, width=5)
        self.conf_spin.set(0.05)
        self.conf_spin.pack(side=tk.LEFT, padx=(0, 25))

        self.start_btn = tk.Button(
            row2,
            text="▶️ เริ่มต้นการทดสอบ (Start Test)",
            bg="#0284c7",
            fg="white",
            font=self.thai_font_bold,
            padx=14,
            pady=3,
            relief=tk.RAISED,
            command=self._start_test_session,
        )
        self.start_btn.pack(side=tk.RIGHT)

        # 2. แถบความคืบหน้า (Progress Bar)
        prog_frame = ttk.Frame(self, padding=(12, 0))
        prog_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.TOP, pady=(2, 4))

        self.progress_label = ttk.Label(prog_frame, text="พร้อมเริ่มการทดสอบ (กดปุ่ม 'เริ่มต้นการทดสอบ' ด้านบน)", font=self.thai_font)
        self.progress_label.pack(side=tk.LEFT)

        self.stats_summary_label = ttk.Label(prog_frame, text="✅ ถูกต้อง: 0 | ❌ ผิด: 0 | รอดำเนินการ: 0", font=self.thai_font_bold)
        self.stats_summary_label.pack(side=tk.RIGHT)

        # 3. พื้นที่แสดงภาพและผลการอ่าน (Main Inspection Card)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # ฝั่งซ้าย: แสดงรูปภาพแผ่นป้าย
        img_container = ttk.LabelFrame(main_paned, text="📷 ภาพแผ่นป้ายทะเบียน (License Plate Image)", padding=10)
        main_paned.add(img_container, weight=3)

        self.view_mode_var = tk.StringVar(value="annotated")
        toggle_frame = ttk.Frame(img_container)
        toggle_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Radiobutton(toggle_frame, text="ภาพพร้อม Bounding Box & Conf", variable=self.view_mode_var, value="annotated", command=self._update_image_display).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(toggle_frame, text="ภาพตัดป้ายต้นฉบับ (Raw Crop)", variable=self.view_mode_var, value="raw", command=self._update_image_display).pack(side=tk.LEFT, padx=6)

        self.img_canvas = tk.Canvas(img_container, bg="#1e293b", highlightthickness=0)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        self.img_path_label = ttk.Label(img_container, text="", font=("Tahoma", 9), foreground="#64748b")
        self.img_path_label.pack(fill=tk.X, pady=(6, 0))

        # ฝั่งขวา: รายละเอียดผลการอ่าน และแผงตรวจด้วยมือ (Human Verification)
        action_container = ttk.LabelFrame(main_paned, text="🔍 ผลการทำนายและตรวจผลด้วยตนเอง", padding=12)
        main_paned.add(action_container, weight=2)

        # การแสดงผลของโมเดล
        ttk.Label(action_container, text="🤖 ผลที่โมเดลอ่านได้ (Prediction):", font=self.thai_font).pack(anchor=tk.W)
        self.pred_display = tk.Label(action_container, text="-", font=self.plate_font, fg="#0284c7", bg="#f1f5f9", padx=10, pady=6, relief=tk.RIDGE)
        self.pred_display.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(action_container, text="🏷️ ค่าจริงจาก Label (Ground Truth):", font=self.thai_font).pack(anchor=tk.W)
        self.gt_display = tk.Label(action_container, text="-", font=("Consolas", 18, "bold"), fg="#334155", bg="#f8fafc", padx=8, pady=4, relief=tk.GROOVE)
        self.gt_display.pack(fill=tk.X, pady=(2, 8))

        # รายละเอียด Confidence รายตัว
        ttk.Label(action_container, text="📊 ความมั่นใจของแต่ละตัวอักษร:", font=self.thai_font).pack(anchor=tk.W)
        self.conf_detail_box = tk.Text(action_container, height=4, font=("Consolas", 10), bg="#f8fafc", relief=tk.SOLID, bd=1)
        self.conf_detail_box.pack(fill=tk.X, pady=(2, 12))
        self.conf_detail_box.config(state=tk.DISABLED)

        # เส้นคั่น
        ttk.Separator(action_container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # โซนตรวจสอบด้วยมือ (Verification Decision)
        ttk.Label(action_container, text="✍️ คุณตรวจแล้ว ข้อความนี้ถูกต้องหรือไม่?", font=self.thai_font_bold).pack(anchor=tk.W, pady=(0, 6))

        btn_grid = ttk.Frame(action_container)
        btn_grid.pack(fill=tk.X, pady=(0, 10))

        self.btn_correct = tk.Button(
            btn_grid,
            text="✅ ถูกต้อง (Y)",
            bg="#16a34a",
            fg="white",
            font=self.thai_font_bold,
            padx=16,
            pady=10,
            relief=tk.RAISED,
            command=self._mark_as_correct,
        )
        self.btn_correct.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_wrong = tk.Button(
            btn_grid,
            text="❌ ผิด (N)",
            bg="#dc2626",
            fg="white",
            font=self.thai_font_bold,
            padx=16,
            pady=10,
            relief=tk.RAISED,
            command=self._mark_as_wrong,
        )
        self.btn_wrong.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        # ช่องกรอกคำตอบที่ถูกต้องถ้าโมเดลอ่านผิด
        ttk.Label(action_container, text="📝 ถ้าโมเดลอ่านผิด คำตอบที่ถูกต้องคือ:", font=self.thai_font).pack(anchor=tk.W)
        self.correction_entry = ttk.Entry(action_container, font=("Consolas", 14, "bold"))
        self.correction_entry.pack(fill=tk.X, pady=(2, 6))
        self.correction_entry.bind("<Return>", lambda event: self._save_correction_and_next())

        # ช่องคอมเมนต์เพิ่มเติม
        ttk.Label(action_container, text="💬 คอมเมนต์ / ข้อสังเกตเพิ่มเติม (ถ้ามี):", font=self.thai_font).pack(anchor=tk.W)
        self.comment_entry = ttk.Entry(action_container, font=self.thai_font)
        self.comment_entry.pack(fill=tk.X, pady=(2, 12))

        # 4. แถบควบคุมด้านล่าง (Navigation & Finish)
        bottom_bar = ttk.Frame(self, padding=(12, 6))
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_prev = ttk.Button(bottom_bar, text="⬅️ ก่อนหน้า (Prev)", command=self._prev_item)
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_next = ttk.Button(bottom_bar, text="ข้ามไปรูปถัดไป ➡️", command=self._next_item)
        self.btn_next.pack(side=tk.LEFT)

        self.btn_finish = tk.Button(
            bottom_bar,
            text="💾 บันทึก Log และสรุปผล (Finish & Save Log)",
            bg="#475569",
            fg="white",
            font=self.thai_font_bold,
            padx=16,
            pady=5,
            relief=tk.RAISED,
            command=self._finish_and_export_log,
        )
        self.btn_finish.pack(side=tk.RIGHT)

        # ผูกปุ่มคีย์ลัดบนคีย์บอร์ด
        self.bind("<Key-y>", lambda event: self._mark_as_correct())
        self.bind("<Key-Y>", lambda event: self._mark_as_correct())
        self.bind("<Key-n>", lambda event: self._mark_as_wrong())
        self.bind("<Key-N>", lambda event: self._mark_as_wrong())
        self.bind("<Left>", lambda event: self._prev_item())
        self.bind("<Right>", lambda event: self._next_item())

    def _refresh_model_list(self):
        models = []
        if DEFAULT_WEIGHTS_DIR.is_dir():
            for f in sorted(DEFAULT_WEIGHTS_DIR.glob("*.pt")):
                models.append(str(f))
        if models:
            self.model_combo["values"] = models
            # เลือก v7 เป็น default ถ้ามี
            v7_match = [m for m in models if "v7" in m]
            if v7_match:
                self.model_combo.set(v7_match[0])
            else:
                self.model_combo.current(0)

    def _browse_model(self):
        filename = filedialog.askopenfilename(
            title="เลือกโมเดล Character Detector (.pt)",
            initialdir=str(DEFAULT_WEIGHTS_DIR),
            filetypes=[("YOLO Weights", "*.pt"), ("All Files", "*.*")],
        )
        if filename:
            current_vals = list(self.model_combo["values"])
            if filename not in current_vals:
                current_vals.append(filename)
                self.model_combo["values"] = current_vals
            self.model_combo.set(filename)

    def _browse_dataset(self):
        dirname = filedialog.askdirectory(
            title="เลือกโฟลเดอร์ชุดข้อมูล (ต้องมี images/ และ labels/)",
            initialdir=str(PROJECT_DIR),
        )
        if dirname:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, dirname)

    def _start_test_session(self):
        model_path_str = self.model_combo.get()
        if not model_path_str or not Path(model_path_str).is_file():
            messagebox.showerror("ผิดพลาด", "กรุณาเลือกไฟล์โมเดล .pt ที่ถูกต้อง")
            return

        dataset_path = Path(self.dataset_entry.get().strip())
        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            messagebox.showerror(
                "ผิดพลาด",
                f"โฟลเดอร์ชุดข้อมูลไม่ถูกต้อง ต้องมีโฟลเดอร์ย่อย images/ และ labels/:\n{dataset_path}",
            )
            return

        try:
            req_count = int(self.count_spin.get())
        except ValueError:
            req_count = 50

        try:
            conf_thresh = float(self.conf_spin.get())
        except ValueError:
            conf_thresh = 0.05

        # ค้นหารายการภาพที่มี label คู่กัน
        pairs = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in img_dir.glob(ext):
                lp = lbl_dir / f"{p.stem}.txt"
                if lp.is_file():
                    pairs.append((p, lp))

        if not pairs:
            messagebox.showerror("ผิดพลาด", f"ไม่พบไฟล์รูปภาพคู่กับ label ใน {dataset_path}")
            return

        pairs.sort(key=lambda x: x[0].name)
        test_count = min(req_count, len(pairs))

        if self.sample_mode_var.get() == "random":
            random.seed(int(time.time()))
            selected_pairs = random.sample(pairs, test_count)
        else:
            selected_pairs = pairs[:test_count]

        # โหลดโมเดล
        if self.current_model_path != model_path_str or self.char_model is None:
            self.progress_label.config(text=f"กำลังโหลดโมเดล: {Path(model_path_str).name}...")
            self.update_idletasks()
            device = "0" if torch.cuda.is_available() else "cpu"
            self.char_model = YOLO(model_path_str)
            self.current_model_path = model_path_str

        device = "0" if torch.cuda.is_available() else "cpu"
        class_names = self.char_model.names

        self.test_items = []
        self.results_data = []

        self.progress_label.config(text=f"กำลังทำการพยากรณ์ล่วงหน้า {test_count} รูป...")
        self.update_idletasks()

        for idx, (img_p, lbl_p) in enumerate(selected_pairs):
            gt_text = get_ground_truth(lbl_p, class_names)
            crop_bgr = cv2.imread(str(img_p))
            if crop_bgr is None:
                continue

            t0 = time.perf_counter()
            pred_text, _, _, _, kept_boxes = parse_thai_license_syntax(
                char_model=self.char_model,
                plate_crop=crop_bgr,
                conf_thresh=conf_thresh,
                device=device,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            # สร้างภาพ Annotated พร้อม Bounding Box และ Confidence
            annotated_pil = draw_annotated_chars(crop_bgr, kept_boxes)

            gt_clean = gt_text.replace(" ", "")
            pred_clean = pred_text.replace(" ", "")
            auto_match = (gt_clean == pred_clean) and (len(gt_clean) > 0)

            item_record = {
                "idx": idx + 1,
                "file_name": img_p.name,
                "file_path": str(img_p.resolve()),
                "crop_bgr": crop_bgr,
                "annotated_pil": annotated_pil,
                "gt": gt_clean,
                "pred": pred_clean,
                "kept_boxes": kept_boxes,
                "dt_ms": dt_ms,
                "auto_match": auto_match,
                # ข้อมูลการตรวจด้วยมือ (Human Review)
                "human_verified": None,  # True (ถูก), False (ผิด), None (ยังไม่ตรวจ)
                "corrected_text": pred_clean,
                "comment": "",
            }
            self.test_items.append(item_record)

        if not self.test_items:
            messagebox.showerror("ผิดพลาด", "ไม่สามารถประมวลผลรูปภาพที่เลือกได้")
            return

        self.current_index = 0
        self._load_current_item()
        messagebox.showinfo(
            "พร้อมตรวจแล้ว",
            f"ประมวลผลรูปภาพเสร็จสิ้น {len(self.test_items)} รูป!\n"
            "คุณสามารถกดปุ่ม '✅ ถูกต้อง' หรือ '❌ ผิด' เพื่อตรวจทานทีละรูปได้เลยครับ\n(ปุ่มลัด: Y = ถูก, N = ผิด)",
        )

    def _load_current_item(self):
        if not self.test_items or not (0 <= self.current_index < len(self.test_items)):
            return

        item = self.test_items[self.current_index]
        total = len(self.test_items)
        curr_num = self.current_index + 1

        # อัปเดตแถบ Progress
        self.progress_var.set((curr_num / total) * 100)
        self.progress_label.config(
            text=f"รูปที่ {curr_num} จาก {total} ({(curr_num / total):.0%}) | ชื่อไฟล์: {item['file_name']}"
        )
        self.img_path_label.config(text=f"📁 {item['file_path']}")

        # แสดงข้อความผลลัพธ์
        self.pred_display.config(text=item["pred"] if item["pred"] else "(อ่านไม่ได้)")
        self.gt_display.config(text=item["gt"] if item["gt"] else "(ไม่มี Label)")

        # รายละเอียด Confidence รายตัว
        self.conf_detail_box.config(state=tk.NORMAL)
        self.conf_detail_box.delete("1.0", tk.END)
        if item["kept_boxes"]:
            lines = [f"{b[4]}: {b[5]:.1%}" for b in item["kept_boxes"]]
            self.conf_detail_box.insert(tk.END, " | ".join(lines))
        else:
            self.conf_detail_box.insert(tk.END, "ไม่พบตัวอักษรใดๆ ที่ผ่าน Threshold")
        self.conf_detail_box.config(state=tk.DISABLED)

        # โหลดค่าคำตอบที่ถูกต้องและคอมเมนต์เดิมถ้ามี
        self.correction_entry.delete(0, tk.END)
        if item["corrected_text"]:
            self.correction_entry.insert(0, item["corrected_text"])
        else:
            self.correction_entry.insert(0, item["pred"])

        self.comment_entry.delete(0, tk.END)
        if item["comment"]:
            self.comment_entry.insert(0, item["comment"])

        # เน้นปุ่มตามสถานะที่เคยตรวจแล้ว
        if item["human_verified"] is True:
            self.btn_correct.config(relief=tk.SUNKEN, bg="#15803d")
            self.btn_wrong.config(relief=tk.RAISED, bg="#dc2626")
        elif item["human_verified"] is False:
            self.btn_correct.config(relief=tk.RAISED, bg="#16a34a")
            self.btn_wrong.config(relief=tk.SUNKEN, bg="#991b1b")
        else:
            self.btn_correct.config(relief=tk.RAISED, bg="#16a34a")
            self.btn_wrong.config(relief=tk.RAISED, bg="#dc2626")

        self._update_stats_summary()
        self._update_image_display()

    def _update_image_display(self):
        if not self.test_items or not (0 <= self.current_index < len(self.test_items)):
            return

        item = self.test_items[self.current_index]
        mode = self.view_mode_var.get()

        if mode == "annotated":
            pil_img = item["annotated_pil"].copy()
        else:
            rgb = cv2.cvtColor(item["crop_bgr"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

        # ปรับขนาดภาพให้พอดีกับ Canvas อย่างสวยงาม
        c_w = self.img_canvas.winfo_width()
        c_h = self.img_canvas.winfo_height()
        if c_w < 50 or c_h < 50:
            c_w, c_h = 580, 420

        orig_w, orig_h = pil_img.size
        ratio = min(c_w / orig_w, c_h / orig_h)
        target_w = max(10, int(orig_w * ratio))
        target_h = max(10, int(orig_h * ratio))

        resized = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.img_canvas.delete("all")
        self.img_canvas.create_image(c_w // 2, c_h // 2, image=self.tk_img, anchor=tk.CENTER)

    def _mark_as_correct(self):
        if not self.test_items:
            return
        item = self.test_items[self.current_index]
        item["human_verified"] = True
        item["corrected_text"] = item["pred"]
        item["comment"] = self.comment_entry.get().strip()
        self._next_item()

    def _mark_as_wrong(self):
        if not self.test_items:
            return
        item = self.test_items[self.current_index]
        item["human_verified"] = False
        self.btn_wrong.config(relief=tk.SUNKEN, bg="#991b1b")
        self.btn_correct.config(relief=tk.RAISED, bg="#16a34a")

        # เติมคำตอบที่ถูกต้องเป็น GT ถ้ามี หรือโฟกัสให้ผู้ใช้พิมพ์แก้
        if item["gt"] and item["gt"] != item["pred"]:
            self.correction_entry.delete(0, tk.END)
            self.correction_entry.insert(0, item["gt"])
        self.correction_entry.focus_set()
        self.correction_entry.select_range(0, tk.END)

    def _save_correction_and_next(self):
        if not self.test_items:
            return
        item = self.test_items[self.current_index]
        item["human_verified"] = False
        item["corrected_text"] = self.correction_entry.get().strip()
        item["comment"] = self.comment_entry.get().strip()
        self._next_item()

    def _next_item(self):
        if not self.test_items:
            return
        # บันทึกค่า input ปัจจุบันก่อนเลื่อน
        item = self.test_items[self.current_index]
        if item["human_verified"] is None:
            # ถ้าผู้ใช้กด Next โดยไม่ได้คลิกถูก/ผิด ให้อนุมานจาก auto_match
            item["human_verified"] = item["auto_match"]
            item["corrected_text"] = self.correction_entry.get().strip()
            item["comment"] = self.comment_entry.get().strip()

        if self.current_index < len(self.test_items) - 1:
            self.current_index += 1
            self._load_current_item()
        else:
            self._finish_and_export_log()

    def _prev_item(self):
        if not self.test_items:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_item()

    def _update_stats_summary(self):
        if not self.test_items:
            return
        total = len(self.test_items)
        correct = sum(1 for it in self.test_items if it["human_verified"] is True)
        wrong = sum(1 for it in self.test_items if it["human_verified"] is False)
        pending = total - (correct + wrong)
        self.stats_summary_label.config(
            text=f"✅ ถูกต้อง: {correct} ({correct/total:.0%}) | ❌ ผิด: {wrong} ({wrong/total:.0%}) | ⏳ รอตรวจ: {pending}"
        )

    def _finish_and_export_log(self):
        if not self.test_items:
            return

        total = len(self.test_items)
        unreviewed = [it for it in self.test_items if it["human_verified"] is None]
        if unreviewed:
            ans = messagebox.askyesno(
                "ตรวจทานยังไม่ครบ",
                f"ยังมีอีก {len(unreviewed)} รูปที่ยังไม่ได้กดเลือก ถูก หรือ ผิด\n"
                "ต้องการให้ระบบเทียบตามผลอัตโนมัติ (Auto-Match กับ Label) แล้วบันทึกเลยหรือไม่?",
            )
            if not ans:
                return
            for it in unreviewed:
                it["human_verified"] = it["auto_match"]

        correct_count = sum(1 for it in self.test_items if it["human_verified"] is True)
        wrong_count = total - correct_count
        acc_pct = (correct_count / total) * 100.0

        # จัดเตรียมข้อมูลบันทึก Log
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")

        log_dir = DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gui_test_log_{file_timestamp}.md"
        latest_path = log_dir / "latest.md"

        model_name = Path(self.current_model_path).name if self.current_model_path else "Unknown"

        md_lines = [
            "# 📋 รายงานผลการทดสอบและตรวจทานด้วยมือ (Human-Verified Test Log)",
            "",
            f"- **วัน-เวลาที่ทดสอบ:** {timestamp_str}",
            f"- **โมเดลที่ใช้:** `{model_name}` (`{self.current_model_path}`)",
            f"- **ชุดข้อมูล:** `{self.dataset_entry.get().strip()}`",
            f"- **โหมดการตรวจ:** Human-in-the-loop Manual Verification (ตรวจยืนยันด้วยสายตา)",
            "",
            "---",
            "",
            "## 📊 สรุปสถิติผลการประเมิน (Verified Statistics)",
            "",
            "| ตัวชี้วัด (Metric) | จำนวน / ค่าที่ได้ | ร้อยละ (Percentage) |",
            "| :--- | :--- | :--- |",
            f"| **จำนวนภาพที่ทดสอบทั้งหมด** | **{total}** รูปภาพ | 100.00% |",
            f"| **✅ ผ่านการตรวจยืนยันว่า 'ถูกต้อง'** | **{correct_count}** รูปภาพ | **{acc_pct:.2f}%** |",
            f"| **❌ ตรวจพบว่า 'อ่านผิด'** | **{wrong_count}** รูปภาพ | **{100.0 - acc_pct:.2f}%** |",
            "",
            "---",
            "",
        ]

        # ตารางเฉพาะรายการที่อ่านผิด พร้อมคำตอบที่มนุษย์แก้ไขให้
        wrong_items = [it for it in self.test_items if it["human_verified"] is False]
        md_lines.append(f"## ⚠️ รายการภาพที่อ่านผิดพร้อมเฉลยที่ถูกต้อง ({len(wrong_items)} รูปภาพ)\n")
        if wrong_items:
            md_lines.extend([
                "| # | ชื่อไฟล์ | ผลที่โมเดลทาย | เฉลยที่มนุษย์แก้ไข | ค่าใน Label เดิม | คอมเมนต์ | พาธไฟล์ (File Path) |",
                "| :-: | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for it in wrong_items:
                comment_text = it["comment"] if it["comment"] else "-"
                gt_text = it["gt"] if it["gt"] else "-"
                md_lines.append(
                    f"| {it['idx']:02d} | `{it['file_name']}` | `{it['pred']}` | **`{it['corrected_text']}`** | `{gt_text}` | {comment_text} | `{it['file_path']}` |"
                )
            md_lines.append("")
        else:
            md_lines.append("🎉 *ยอดเยี่ยมมาก! ไม่มีรูปภาพที่อ่านผิดเลยในการทดสอบรอบนี้*\n")

        md_lines.extend([
            "---",
            "",
            "## 📑 ตารางบันทึกผลการทดสอบทุกไฟล์ (Complete Test Records)",
            "",
            "| # | ผลตรวจ | ชื่อไฟล์ | ผลโมเดลทาย | คำตอบที่ถูกต้อง | Label เดิม | Conf เฉลี่ย | พาธไฟล์เต็ม (Full Path) |",
            "| :-: | :-: | :--- | :--- | :--- | :--- | :-: | :--- |",
        ])

        for it in self.test_items:
            verdict_icon = "✅ ถูกต้อง" if it["human_verified"] is True else "❌ อ่านผิด"
            avg_c = (
                sum(b[5] for b in it["kept_boxes"]) / len(it["kept_boxes"])
                if it["kept_boxes"]
                else 0.0
            )
            gt_text = it["gt"] if it["gt"] else "-"
            md_lines.append(
                f"| {it['idx']:02d} | {verdict_icon} | `{it['file_name']}` | `{it['pred']}` | **`{it['corrected_text']}`** | `{gt_text}` | {avg_c:.1%} | `{it['file_path']}` |"
            )

        content = "\n".join(md_lines) + "\n"
        log_path.write_text(content, encoding="utf-8")
        latest_path.write_text(content, encoding="utf-8")

        messagebox.showinfo(
            "บันทึก Log เรียบร้อยแล้ว",
            f"🎉 ตรวจทานเสร็จสิ้นเรียบร้อย!\n\n"
            f"📊 สรุปผล:\n"
            f"   - ถูกต้อง: {correct_count}/{total} รูปภาพ ({acc_pct:.2f}%)\n"
            f"   - อ่านผิด: {wrong_count}/{total} รูปภาพ ({100.0 - acc_pct:.2f}%)\n\n"
            f"📝 บันทึก Log แล้วที่:\n{log_path}",
        )


def main():
    app_gui = TLPRTestVerifierGUI()
    app_gui.mainloop()


if __name__ == "__main__":
    main()

