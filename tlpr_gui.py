"""
TLPR Studio: Interactive Thai License Plate Recognition GUI
Supports Images and Videos with Full Pre-Processing and Pipeline Configuration
"""

import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, Union
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from tlpr.constants import (
    DEFAULT_CHAR_MODEL,
    DEFAULT_PLATE_MODEL,
    PROJECT_DIR,
)
from tlpr.pipeline import TLPRPipeline, TLPRResult, PlateDetection
from tlpr.visualization import get_default_thai_font


class TLPRStudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TLPR Studio - Thai License Plate Recognition & Pre-processing Lab")
        self.geometry("1060x690")
        self.minsize(780, 500)

        # Fonts & Styling
        self.thai_font = ("Tahoma", 9)
        self.thai_font_bold = ("Tahoma", 9, "bold")
        self.title_font = ("Tahoma", 13, "bold")
        self.plate_font = ("Consolas", 22, "bold")
        self.province_font = ("Tahoma", 12, "bold")

        # Core State
        self.pipeline: Optional[TLPRPipeline] = None
        self.pipeline_lock = threading.Lock()
        self.current_media_type = None  # "image" or "video"
        self.current_media_path: Optional[Path] = None
        self.current_cv_image: Optional[np.ndarray] = None
        self.last_result: Optional[TLPRResult] = None

        # Video State
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_playing = False
        self.video_thread: Optional[threading.Thread] = None
        self.stop_video_flag = threading.Event()
        self.video_queue = queue.Queue(maxsize=2)
        self.video_fps = 30
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False

        # Image cache for Tkinter
        self.tk_main_image = None
        self.stage_canvases = {"raw": [], "enhanced": [], "rectified": [], "binarized": [], "chars": []}

        self._init_ui()
        self._init_pipeline_async()

    # ---------------------------------------------------------
    # UI Layout Initialization
    # ---------------------------------------------------------
    def _init_ui(self):
        # 1. Top Header Bar (Compact)
        header = tk.Frame(self, bg="#0f172a", height=42, padx=12, pady=6)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="🚗 TLPR Studio",
            font=self.title_font,
            fg="#38bdf8",
            bg="#0f172a",
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header,
            text="ระบบทดสอบอ่านป้ายทะเบียนไทย (รองรับภาพนิ่ง/วิดีโอคลิป + ปรับแต่ง Pre-Processing)",
            font=self.thai_font,
            fg="#94a3b8",
            bg="#0f172a",
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(10, 0))

        self.status_lbl = tk.Label(
            header,
            text="⏳ กำลังโหลดโมเดล...",
            font=self.thai_font_bold,
            fg="#fbbf24",
            bg="#0f172a",
        )
        self.status_lbl.pack(side=tk.RIGHT)

        # 2. Main Content Panes: Left (Config & Controls) | Right (Visuals & Results)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left Sidebar (Controls & Config) - Compact 285px width
        left_container = ttk.Frame(main_paned, width=285)
        main_paned.add(left_container, weight=0)

        # Right Container (Viewports & Inspection)
        right_container = ttk.Frame(main_paned)
        main_paned.add(right_container, weight=1)

        self._build_sidebar(left_container)
        self._build_right_viewport(right_container)

        # Bottom Status Bar (Compact)
        bottom_bar = tk.Frame(self, bg="#1e293b", padx=10, pady=3)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.footer_info = tk.Label(
            bottom_bar,
            text="พร้อมใช้งาน: เลือกภาพหรือวิดีโอเพื่อเริ่มการทดสอบ",
            font=self.thai_font,
            fg="#cbd5e1",
            bg="#1e293b",
        )
        self.footer_info.pack(side=tk.LEFT)

        self.perf_info = tk.Label(
            bottom_bar,
            text="ความเร็ว: - ms | 0.0 FPS",
            font=self.thai_font_bold,
            fg="#38bdf8",
            bg="#1e293b",
        )
        self.perf_info.pack(side=tk.RIGHT)

    def _build_sidebar(self, parent):
        # Scrollable container for controls
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0, width=280)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        pad_opts = {"padx": 4, "pady": 3}

        # --- Group 1: Source Import ---
        src_box = ttk.LabelFrame(scrollable_frame, text="📂 1. แหล่งข้อมูล (Source)", padding=6)
        src_box.pack(fill=tk.X, **pad_opts)

        btn_img = tk.Button(
            src_box,
            text="🖼️ เปิดภาพนิ่ง (Import Image)...",
            bg="#0284c7",
            fg="white",
            font=self.thai_font_bold,
            pady=3,
            command=self._choose_image,
        )
        btn_img.pack(fill=tk.X, pady=(0, 3))

        btn_video = tk.Button(
            src_box,
            text="🎬 เปิดวิดีโอคลิป (Import Video)...",
            bg="#7c3aed",
            fg="white",
            font=self.thai_font_bold,
            pady=3,
            command=self._choose_video,
        )
        btn_video.pack(fill=tk.X, pady=(0, 3))

        sample_row = ttk.Frame(src_box)
        sample_row.pack(fill=tk.X, pady=1)
        ttk.Label(sample_row, text="ตัวอย่าง:", font=self.thai_font).pack(side=tk.LEFT)

        for i in [1, 2, 3, 5]:
            ttk.Button(
                sample_row,
                text=f"{i}",
                width=3,
                command=lambda n=i: self._load_quick_sample(n),
            ).pack(side=tk.LEFT, padx=1)

        self.source_label = ttk.Label(src_box, text="ยังไม่ได้เลือกสื่อ", font=self.thai_font, foreground="#64748b")
        self.source_label.pack(anchor=tk.W, pady=(2, 0))

        # --- Group 2: Video Playback Controls ---
        self.video_box = ttk.LabelFrame(scrollable_frame, text="⏯️ เล่นวิดีโอ (Video Controls)", padding=6)
        self.video_box.pack(fill=tk.X, **pad_opts)

        v_btn_row = ttk.Frame(self.video_box)
        v_btn_row.pack(fill=tk.X, pady=(0, 2))

        self.btn_play = ttk.Button(v_btn_row, text="▶️ เล่น", width=6, command=self._play_video)
        self.btn_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))

        self.btn_pause = ttk.Button(v_btn_row, text="⏸️ หยุดชั่วคราว", width=10, command=self._pause_video)
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        self.btn_stop = ttk.Button(v_btn_row, text="⏹️ หยุด", width=6, command=self._stop_video)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        self.seek_var = tk.DoubleVar(value=0)
        self.seek_slider = ttk.Scale(
            self.video_box,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=self._on_seek,
        )
        self.seek_slider.pack(fill=tk.X, pady=1)

        self.frame_counter_lbl = ttk.Label(self.video_box, text="Frame: 0 / 0", font=self.thai_font)
        self.frame_counter_lbl.pack(anchor=tk.W)

        skip_row = ttk.Frame(self.video_box)
        skip_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(skip_row, text="ข้ามเฟรม:", font=self.thai_font).pack(side=tk.LEFT)
        self.skip_combo = ttk.Combobox(skip_row, values=["1 (ทุกเฟรม)", "2 (สมดุล)", "3 (เร็ว)", "5 (เร็วมาก)"], state="readonly", width=13)
        self.skip_combo.current(1)
        self.skip_combo.pack(side=tk.RIGHT)

        self.save_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.video_box,
            text="บันทึกคลิปผลลัพธ์เป็น MP4",
            variable=self.save_video_var,
        ).pack(anchor=tk.W, pady=(2, 0))

        # --- Group 3: Pre-processing: Image Enhancement ---
        enh_box = ttk.LabelFrame(scrollable_frame, text="✨ 2. Enhancement (CLAHE / Denoise)", padding=6)
        enh_box.pack(fill=tk.X, **pad_opts)

        self.enable_enh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            enh_box,
            text="เปิด CLAHE & Denoising",
            variable=self.enable_enh_var,
            command=self._on_param_change,
        ).pack(anchor=tk.W)

        row_clahe = ttk.Frame(enh_box)
        row_clahe.pack(fill=tk.X, pady=1)
        ttk.Label(row_clahe, text="CLAHE Clip:").pack(side=tk.LEFT)
        self.clahe_val_lbl = ttk.Label(row_clahe, text="2.0", font=self.thai_font_bold)
        self.clahe_val_lbl.pack(side=tk.RIGHT)

        self.clahe_slider = ttk.Scale(
            enh_box,
            from_=0.5,
            to=5.0,
            value=2.0,
            command=self._on_clahe_slide,
        )
        self.clahe_slider.pack(fill=tk.X, pady=(0, 2))

        row_denoise = ttk.Frame(enh_box)
        row_denoise.pack(fill=tk.X, pady=1)
        ttk.Label(row_denoise, text="วิธี Denoise:").pack(side=tk.LEFT)
        self.denoise_combo = ttk.Combobox(
            row_denoise,
            values=["bilateral", "nlmeans", "gaussian", "none"],
            state="readonly",
            width=10,
        )
        self.denoise_combo.set("bilateral")
        self.denoise_combo.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.denoise_combo.pack(side=tk.RIGHT)

        # --- Group 4: Pre-processing: Plate Rectification ---
        rect_box = ttk.LabelFrame(scrollable_frame, text="📐 3. Rectification (ดึงป้ายหน้าตรง)", padding=6)
        rect_box.pack(fill=tk.X, **pad_opts)

        self.enable_rect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rect_box,
            text="เปิด Plate Rectification",
            variable=self.enable_rect_var,
            command=self._on_param_change,
        ).pack(anchor=tk.W)

        row_rect = ttk.Frame(rect_box)
        row_rect.pack(fill=tk.X, pady=1)
        ttk.Label(row_rect, text="วิธี Rectify:").pack(side=tk.LEFT)
        self.rect_combo = ttk.Combobox(
            row_rect,
            values=["homography", "affine"],
            state="readonly",
            width=10,
        )
        self.rect_combo.set("homography")
        self.rect_combo.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.rect_combo.pack(side=tk.RIGHT)

        self.use_rect_char_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            rect_box,
            text="ส่งภาพ Rectified ให้อ่านอักษร",
            variable=self.use_rect_char_var,
            command=self._on_param_change,
        ).pack(anchor=tk.W, pady=(1, 0))

        # --- Group 5: Pre-processing: Binarization ---
        bin_box = ttk.LabelFrame(scrollable_frame, text="🌓 4. Binarization (ภาพขาว-ดำ)", padding=6)
        bin_box.pack(fill=tk.X, **pad_opts)

        self.enable_bin_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bin_box,
            text="เปิด Binarization & Edges",
            variable=self.enable_bin_var,
            command=self._on_param_change,
        ).pack(anchor=tk.W)

        row_bin = ttk.Frame(bin_box)
        row_bin.pack(fill=tk.X, pady=1)
        ttk.Label(row_bin, text="วิธี Binarize:").pack(side=tk.LEFT)
        self.bin_combo = ttk.Combobox(
            row_bin,
            values=["adaptive", "blackhat", "otsu"],
            state="readonly",
            width=10,
        )
        self.bin_combo.set("adaptive")
        self.bin_combo.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        self.bin_combo.pack(side=tk.RIGHT)

        # --- Group 6: Detection & Model Options ---
        det_box = ttk.LabelFrame(scrollable_frame, text="🎯 5. Model & Thresholds", padding=6)
        det_box.pack(fill=tk.X, **pad_opts)

        row_conf = ttk.Frame(det_box)
        row_conf.pack(fill=tk.X, pady=1)
        ttk.Label(row_conf, text="Confidence:").pack(side=tk.LEFT)
        self.conf_val_lbl = ttk.Label(row_conf, text="0.08", font=self.thai_font_bold)
        self.conf_val_lbl.pack(side=tk.RIGHT)

        self.conf_slider = ttk.Scale(
            det_box,
            from_=0.02,
            to=0.80,
            value=0.08,
            command=self._on_conf_slide,
        )
        self.conf_slider.pack(fill=tk.X, pady=(0, 2))

        row_dev = ttk.Frame(det_box)
        row_dev.pack(fill=tk.X, pady=1)
        ttk.Label(row_dev, text="Device:").pack(side=tk.LEFT)
        self.device_combo = ttk.Combobox(
            row_dev,
            values=["0 (GPU)", "cpu"],
            state="readonly",
            width=8,
        )
        self.device_combo.set("0 (GPU)")
        self.device_combo.bind("<<ComboboxSelected>>", lambda e: self._on_device_change())
        self.device_combo.pack(side=tk.RIGHT)

        self.enable_ocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            det_box,
            text="อ่านกล่องจังหวัด (EasyOCR)",
            variable=self.enable_ocr_var,
            command=self._on_device_change,
        ).pack(anchor=tk.W, pady=(1, 0))

        # Re-run button for static image
        self.btn_rerun = tk.Button(
            scrollable_frame,
            text="🔄 ประมวลผลใหม่ (Re-Run)",
            bg="#059669",
            fg="white",
            font=self.thai_font_bold,
            pady=4,
            command=self._reprocess_current_image,
        )
        self.btn_rerun.pack(fill=tk.X, padx=4, pady=6)

    def _build_right_viewport(self, parent):
        # 1. Top Result Banner (Compact)
        result_card = tk.Frame(parent, bg="#1e293b", relief=tk.RIDGE, bd=1, padx=12, pady=6)
        result_card.pack(fill=tk.X, pady=(0, 4))

        banner_row = tk.Frame(result_card, bg="#1e293b")
        banner_row.pack(fill=tk.X)

        self.plate_text_lbl = tk.Label(
            banner_row,
            text="-- รอผลการอ่าน --",
            font=self.plate_font,
            fg="#38bdf8",
            bg="#1e293b",
        )
        self.plate_text_lbl.pack(side=tk.LEFT, padx=(0, 16))

        self.province_text_lbl = tk.Label(
            banner_row,
            text="",
            font=self.province_font,
            fg="#4ade80",
            bg="#1e293b",
        )
        self.province_text_lbl.pack(side=tk.LEFT)

        self.btn_save_snap = ttk.Button(banner_row, text="📸 บันทึกภาพ", command=self._save_snapshot)
        self.btn_save_snap.pack(side=tk.RIGHT)

        # 2. Main Vehicle Image / Video Viewport (Expands to take all available space)
        main_box = ttk.LabelFrame(parent, text="📺 ภาพตรวจจับหลัก (Annotated Vehicle & Detection View)", padding=2)
        main_box.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.main_canvas = tk.Canvas(main_box, bg="#020617", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<Configure>", lambda e: self._redraw_main_image())

        # 3. Bottom Pre-processing Inspection Bar (Compact & Collapsible / Tabbed)
        self.bottom_stages_frame = ttk.LabelFrame(parent, padding=4)
        self.bottom_stages_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Header with toggle button
        stages_header = ttk.Frame(self.bottom_stages_frame)
        stages_header.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(stages_header, text="🔬 ขั้นตอน Pre-Processing (5 ขั้นตอน):", font=self.thai_font_bold).pack(side=tk.LEFT)

        self.btn_toggle_stages = ttk.Button(
            stages_header,
            text="🔼 ย่อแถบนี้",
            width=12,
            command=self._toggle_bottom_stages,
        )
        self.btn_toggle_stages.pack(side=tk.RIGHT)

        # Tabbed Container (Takes very minimal vertical space, fits any resolution)
        self.bottom_notebook = ttk.Notebook(self.bottom_stages_frame)
        self.bottom_notebook.pack(fill=tk.X, expand=False)

        # Tab 1: Overview 5 Steps in Compact Mini Row (Height ~75px)
        tab_overview = ttk.Frame(self.bottom_notebook, padding=2)
        self.bottom_notebook.add(tab_overview, text="📋 ภาพรวม (5 ขั้นตอน)")

        stage_specs = [
            ("raw", "1. Raw"),
            ("enhanced", "2. CLAHE"),
            ("rectified", "3. Rectified"),
            ("binarized", "4. Binarized"),
            ("chars", "5. Chars"),
        ]

        overview_grid = ttk.Frame(tab_overview)
        overview_grid.pack(fill=tk.BOTH, expand=True)

        for key, short_title in stage_specs:
            cell = ttk.Frame(overview_grid)
            cell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

            lbl = ttk.Label(cell, text=short_title, font=self.thai_font, anchor="center")
            lbl.pack(side=tk.TOP, fill=tk.X)

            cvs = tk.Canvas(cell, bg="#0f172a", height=70, highlightthickness=1, highlightbackground="#334155")
            cvs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.stage_canvases[key].append(cvs)

        # Tabs 2-6: Individual Detail Views for each step
        detail_names = [
            ("raw", "1. Raw Crop"),
            ("enhanced", "2. CLAHE & Denoise"),
            ("rectified", "3. Rectified (หน้าตรง)"),
            ("binarized", "4. Binarized (ขาวดำ)"),
            ("chars", "5. Stage 2 Chars"),
        ]

        for key, full_title in detail_names:
            detail_tab = ttk.Frame(self.bottom_notebook, padding=2)
            self.bottom_notebook.add(detail_tab, text=full_title)

            cvs = tk.Canvas(detail_tab, bg="#0f172a", height=110, highlightthickness=1, highlightbackground="#334155")
            cvs.pack(fill=tk.BOTH, expand=True)
            self.stage_canvases[key].append(cvs)

    def _toggle_bottom_stages(self):
        if self.bottom_notebook.winfo_ismapped():
            self.bottom_notebook.pack_forget()
            self.btn_toggle_stages.configure(text="🔽 ขยายดู Pre-processing")
        else:
            self.bottom_notebook.pack(fill=tk.X, expand=False)
            self.btn_toggle_stages.configure(text="🔼 ย่อแถบนี้")

    # ---------------------------------------------------------
    # Pipeline Initialization (Async)
    # ---------------------------------------------------------
    def _init_pipeline_async(self):
        def _load():
            try:
                device_str = "0" if "GPU" in self.device_combo.get() else "cpu"
                ocr_enabled = self.enable_ocr_var.get()
                p = TLPRPipeline(
                    device=device_str,
                    enable_ocr=ocr_enabled,
                )
                with self.pipeline_lock:
                    self.pipeline = p
                self.after(0, lambda: self.status_lbl.configure(text="✅ โมเดลพร้อมใช้งาน", fg="#4ade80"))
                self.after(200, lambda: self._load_quick_sample(1))
            except Exception as e:
                self.after(0, lambda: self.status_lbl.configure(text=f"❌ โหลดโมเดลล้มเหลว: {e}", fg="#f87171"))

        threading.Thread(target=_load, daemon=True).start()

    def _on_device_change(self):
        self.status_lbl.configure(text="⏳ กำลังรีโหลดโมเดล...", fg="#fbbf24")
        self._init_pipeline_async()

    # ---------------------------------------------------------
    # Slider & Param Callbacks
    # ---------------------------------------------------------
    def _on_clahe_slide(self, val):
        self.clahe_val_lbl.configure(text=f"{float(val):.1f}")
        if self.current_media_type == "image":
            self._reprocess_current_image()

    def _on_conf_slide(self, val):
        self.conf_val_lbl.configure(text=f"{float(val):.2f}")
        if self.current_media_type == "image":
            self._reprocess_current_image()

    def _on_param_change(self):
        if self.current_media_type == "image":
            self._reprocess_current_image()

    # ---------------------------------------------------------
    # Media Import Handlers
    # ---------------------------------------------------------
    def _choose_image(self):
        self._stop_video()
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
        p = filedialog.askopenfilename(title="เลือกไฟล์รูปภาพเพื่อทดสอบ", filetypes=filetypes)
        if not p:
            return
        self._load_image_file(Path(p))

    def _choose_video(self):
        self._stop_video()
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"),
            ("All files", "*.*"),
        ]
        p = filedialog.askopenfilename(title="เลือกไฟล์วิดีโอคลิปเพื่อทดสอบ", filetypes=filetypes)
        if not p:
            return
        self._load_video_file(Path(p))

    def _load_quick_sample(self, num: int):
        self._stop_video()
        sample_path = PROJECT_DIR / "samples" / f"test{num if num > 1 else ''}.jpg"
        if not sample_path.is_file():
            sample_path = PROJECT_DIR / "samples" / "test.jpg"
        if sample_path.is_file():
            self._load_image_file(sample_path)
        else:
            messagebox.showwarning("ไม่พบไฟล์", f"ไม่พบไฟล์ตัวอย่าง: {sample_path}")

    # ---------------------------------------------------------
    # Helper Utilities
    # ---------------------------------------------------------
    @staticmethod
    def _truncate_filename(name: str, max_chars: int = 10) -> str:
        """ตัดทอนชื่อไฟล์ให้แสดงเฉพาะ 10 ตัวอักษรแรกแล้วตามด้วย ... เพื่อป้องกันไม่ให้ยืดหน้าจอ"""
        if len(name) > max_chars:
            return name[:max_chars] + "..."
        return name

    def _load_image_file(self, path: Path):
        self.current_media_type = "image"
        self.current_media_path = path
        short_name = self._truncate_filename(path.name, 10)
        self.source_label.configure(text=f"🖼️ {short_name}", foreground="#0284c7")
        self.footer_info.configure(text=f"กำลังเปิดภาพ: {short_name}")

        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", f"ไม่สามารถเปิดไฟล์รูปภาพได้: {path}")
            return
        self.current_cv_image = img
        self._reprocess_current_image()

    # ---------------------------------------------------------
    # Inference & Pre-processing Execution (Image)
    # ---------------------------------------------------------
    def _reprocess_current_image(self):
        if self.current_cv_image is None:
            return

        with self.pipeline_lock:
            pipe = self.pipeline

        if pipe is None:
            self.footer_info.configure(text="โมเดลยังโหลดไม่เสร็จ กรุณารอสักครู่...")
            return

        t0 = time.perf_counter()

        clahe_clip = float(self.clahe_slider.get())
        denoise_method = self.denoise_combo.get()
        enable_enh = self.enable_enh_var.get()

        enable_rect = self.enable_rect_var.get()
        rect_method = self.rect_combo.get()
        use_rect_char = self.use_rect_char_var.get()

        enable_bin = self.enable_bin_var.get()
        bin_method = self.bin_combo.get()

        conf_val = float(self.conf_slider.get())

        result: TLPRResult = pipe.process(
            image_or_path=self.current_cv_image,
            conf=conf_val,
            enable_enhancement=enable_enh,
            clahe_clip=clahe_clip,
            denoise_method=denoise_method,
            enable_rectification=enable_rect,
            rectify_method=rect_method,
            enable_binarization=enable_bin,
            binarize_method=bin_method,
            use_rectified_for_char=use_rect_char,
        )

        dt = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / dt if dt > 0 else 0.0

        self.last_result = result
        self.perf_info.configure(text=f"เวลาประมวลผล: {dt:.1f} ms | {fps:.1f} FPS")

        # Update Views
        self._update_ui_with_result(result, self.current_cv_image)

    # ---------------------------------------------------------
    # Video Processing Logic
    # ---------------------------------------------------------
    def _load_video_file(self, path: Path):
        self.current_media_type = "video"
        self.current_media_path = path
        short_name = self._truncate_filename(path.name, 10)
        self.source_label.configure(text=f"🎬 {short_name}", foreground="#7c3aed")

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"ไม่สามารถเปิดไฟล์วิดีโอได้: {path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.current_frame_idx = 0

        self.seek_slider.configure(to=max(1, self.total_frames - 1))
        self.seek_var.set(0)
        self.frame_counter_lbl.configure(text=f"Frame: 0 / {self.total_frames} ({self.video_fps} FPS)")

        ret, frame = self.cap.read()
        if ret:
            self.current_cv_image = frame
            self._reprocess_current_image()

        self.footer_info.configure(text=f"โหลดวิดีโอเรียบร้อย ({self.total_frames} เฟรม) - กด 'เล่น' เพื่อเริ่มประมวลผล")

    def _play_video(self):
        if self.cap is None or not self.cap.isOpened():
            return
        if self.is_playing:
            return

        self.is_playing = True
        self.stop_video_flag.clear()

        if self.save_video_var.get() and self.video_writer is None:
            self._start_video_writer()

        self.video_thread = threading.Thread(target=self._video_worker_loop, daemon=True)
        self.video_thread.start()
        self.btn_play.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.footer_info.configure(text="🎬 กำลังเล่นและประมวลผลวิดีโอ...")

    def _pause_video(self):
        self.is_playing = False
        self.stop_video_flag.set()
        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.footer_info.configure(text="⏸️ หยุดชั่วคราว")

    def _stop_video(self):
        self.is_playing = False
        self.stop_video_flag.set()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            messagebox.showinfo("บันทึกสำเร็จ", "บันทึกคลิปผลลัพธ์วิดีโอเรียบร้อยแล้ว!")

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.seek_var.set(0)
            self.frame_counter_lbl.configure(text=f"Frame: 0 / {self.total_frames}")

        self.btn_play.configure(state="normal")
        self.btn_pause.configure(state="normal")
        self.footer_info.configure(text="⏹️ หยุดการเล่นวิดีโอ")

    def _on_seek(self, val):
        if self.cap is None or not self.cap.isOpened() or self.is_playing:
            return
        target_f = int(float(val))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        self.current_frame_idx = target_f
        ret, frame = self.cap.read()
        if ret:
            self.current_cv_image = frame
            self.frame_counter_lbl.configure(text=f"Frame: {target_f} / {self.total_frames}")
            self._reprocess_current_image()

    def _get_skip_count(self) -> int:
        txt = self.skip_combo.get()
        if "1" in txt:
            return 1
        elif "2" in txt:
            return 2
        elif "3" in txt:
            return 3
        elif "5" in txt:
            return 5
        return 2

    def _start_video_writer(self):
        out_path = PROJECT_DIR / "runs" / f"output_{int(time.time())}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(out_path), fourcc, self.video_fps, (w, h))
        short_out = self._truncate_filename(out_path.name, 10)
        self.footer_info.configure(text=f"กำลังบันทึกผลลัพธ์ไปยัง: {short_out}")

    def _video_worker_loop(self):
        skip_step = self._get_skip_count()
        last_drawn_frame = None
        last_res = None

        while not self.stop_video_flag.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                self.after(0, self._stop_video)
                break

            self.current_frame_idx += 1
            is_detection_frame = (self.current_frame_idx % skip_step == 0)

            t0 = time.perf_counter()

            if is_detection_frame:
                with self.pipeline_lock:
                    pipe = self.pipeline

                if pipe is not None:
                    clahe_clip = float(self.clahe_slider.get())
                    denoise_method = self.denoise_combo.get()
                    enable_enh = self.enable_enh_var.get()
                    enable_rect = self.enable_rect_var.get()
                    rect_method = self.rect_combo.get()
                    use_rect_char = self.use_rect_char_var.get()
                    enable_bin = self.enable_bin_var.get()
                    bin_method = self.bin_combo.get()
                    conf_val = float(self.conf_slider.get())

                    last_res = pipe.process(
                        image_or_path=frame,
                        conf=conf_val,
                        enable_enhancement=enable_enh,
                        clahe_clip=clahe_clip,
                        denoise_method=denoise_method,
                        enable_rectification=enable_rect,
                        rectify_method=rect_method,
                        enable_binarization=enable_bin,
                        binarize_method=bin_method,
                        use_rectified_for_char=use_rect_char,
                    )
                    last_drawn_frame = last_res.annotated_vehicle if last_res.annotated_vehicle is not None else frame
            else:
                if last_drawn_frame is None:
                    last_drawn_frame = frame

            dt = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / dt if dt > 0 else 0.0

            if self.video_writer is not None and last_drawn_frame is not None:
                self.video_writer.write(last_drawn_frame)

            frame_to_show = last_drawn_frame if last_drawn_frame is not None else frame
            res_to_show = last_res
            cur_idx = self.current_frame_idx

            self.after(0, lambda f=frame_to_show, r=res_to_show, idx=cur_idx, d=dt, p=fps: self._sync_video_ui(f, r, idx, d, p))

            target_delay = max(0.001, (1.0 / self.video_fps) - (dt / 1000.0))
            time.sleep(target_delay)

    def _sync_video_ui(self, frame: np.ndarray, result: Optional[TLPRResult], frame_idx: int, dt: float, fps: float):
        self.seek_var.set(frame_idx)
        self.frame_counter_lbl.configure(text=f"Frame: {frame_idx} / {self.total_frames}")
        self.perf_info.configure(text=f"เวลาประมวลผล: {dt:.1f} ms | {fps:.1f} FPS")

        if result is not None:
            self._update_ui_with_result(result, frame)
        else:
            self._display_cv_image_on_canvas(frame, self.main_canvas)

    # ---------------------------------------------------------
    # Render & Display Methods
    # ---------------------------------------------------------
    def _update_ui_with_result(self, result: TLPRResult, base_image: np.ndarray):
        # 1. Update Recognition Banner
        if result.plates:
            best_plate: PlateDetection = result.plates[0]
            conf_str = f"({best_plate.confidence:.0%})"
            self.plate_text_lbl.configure(
                text=f"{best_plate.plate_text}  {conf_str}",
                fg="#38bdf8",
            )
            prov = best_plate.province_name or "ไม่พบจังหวัด"
            self.province_text_lbl.configure(text=f"📍 {prov}")

            # 2. Update Inspection Stages Cards (both overview & detail canvases)
            stage_data = {
                "raw": best_plate.raw_crop,
                "enhanced": best_plate.enhanced_crop,
                "rectified": best_plate.rectified_crop,
                "binarized": best_plate.binarized_crop,
            }

            for key, crop in stage_data.items():
                for cvs in self.stage_canvases.get(key, []):
                    self._display_crop_on_canvas(crop, cvs)

            for cvs in self.stage_canvases.get("chars", []):
                if best_plate.annotated_crop is not None:
                    self._display_pil_image_on_canvas(best_plate.annotated_crop, cvs)
                else:
                    self._clear_canvas(cvs)
        else:
            self.plate_text_lbl.configure(text="❌ ไม่พบป้ายทะเบียนในเฟรมนี้", fg="#f87171")
            self.province_text_lbl.configure(text="")
            for key in self.stage_canvases:
                for cvs in self.stage_canvases[key]:
                    self._clear_canvas(cvs)

        # 3. Update Main Canvas with Annotated Vehicle
        display_frame = result.annotated_vehicle if result.annotated_vehicle is not None else base_image
        self._display_cv_image_on_canvas(display_frame, self.main_canvas)

    def _display_cv_image_on_canvas(self, cv_img: Optional[np.ndarray], canvas: tk.Canvas):
        if cv_img is None or cv_img.size == 0:
            self._clear_canvas(canvas)
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 20 or ch < 20:
            cw, ch = 640, 420

        ih, iw = cv_img.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)
        canvas.image = tk_img

    def _display_crop_on_canvas(self, crop: Optional[np.ndarray], canvas: tk.Canvas):
        if crop is None or crop.size == 0:
            self._clear_canvas(canvas)
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 20 or ch < 20:
            cw, ch = 120, 70

        if len(crop.shape) == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        ih, iw = rgb.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)

        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)
        canvas.image = tk_img

    def _display_pil_image_on_canvas(self, pil_img: Optional[Image.Image], canvas: tk.Canvas):
        if pil_img is None:
            self._clear_canvas(canvas)
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 20 or ch < 20:
            cw, ch = 120, 70

        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor=tk.CENTER)
        canvas.image = tk_img

    def _clear_canvas(self, canvas: tk.Canvas):
        canvas.delete("all")
        canvas.image = None

    def _redraw_main_image(self):
        if self.last_result and self.last_result.annotated_vehicle is not None:
            self._display_cv_image_on_canvas(self.last_result.annotated_vehicle, self.main_canvas)
        elif self.current_cv_image is not None:
            self._display_cv_image_on_canvas(self.current_cv_image, self.main_canvas)

    def _save_snapshot(self):
        if self.last_result is None or self.last_result.annotated_vehicle is None:
            messagebox.showinfo("แจ้งเตือน", "ยังไม่มีภาพผลลัพธ์ที่จะบันทึก")
            return

        out_path = filedialog.asksaveasfilename(
            title="บันทึกภาพผลลัพธ์ Snapshot",
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
            initialfile=f"tlpr_result_{int(time.time())}.jpg",
        )
        if out_path:
            cv2.imwrite(out_path, self.last_result.annotated_vehicle)
            messagebox.showinfo("สำเร็จ", f"บันทึกภาพสำเร็จ:\n{out_path}")


def main():
    app = TLPRStudioGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
