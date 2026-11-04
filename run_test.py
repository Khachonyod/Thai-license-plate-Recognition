import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from tlpr.constants import CLASS_NAME_TO_CHAR, DEFAULT_CHAR_MODEL, PROJECT_DIR
from tlpr.syntax import parse_thai_license_syntax
from tlpr.visualization import draw_annotated_chars

DEFAULT_MODEL = DEFAULT_CHAR_MODEL
DEFAULT_DATASET = PROJECT_DIR / "thai-char-v4" / "train"
DEFAULT_LOG_DIR = PROJECT_DIR / "run_test_log"


def get_ground_truth(label_path: Path, class_names: list) -> str:
    """
    อ่านและจัดเรียงตัวอักษรจริง (Ground Truth) จากไฟล์ .txt ของ YOLO Label
    โดยกรองกล่องจังหวัด (Class 5) และกล่องที่อยู่ครึ่งล่างของป้ายออก
    """
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
            # ข้ามกล่องจังหวัด (cid 5) หรือตัวหนังสือที่อยู่ครึ่งล่าง
            if cid == 5 or yc > 0.60:
                continue
            cls_name = class_names[cid]
            char_text = CLASS_NAME_TO_CHAR.get(cls_name, cls_name)
            chars.append((xc, char_text))

    # เรียงจากซ้ายไปขวาตามแกน X
    chars.sort(key=lambda item: item[0])
    return "".join(c[1] for c in chars)


def save_markdown_log(log_dir: Path, meta: dict, summary: dict, records: list) -> Path:
    """
    บันทึก Log ผลการทดสอบทั้งหมดเป็นไฟล์ Markdown (.md)
    พร้อมตารางสรุปสถิติ เปอร์เซ็นต์ความถูกต้อง และ Path ของไฟล์รูปภาพทั้งหมด
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = meta["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = meta["timestamp"].strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_log_{file_timestamp}.md"
    latest_file = log_dir / "latest.md"

    wrong_records = [r for r in records if not r["is_match"]]

    md_lines = [
        "# 📋 รายงานผลการทดสอบโมเดล Character Detector (Test Run Log)",
        "",
        f"- **วัน-เวลาที่ทดสอบ:** {timestamp_str}",
        f"- **โมเดลที่ใช้ทดสอบ:** `{meta['model_name']}` (`{meta['model_path']}`)",
        f"- **ชุดข้อมูลที่ทดสอบ:** `{meta['dataset_path']}`",
        f"- **อุปกรณ์ประมวลผล (Device):** `{meta['device_name']}`",
        f"- **Confidence Threshold:** `{meta['conf_thresh']}`",
        f"- **วิธีการเลือกภาพ:** {'สุ่มเลือก (Random)' if meta['random_sample'] else 'เรียงตามลำดับชื่อไฟล์ (Sequential)'}",
        "",
        "---",
        "",
        "## 📊 สรุปสถิติผลการประเมิน (Evaluation Summary)",
        "",
        "| ตัวชี้วัด (Metric) | จำนวน / ค่าที่ได้ | ร้อยละ (Percentage) |",
        "| :--- | :--- | :--- |",
        f"| **จำนวนภาพที่ทดสอบทั้งหมด** | **{summary['total_tests']}** รูปภาพ | 100.00% |",
        f"| **✅ อ่านถูกต้องทั้งหมด (MATCH)** | **{summary['correct_plates']}** รูปภาพ | **{summary['plate_acc']:.2f}%** |",
        f"| **❌ อ่านผิดบางตัว/ทั้งหมด (WRONG)** | **{summary['wrong_plates']}** รูปภาพ | **{summary['wrong_acc']:.2f}%** |",
        f"| **ความแม่นยำรายตัวอักษร (Character Accuracy)** | **{summary['correct_chars']} / {summary['total_chars']}** ตัว | **{summary['char_acc']:.2f}%** |",
        f"| **ความเร็วในการประมวลผลเฉลี่ย** | **{summary['avg_speed_ms']:.1f}** ms/รูป | **{summary['fps']:.1f}** FPS |",
        "",
        "---",
        "",
    ]

    # ส่วนรายการที่อ่านผิด (เพื่อให้อ่านวิเคราะห์ได้ทันที)
    md_lines.extend([
        f"## ⚠️ รายการภาพที่อ่านผิด ({len(wrong_records)} รูปภาพ)",
        "",
    ])
    if wrong_records:
        md_lines.extend([
            "| # | ชื่อไฟล์ | ค่าจริง (GT) | ผลทำนาย (Pred) | Conf เฉลี่ย | พาธไฟล์ (File Path) |",
            "| :-: | :--- | :--- | :--- | :-: | :--- |",
        ])
        for r in wrong_records:
            md_lines.append(
                f"| {r['idx']:02d} | `{r['file_name']}` | **{r['gt']}** | `{r['pred']}` | {r['avg_conf']:.1%} | `{r['file_path']}` |"
            )
        md_lines.append("")
    else:
        md_lines.append("🎉 *ยอดเยี่ยม! ไม่มีรูปภาพที่อ่านผิดเลยในการทดสอบชุดนี้*\n")

    md_lines.extend([
        "---",
        "",
        "## 📑 ตารางบันทึกผลการทดสอบทุกไฟล์ (All Test Records)",
        "",
        "| # | สถานะ | ชื่อไฟล์ | ค่าจริง (GT) | ผลทำนาย (Pred) | Conf เฉลี่ย | รายละเอียด Conf แต่ละตัว | พาธไฟล์เต็ม (Full Path) |",
        "| :-: | :-: | :--- | :--- | :--- | :-: | :--- | :--- |",
    ])

    for r in records:
        status_icon = "✅ MATCH" if r["is_match"] else "❌ WRONG"
        conf_detail_str = "<br>".join(f"{ch}: {c:.1%}" for ch, c in r["char_confs"]) if r["char_confs"] else "-"
        md_lines.append(
            f"| {r['idx']:02d} | {status_icon} | `{r['file_name']}` | **{r['gt']}** | `{r['pred']}` | {r['avg_conf']:.1%} | {conf_detail_str} | `{r['file_path']}` |"
        )

    md_content = "\n".join(md_lines) + "\n"
    log_file.write_text(md_content, encoding="utf-8")
    latest_file.write_text(md_content, encoding="utf-8")

    return log_file


def main():
    parser = argparse.ArgumentParser(
        description="ทดสอบความแม่นยำของโมเดล Character Detector บนภาพป้ายทะเบียนที่ Crop แล้ว พร้อมบันทึก Log ลงใน run_test_log/"
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path ของไฟล์โมเดลที่ต้องการทดสอบ (Default: {DEFAULT_MODEL.name})",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path โฟลเดอร์ชุดข้อมูล (ต้องมี images/ และ labels/ ข้างใน) (Default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="จำนวนรูปภาพที่ต้องการทดสอบ (Default: 50)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Confidence Threshold สำหรับ Character Detector (Default: 0.05)",
    )
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="อุปกรณ์ประมวลผล ('0' สำหรับ GPU หรือ 'cpu')",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="สุ่มเลือกภาพทดสอบแทนการเลือกภาพแรกๆ เรียงตามชื่อ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed สำหรับการสุ่มภาพ (Default: 42)",
    )
    parser.add_argument(
        "--save-errors",
        action="store_true",
        help="บันทึกภาพและ Bounding Box ของเคสที่อ่านผิดไว้ในโฟลเดอร์ runs/test_errors/",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help=f"โฟลเดอร์สำหรับบันทึก Log (Default: {DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        alt = PROJECT_DIR / "weights" / "char_detector.pt"
        if alt.is_file():
            model_path = alt
        else:
            raise SystemExit(f"❌ ไม่พบไฟล์โมเดล: {model_path}")

    dataset_dir = Path(args.dataset)
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"❌ โฟลเดอร์ชุดข้อมูลไม่ถูกต้อง (ต้องมี images/ และ labels/): {dataset_dir}")

    valid_pairs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in img_dir.glob(ext):
            lp = lbl_dir / f"{p.stem}.txt"
            if lp.is_file():
                valid_pairs.append((p, lp))

    if not valid_pairs:
        raise SystemExit(f"❌ ไม่พบภาพคู่กับ label ใน: {dataset_dir}")

    valid_pairs.sort(key=lambda x: x[0].name)

    test_count = min(args.count, len(valid_pairs))
    if args.random_sample:
        random.seed(args.seed)
        selected_pairs = random.sample(valid_pairs, test_count)
    else:
        selected_pairs = valid_pairs[:test_count]

    error_dir = PROJECT_DIR / "runs" / "test_errors"
    if args.save_errors:
        error_dir.mkdir(parents=True, exist_ok=True)

    device_desc = (
        f"GPU: {torch.cuda.get_device_name(0)}"
        if args.device != "cpu" and torch.cuda.is_available()
        else "CPU"
    )

    print("=" * 75)
    print("🚀 เริ่มต้นการทดสอบ Character Detection Accuracy & Logging")
    print(f"🎯 โมเดล:          {model_path.name}")
    print(f"📁 ชุดข้อมูล:       {dataset_dir}")
    print(f"🖼️  จำนวนภาพ:      {test_count} รูปภาพ ({'สุ่มเลือก' if args.random_sample else 'เรียงตามชื่อ'})")
    print(f"⚙️  Confidence:     {args.conf}")
    print(f"💻 Device:         {args.device} ({device_desc})")
    print("=" * 75)

    char_model = YOLO(str(model_path))
    class_names = char_model.names

    correct_plates = 0
    total_chars_gt = 0
    correct_chars_cnt = 0
    total_time_ms = 0.0

    test_records = []
    errors_list = []

    for idx, (img_p, lbl_p) in enumerate(selected_pairs, start=1):
        gt_raw = get_ground_truth(lbl_p, class_names)
        gt_clean = gt_raw.replace(" ", "")

        crop = cv2.imread(str(img_p))
        if crop is None:
            continue

        t0 = time.perf_counter()
        pred_text, _, _, _, kept_boxes = parse_thai_license_syntax(
            char_model=char_model,
            plate_crop=crop,
            conf_thresh=args.conf,
            device=args.device,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_time_ms += dt_ms

        pred_clean = pred_text.replace(" ", "")

        total_chars_gt += len(gt_clean)
        char_matches = sum(1 for g, p in zip(gt_clean, pred_clean) if g == p)
        correct_chars_cnt += char_matches

        is_match = (gt_clean == pred_clean) and (len(gt_clean) > 0)
        if is_match:
            correct_plates += 1
            status_tag = "✅ MATCH"
        else:
            status_tag = "❌ WRONG"
            errors_list.append({
                "idx": idx,
                "file": img_p.name,
                "gt": gt_clean,
                "pred": pred_clean,
                "boxes": kept_boxes,
            })
            if args.save_errors:
                annotated = draw_annotated_chars(crop, kept_boxes)
                save_p = error_dir / f"err_{idx:02d}_{img_p.stem}.jpg"
                annotated.save(str(save_p))

        avg_conf = (
            sum(b[5] for b in kept_boxes) / len(kept_boxes) if kept_boxes else 0.0
        )
        char_confs = [(b[4], b[5]) for b in kept_boxes]
        conf_str = " | ".join(f"{ch}:{c:.0%}" for ch, c in char_confs)

        test_records.append({
            "idx": idx,
            "file_name": img_p.name,
            "file_path": str(img_p.resolve()),
            "gt": gt_clean,
            "pred": pred_clean,
            "is_match": is_match,
            "avg_conf": avg_conf,
            "char_confs": char_confs,
            "dt_ms": dt_ms,
        })

        print(
            f"[{idx:02d}/{test_count:02d}] {status_tag} | "
            f"GT: {gt_clean:<8s} | "
            f"Pred: {pred_clean:<8s} | "
            f"Conf: {avg_conf:.1%} | "
            f"Time: {dt_ms:.1f}ms"
        )
        if not is_match:
            print(f"       ↳ ความมั่นใจรายตัว: [{conf_str}]")

    print("=" * 75)
    plate_acc = (correct_plates / test_count) * 100.0 if test_count > 0 else 0.0
    wrong_plates = test_count - correct_plates
    wrong_acc = 100.0 - plate_acc
    char_acc = (correct_chars_cnt / total_chars_gt) * 100.0 if total_chars_gt > 0 else 0.0
    avg_speed = total_time_ms / test_count if test_count > 0 else 0.0
    fps = 1000.0 / (avg_speed + 1e-5)

    summary_data = {
        "total_tests": test_count,
        "correct_plates": correct_plates,
        "wrong_plates": wrong_plates,
        "plate_acc": plate_acc,
        "wrong_acc": wrong_acc,
        "total_chars": total_chars_gt,
        "correct_chars": correct_chars_cnt,
        "char_acc": char_acc,
        "avg_speed_ms": avg_speed,
        "fps": fps,
    }

    meta_data = {
        "timestamp": datetime.now(),
        "model_name": model_path.name,
        "model_path": str(model_path.resolve()),
        "dataset_path": str(dataset_dir.resolve()),
        "device_name": f"{args.device} ({device_desc})",
        "conf_thresh": args.conf,
        "random_sample": args.random_sample,
    }

    print("📊 สรุปผลการประเมิน (Evaluation Summary)")
    print(f"   ► จำนวนป้ายที่อ่านถูกต้องทั้งแผ่น: {correct_plates}/{test_count} รูปภาพ ({plate_acc:.2f}%)")
    print(f"   ► จำนวนป้ายที่อ่านผิด:          {wrong_plates}/{test_count} รูปภาพ ({wrong_acc:.2f}%)")
    print(f"   ► Character-Level Accuracy:    {char_acc:.2f}% ({correct_chars_cnt}/{total_chars_gt} ตัว)")
    print(f"   ► ความเร็วเฉลี่ยต่อรูปภาพ:       {avg_speed:.1f} ms ({fps:.1f} FPS)")
    print("=" * 75)

    # บันทึกไฟล์ Log ลง run_test_log/
    log_dir = Path(args.log_dir)
    saved_log_path = save_markdown_log(log_dir, meta_data, summary_data, test_records)
    print(f"\n📝 บันทึก Log ผลการทดสอบทั้งหมดแล้วที่:")
    print(f"   ► ไฟล์ประวัติ: {saved_log_path}")
    print(f"   ► ไฟล์ล่าสุด:  {log_dir / 'latest.md'}")

    if errors_list:
        print(f"\n⚠️  รายการป้ายที่อ่านผิด ({len(errors_list)} ภาพ):")
        for err in errors_list[:10]:
            print(f"   - [#{err['idx']:02d}] {err['file'][:30]}... | จริง: {err['gt']} | ทาย: {err['pred']}")
        if len(errors_list) > 10:
            print(f"   ... และอีก {len(errors_list) - 10} ภาพ (ดูรายละเอียดทั้งหมดได้ในไฟล์ Log .md)")
        if args.save_errors:
            print(f"\n📁 บันทึกภาพที่อ่านผิดพร้อม Bounding Box ไว้ที่: {error_dir}")


if __name__ == "__main__":
    main()
