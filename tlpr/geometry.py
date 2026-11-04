def compute_iou(box1: tuple, box2: tuple) -> float:
    """คำนวณพื้นที่ทับซ้อน 2 มิติ (2D Intersection over Union) ระหว่าง 2 กล่อง (x1, y1, x2, y2)"""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_x_overlap(box1: tuple, box2: tuple) -> float:
    """คำนวณสัดส่วนการทับซ้อนในแนวนอน (1D Horizontal Overlap) เทียบกับกล่องที่แคบกว่า"""
    ix1, ix2 = max(box1[0], box2[0]), min(box1[2], box2[2])
    iw = max(0.0, ix2 - ix1)
    min_w = min(box1[2] - box1[0], box2[2] - box2[0])
    return iw / min_w if min_w > 0 else 0.0


def inter_class_nms(
    candidates: list,
    iou_thresh: float = 0.30,
    x_overlap_thresh: float = 0.45,
) -> list:
    """
    Inter-class Non-Maximum Suppression (IoU + 1D Horizontal Overlap Suppression)
    ตัดกล่องที่ทำนายทับซ้อนกันในตำแหน่งตัวอักษรเดียวกันข้ามคลาส โดยเลือกกล่องที่ Confidence สูงสุด
    """
    sorted_cands = sorted(candidates, key=lambda c: c[9], reverse=True)
    kept = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        kept.append(best)
        b_box = (best[0], best[1], best[2], best[3])
        rem = []
        for cand in sorted_cands:
            c_box = (cand[0], cand[1], cand[2], cand[3])
            iou = compute_iou(b_box, c_box)
            x_ov = compute_x_overlap(b_box, c_box)
            if iou > iou_thresh or x_ov > x_overlap_thresh:
                continue
            rem.append(cand)
        sorted_cands = rem
    return kept

