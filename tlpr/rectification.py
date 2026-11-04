import cv2
import numpy as np


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    จัดเรียงจุด 4 จุดของสี่เหลี่ยมตามลำดับ:
    0: บน-ซ้าย (Top-Left)
    1: บน-ขวา (Top-Right)
    2: ล่าง-ขวา (Bottom-Right)
    3: ล่าง-ซ้าย (Bottom-Left)
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # ผลรวม x + y: ค่าน้อยสุด = บน-ซ้าย, ค่ามากสุด = ล่าง-ขวา
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # ผลต่าง y - x: บน-ขวา มีค่าน้อยสุด, ล่าง-ซ้าย มีค่ามากสุด
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def rectify_license_plate(
    plate_crop: np.ndarray,
    method: str = "homography",
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """
    ปรับมุมมองและดึงภาพแผ่นป้ายทะเบียนให้กลับมาเป็นหน้าตรงและตั้งตรง (Plate Rectification / Deskewing)
    โดยใช้เทคนิค Homography (Perspective Transformation) หรือ Affine Transformation

    Args:
        plate_crop: ภาพ BGR ของแผ่นป้ายที่ถูกตัดออกมา (Cropped Plate)
        method:
            - 'homography': คำนวณ Homography Matrix 3x3 (Perspective Transform)
              แก้ปัญหามุมมอง 3 มิติ (กล้องเอียงข้าง/ก้มเงย)
            - 'affine': คำนวณ Affine Matrix 2x3 (Rotation Deskew)
              หมุนปรับระดับองศาในระนาบ 2D ให้ตั้งตรง

    Returns:
        tuple[np.ndarray, np.ndarray | None, float]:
            (ภาพที่ถูกปรับหน้าตรงแล้ว, Transformation Matrix, องศาการเอียงที่ตรวจพบ)
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, None, 0.0

    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)

    # เชื่อมเส้นขอบแผ่นป้ายที่อาจขาดตอน
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, None, 0.0

    # เลือกคอนทัวร์ที่มีพื้นที่ใหญ่ที่สุดซึ่งน่าจะเป็นกรอบแผ่นป้าย
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < (w * h * 0.08):
        return plate_crop, None, 0.0

    rect = cv2.minAreaRect(cnt)
    center, (rw, rh), angle = rect

    # ปรับมุมองศาการเอียงของ minAreaRect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0

    if method == "homography":
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        if len(approx) == 4:
            src_pts = approx.reshape(4, 2).astype(np.float32)
            src_pts = order_quad_points(src_pts)
        else:
            box = cv2.boxPoints(rect)
            src_pts = order_quad_points(box)

        (tl, tr, br, bl) = src_pts

        # คำนวณความกว้างและความสูงเป้าหมาย
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        max_w = int(max(w_top, w_bot))

        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        max_h = int(max(h_left, h_right))

        if max_w < 30 or max_h < 30:
            return plate_crop, None, angle

        # Destination points
        dst_pts = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        # Perspective Transform 3x3
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified = cv2.warpPerspective(
            plate_crop,
            H,
            (max_w, max_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if max_h > max_w * 1.3:
            rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

        return rectified, H, angle

    elif method == "affine":
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        rectified = cv2.warpAffine(
            plate_crop,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rectified, M, angle

    else:
        raise ValueError(
            f"Unknown rectify method: '{method}'. Choose 'homography' or 'affine'."
        )

