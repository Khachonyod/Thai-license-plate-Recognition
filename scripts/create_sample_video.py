import cv2
import numpy as np
from pathlib import Path

def create_sample_video():
    samples_dir = Path("samples")
    img_paths = sorted(list(samples_dir.glob("test*.jpg")))
    if not img_paths:
        print("No test images found.")
        return

    out_path = samples_dir / "sample_video.mp4"
    target_w, target_h = 1280, 720
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))

    for img_p in img_paths:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        # Resize/pad to 1280x720
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - nh) // 2
        x_off = (target_w - nw) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

        # Add 30 frames per image (approx 1.2 sec per vehicle)
        for _ in range(30):
            out.write(canvas)

    out.release()
    print(f"Created sample video: {out_path} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    create_sample_video()

