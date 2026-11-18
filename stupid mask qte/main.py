import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_FILES = {
    "left": BASE_DIR / "left.png",
    "right": BASE_DIR / "right.png",
    "up": BASE_DIR / "up.png",
    "down": BASE_DIR / "down.png",
}


# ============================================================
# CONFIG
# ============================================================
SCAN_INTERVAL = 0.035

KEY_COOLDOWN = 0.36
SAME_ARROW_EXTRA_COOLDOWN = 0.36
KEY_HOLD_TIME = 0.08
POST_PRESS_LOCKOUT = 0.22

# Wider search area because the new arrows are bigger/lower
# and the Roblox character can cover part of them.
ROI_X1 = 0.30
ROI_Y1 = 0.34
ROI_X2 = 0.70
ROI_Y2 = 0.82

# ============================================================
# NEW PURPLE ARROW DETECTION
# ============================================================
# OpenCV uses BGR, not RGB.
# These values are based on the newer purple screenshots.
TARGET_PURPLES_BGR = [
    np.array([244, 70, 186], dtype=np.int16),  # main purple, RGB approx (186, 70, 244)
    np.array([250, 70, 190], dtype=np.int16),  # bright highlight
    np.array([240, 70, 180], dtype=np.int16),  # bright fill variation
    np.array([230, 60, 175], dtype=np.int16),  # darker fill
    np.array([210, 60, 160], dtype=np.int16),  # darker purple
    np.array([190, 50, 140], dtype=np.int16),  # shadow
    np.array([160, 40, 120], dtype=np.int16),  # deep shadow
]

# Raise if it misses arrows. Lower if it catches wrong objects.
COLOR_TOLERANCE = 85

# HSV range for purple/violet. OpenCV hue range is 0-179.
# This helps with lighting, antialiasing, and slight purple changes.
PURPLE_HSV_LOWER = np.array([130, 45, 80], dtype=np.uint8)
PURPLE_HSV_UPPER = np.array([170, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 400

# Raise if wrong direction presses happen. Lower if it sees arrow but refuses to press.
MATCH_THRESHOLD = 0.48

# Turn this on only when tuning. Debug windows slow things down.
SHOW_DEBUG = False

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

pydirectinput.FAILSAFE = True
pydirectinput.PAUSE = 0.01


# ============================================================
# FILE CHECK
# ============================================================
def check_required_files():
    missing = [str(p) for p in TEMPLATE_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing template files:\n"
            + "\n".join(missing)
            + "\n\nPut left.png, right.png, up.png, and down.png in the same folder as this script."
        )


# ============================================================
# COLOR MASK
# ============================================================
def make_purple_mask(bgr_img):
    """
    Build a binary mask for the new purple arrows.
    Combines HSV purple detection with direct BGR target colors.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER)

    img16 = bgr_img.astype(np.int16)
    bgr_mask = np.zeros(bgr_img.shape[:2], dtype=np.uint8)

    for target in TARGET_PURPLES_BGR:
        diff = np.abs(img16 - target)
        dist = diff.sum(axis=2)
        target_mask = np.where(dist <= COLOR_TOLERANCE, 255, 0).astype(np.uint8)
        bgr_mask = cv2.bitwise_or(bgr_mask, target_mask)

    mask = cv2.bitwise_or(hsv_mask, bgr_mask)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ============================================================
# ROI
# ============================================================
def crop_roi(frame_bgr):
    h, w = frame_bgr.shape[:2]

    x1 = int(w * ROI_X1)
    y1 = int(h * ROI_Y1)
    x2 = int(w * ROI_X2)
    y2 = int(h * ROI_Y2)

    roi = frame_bgr[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


# ============================================================
# TEMPLATE LOADING
# ============================================================
def largest_contour_bbox(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None

    biggest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(biggest)


def load_templates():
    templates = {}

    for direction, path in TEMPLATE_FILES.items():
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load template: {path}")

        mask = make_purple_mask(img)
        bbox = largest_contour_bbox(mask, min_area=50)

        if bbox is None:
            raise RuntimeError(
                f"No purple arrow found in template: {path}\n"
                "Make sure left.png, right.png, up.png, and down.png are the new purple arrow templates."
            )

        x, y, w, h = bbox
        arrow_mask = mask[y:y + h, x:x + w]

        templates[direction] = {
            "mask": arrow_mask,
            "size": (w, h),
            "file": path.name,
        }

        print(f"[TEMPLATE] {direction:<5} loaded from {path.name} | size={w}x{h}")

    return templates


# ============================================================
# DETECT MAIN ARROW IN ROI
# ============================================================
def choose_best_purple_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask.shape[:2]
    roi_cx = w / 2.0
    roi_cy = h / 2.0

    best = None
    best_score = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        dist = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5

        # Prefer bigger objects near the middle/lower center.
        score = area - (dist * 3.0)

        if best is None or score > best_score:
            best = contour
            best_score = score

    return best


# ============================================================
# SHAPE MATCHING
# ============================================================
def compare_masks_iou(mask_a, mask_b):
    """
    Compare two binary masks of the same size.
    Returns overlap score from 0.0 to 1.0.
    """
    a = mask_a > 0
    b = mask_b > 0

    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def classify_arrow_by_template(arrow_mask, templates):
    best_direction = None
    best_score = -1.0

    for direction, tpl in templates.items():
        tpl_mask = tpl["mask"]

        resized_arrow = cv2.resize(
            arrow_mask,
            (tpl_mask.shape[1], tpl_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        score = compare_masks_iou(resized_arrow, tpl_mask)

        if score > best_score:
            best_score = score
            best_direction = direction

    return best_direction, best_score


def get_arrow_result(roi_bgr, templates):
    roi_mask = make_purple_mask(roi_bgr)
    contour = choose_best_purple_contour(roi_mask)

    if contour is None:
        return None, roi_mask, None

    x, y, w, h = cv2.boundingRect(contour)
    arrow_crop = roi_mask[y:y + h, x:x + w]

    # Reject fake detections that are too thin, too wide, too small, or too huge.
    # Real arrows are roughly square-ish, not long flat lines.
    if w < 70 or h < 70:
        return None, roi_mask, None

    if w > 240 or h > 240:
        return None, roi_mask, None

    aspect_ratio = w / float(h)

    if aspect_ratio < 0.55 or aspect_ratio > 1.85:
        return None, roi_mask, None

    direction, score = classify_arrow_by_template(arrow_crop, templates)

    if direction is None:
        return None, roi_mask, arrow_crop

    result = {
        "direction": direction,
        "score": score,
        "bbox": (x, y, w, h),
        "area": float(cv2.contourArea(contour)),
        "contour": contour,
        "below_threshold": score < MATCH_THRESHOLD,
    }

    return result, roi_mask, arrow_crop


# ============================================================
# INPUT
# ============================================================
def press_arrow(direction: str):
    pydirectinput.press(direction)
    time.sleep(KEY_HOLD_TIME)


# ============================================================
# MAIN
# ============================================================
def main():
    check_required_files()
    templates = load_templates()

    last_press_time = 0.0
    last_direction = None
    last_same_direction_time = 0.0

    print("[INFO] Starting.")
    print("[INFO] Using NEW purple color mask + template shape matching.")
    print("[INFO] Keypress cooldown is about 0.40 seconds.")
    print("[INFO] Put Roblox in focus.")
    print("[INFO] Move mouse to top-left corner to emergency stop.")
    if SHOW_DEBUG:
        print("[INFO] Press ESC in an OpenCV debug window to quit.")
    print()

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            raw = np.array(sct.grab(monitor))
            frame_bgr = raw[:, :, :3]

            roi_bgr, _ = crop_roi(frame_bgr)
            result, roi_mask, arrow_crop = get_arrow_result(roi_bgr, templates)

            now = time.time()

            if result is not None:
                direction = result["direction"]
                score = result["score"]
                x, y, w, h = result["bbox"]
                below_threshold = result["below_threshold"]

                if below_threshold:
                    print(f"[SEEN ] {direction.upper():<5} | score={score:.3f} | below threshold")
                else:
                    can_press = (now - last_press_time) >= KEY_COOLDOWN

                    if direction == last_direction:
                        can_press = can_press and (
                            (now - last_same_direction_time) >= SAME_ARROW_EXTRA_COOLDOWN
                        )

                    if can_press:
                        press_arrow(direction)

                        last_press_time = now
                        last_same_direction_time = now
                        last_direction = direction

                        print(
                            f"[PRESS] {direction.upper():<5} | "
                            f"score={score:.3f} | bbox=({x},{y},{w},{h})"
                        )

                        time.sleep(POST_PRESS_LOCKOUT)
                    else:
                        print(
                            f"[HOLD ] {direction.upper():<5} | "
                            f"score={score:.3f} | bbox=({x},{y},{w},{h})"
                        )
            else:
                print("[NONE ] no purple arrow object found")

            if SHOW_DEBUG:
                debug_view = roi_bgr.copy()

                h, w = debug_view.shape[:2]
                cv2.line(debug_view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                cv2.line(debug_view, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

                if result is not None:
                    contour = result["contour"]
                    x, y, bw, bh = result["bbox"]
                    color = (0, 255, 0) if not result["below_threshold"] else (0, 255, 255)

                    cv2.drawContours(debug_view, [contour], -1, color, 2)
                    cv2.rectangle(debug_view, (x, y), (x + bw, y + bh), color, 2)
                    cv2.putText(
                        debug_view,
                        f"{result['direction']} {result['score']:.2f}",
                        (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("ROI", debug_view)
                cv2.imshow("ROI Purple Mask", roi_mask)

                if arrow_crop is not None and arrow_crop.size > 0:
                    cv2.imshow("Detected Arrow", arrow_crop)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(SCAN_INTERVAL)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
