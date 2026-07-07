"""Crop a detection bbox out of a frame and JPEG-encode it for ANPR/storage."""

import cv2


def crop_to_jpeg(frame_bgr, box):
    """Return JPEG bytes of the box region, clamped to the frame bounds."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    crop = frame_bgr[y1:y2, x1:x2]
    ok, buffer = cv2.imencode(".jpg", crop)
    if not ok:
        return b""
    return buffer.tobytes()
