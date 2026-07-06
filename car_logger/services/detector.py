"""The entire CV layer: a thin wrapper over jetson.inference SSD-Mobilenet-v2.

jetson.inference/jetson.utils are imported lazily inside __init__ so importing
this module (e.g. during test collection or on a non-Jetson box) does not
require CUDA to be present."""

from collections import namedtuple

import cv2

Detection = namedtuple(
    "Detection", ["x1", "y1", "x2", "y2", "confidence", "class_id"]
)

# COCO class ids we keep: 3=car, 4=motorcycle, 6=bus, 8=truck.
VEHICLE_CLASS_IDS = frozenset([3, 4, 6, 8])


class Detector(object):
    def __init__(self, threshold=0.5):
        import jetson.inference
        import jetson.utils
        self._utils = jetson.utils
        self._net = jetson.inference.detectNet(
            "ssd-mobilenet-v2", threshold=threshold
        )

    def detect(self, frame_bgr):
        """Run detection on one BGR frame; return vehicle Detections only."""
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        cuda_img = self._utils.cudaFromNumpy(rgba)
        raw = self._net.Detect(cuda_img, overlay="none")
        results = []
        for d in raw:
            if int(d.ClassID) not in VEHICLE_CLASS_IDS:
                continue
            results.append(Detection(
                int(d.Left), int(d.Top), int(d.Right), int(d.Bottom),
                float(d.Confidence), int(d.ClassID),
            ))
        return results
