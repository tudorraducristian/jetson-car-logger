"""Local two-stage ANPR client: plate detection -> OCR -> vote.

Drop-in replacement for the v1 cloud client behind the same PlateResult
contract. Engines are injected so unit tests run with fakes — no models,
no Jetson (the same pattern as the injected httpx client before it).

"Never raises" contract preserved: corrupt image, missing model file,
engine exception -> PlateResult(None, None, "failed", None); the worker
loop's defense-in-depth stays as the second net."""

import logging

import cv2
import numpy as np

from car_logger.services.plate_result import PlateResult
from car_logger.services.plate_rules import normalize_plate
from car_logger.services.plate_voting import vote_on_reads

log = logging.getLogger(__name__)


class LocalAnprClient(object):
    def __init__(self, detector_engine, ocr_engine):
        self._detector = detector_engine
        self._ocr = ocr_engine

    def close(self):
        """Release the engines (symmetric with the v1 client's close)."""
        for engine in (self._detector, self._ocr):
            close = getattr(engine, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                log.exception("engine close failed")

    def read_plate(self, image_bytes):
        """One vehicle crop (JPEG bytes) -> one PlateResult. Never raises."""
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                return PlateResult(None, None, "failed", None)
            box = self._detector.detect_plate(image)
            if box is None:
                # Stage 1 found nothing -> OCR never runs (spec requirement)
                return PlateResult(None, None, "no_plate", None)
            x1, y1, x2, y2 = box
            text, confidence, region = self._ocr.read(image[y1:y2, x1:x2])
            text = normalize_plate(text)
            if not text:
                # a plate was seen but nothing decodable came off it
                return PlateResult(None, None, "failed", None)
            return PlateResult(text, confidence, "success", region)
        except Exception:
            log.exception("local ANPR read failed")
            return PlateResult(None, None, "failed", None)

    def read_plate_multi(self, crops):
        """N crops of one track -> (verdict, evidence_bytes). Never raises.

        evidence_bytes is the winning read's crop — the image saved for
        the event matches the text it claims; first crop otherwise."""
        try:
            reads = [self.read_plate(crop) for crop in crops]
            result, winner_index = vote_on_reads(reads)
            evidence = crops[winner_index] if crops else b""
            return result, evidence
        except Exception:
            log.exception("local ANPR multi-read failed")
            return (PlateResult(None, None, "failed", None),
                    crops[0] if crops else b"")
