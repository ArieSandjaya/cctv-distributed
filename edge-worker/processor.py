"""
People counting processor using Supervision + YOLO.
Supports RTSP streams and video files.

Now with optional face recognition integration.
"""

import cv2
import logging
import numpy as np
from supervision import Detections, LineZone, ByteTrack
from supervision.draw.color import ColorPalette
from supervision.geometry.core import Point
from ultralytics import YOLO

from face_rec import FaceRecognizer

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0


class PeopleCounter:
    """
    Processes video frames and counts people crossing a line.
    Optionally recognizes known faces.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        face_recognition_enabled: bool = True,
        face_tolerance: float = 0.5,
    ):
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.tracker = ByteTrack(minimum_matching_threshold=0.8)
        self.line_zone = LineZone(
            start=Point(0, 0),
            end=Point(0, 0),
        )
        self.frame_size: tuple[int, int] = (0, 0)
        self._last_detections: Detections | None = None

        # Face recognition
        self._face_enabled = face_recognition_enabled
        self._face_recognizer = FaceRecognizer(tolerance=face_tolerance)

        if face_recognition_enabled:
            try:
                self._face_recognizer.load_known_faces()
                if self._face_recognizer.is_active:
                    logger.info(f"Face recognition active: {self._face_recognizer.known_names}")
                else:
                    logger.info("Face recognition loaded but no known faces found.")
            except Exception as e:
                logger.warning(f"Face recognition init failed (non-blocking): {e}")

    # ── Public API ──────────────────────────────────────────────

    def set_line(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self.line_zone = LineZone(
            start=Point(start_x, start_y),
            end=Point(end_x, end_y),
        )

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (width, height)

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run inference + tracking + line counting + optional face recognition.
        Returns analytics data.
        """
        detections = self._run_inference(frame)
        self._last_detections = detections

        # Extract person bounding boxes (before tracking ids may change)
        person_bboxes = [map(int, detections.xyxy[i]) for i in range(len(detections))]

        # Face recognition
        recognized: dict[str, int] = {}
        if self._face_enabled and self._face_recognizer.is_active and len(detections) > 0:
            try:
                recognized = self._face_recognizer.process_frame(frame, person_bboxes)
            except Exception as e:
                logger.warning(f"Face rec error: {e}")

        return {
            "count_in": self.line_zone.in_count,
            "count_out": self.line_zone.out_count,
            "total_inside": max(0, self.line_zone.in_count - self.line_zone.out_count),
            "people_now": len(detections),
            "people_ids": detections.tracker_id.tolist() if detections.tracker_id is not None else [],
            "recognized": recognized,
        }

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes, tracking IDs, counting line, and face names."""
        annotated = frame.copy()
        detections = self._last_detections

        # Counting line
        start = self.line_zone.start
        end = self.line_zone.end
        cv2.line(annotated, (int(start.x), int(start.y)),
                 (int(end.x), int(end.y)), (0, 255, 0), 3)
        cv2.putText(annotated, "IN/OUT line",
                    (int(start.x) + 5, int(start.y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay counts
        y_offset = 40
        for text, val in [
            (f"IN:  {self.line_zone.in_count}", (0, 255, 0)),
            (f"OUT: {self.line_zone.out_count}", (0, 0, 255)),
            (f"INSIDE: {max(0, self.line_zone.in_count - self.line_zone.out_count)}", (255, 255, 0)),
        ]:
            cv2.putText(annotated, text, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, val, 2)
            y_offset += 35

        # Bounding boxes + names
        if detections is not None and len(detections) > 0:
            palette = ColorPalette.DEFAULT
            for i in range(len(detections)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                tid = detections.tracker_id[i] if detections.tracker_id is not None else None
                color = palette.by_idx(i).as_bgr()

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Try face recognition on this person (simplified inline label)
                label = f"ID:{tid}" if tid is not None else "person"

                if self._face_enabled and self._face_recognizer.is_active:
                    try:
                        import face_recognition as fr_
                        rgb_crop = cv2.cvtColor(frame[y1:y1+(y2-y1)//2, x1:x2], cv2.COLOR_BGR2RGB)
                        locs = fr_.face_locations(rgb_crop)
                        if locs:
                            t, r, b, l = max(locs, key=lambda f: (f[2]-f[0])*(f[1]-f[3]))
                            name = self._face_recognizer.recognize(rgb_crop[t:b, l:r])
                            if name != "Unknown":
                                label = name
                    except Exception:
                        pass

                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated

    # ── Internal ────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> Detections:
        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)

        person_mask = detections.class_id == PERSON_CLASS_ID
        detections = detections[person_mask]

        detections = self.tracker.update_with_detections(detections)
        return detections
