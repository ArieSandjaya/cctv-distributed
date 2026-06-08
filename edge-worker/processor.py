"""
People counting processor using Supervision + YOLO.
Now with PolygonZone (ROI) instead of LineZone.
"""

import cv2
import logging
import numpy as np
from typing import Optional
from supervision import Detections, PolygonZone, ByteTrack
from supervision.draw.color import ColorPalette
from supervision.geometry.core import Point, Polygon
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0


class PeopleCounter:
    """
    Processes video frames and counts people inside a polygon zone (ROI).
    Configurable remotely via MQTT/Central Server.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        face_recognition_enabled: bool = True,
        face_tolerance: float = 0.5,
    ):
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)

        self.tracker = ByteTrack(minimum_matching_threshold=0.8)

        # Default ROI (full frame, will be overwritten by config)
        self.polygon_zone: PolygonZone | None = None
        self._roi_points: list[tuple[int, int]] = []
        self.frame_size: tuple[int, int] = (640, 480)

        self._last_detections: Detections | None = None
        self._last_annotated_frame: np.ndarray | None = None

        # Face recognition
        self._face_enabled = face_recognition_enabled
        self._face_recognizer = None
        if face_recognition_enabled:
            try:
                from face_rec import FaceRecognizer
                self._face_recognizer = FaceRecognizer(tolerance=face_tolerance)
                self._face_recognizer.load_known_faces()
                if self._face_recognizer.is_active:
                    logger.info(f"Face rec active: {self._face_recognizer.known_names}")
            except Exception as e:
                logger.warning(f"Face rec init failed: {e}")

    # ── ROI Config ─────────────────────────────────────────────

    def set_roi(self, points: list[list[int]] | list[tuple[int, int]]) -> None:
        """
        Set ROI polygon from list of [x,y] or (x,y) points.
        Example: [[100,100], [500,100], [400,400], [200,400]]
        """
        self._roi_points = [(int(p[0]), int(p[1])) for p in points]
        polygon = Polygon([Point(x, y) for x, y in self._roi_points])
        self.polygon_zone = PolygonZone(
            polygon=polygon,
            frame_resolution_wh=self.frame_size,
        )
        logger.info(f"ROI updated: {len(points)} points")

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (width, height)
        if self._roi_points:
            self.set_roi(self._roi_points)

    # ── Processing ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run inference → filter person → track → check ROI → face rec."""
        detections = self._run_inference(frame)
        self._last_detections = detections

        # Count people inside ROI
        zone_count = 0
        if self.polygon_zone is not None and len(detections) > 0:
            # PolygonZone.trigger returns boolean mask for inside/outside
            mask = self.polygon_zone.trigger(detections)
            inside = detections[mask]
            zone_count = len(inside)
        else:
            zone_count = len(detections)

        # Face recognition on people inside ROI
        recognized: dict[str, int] = {}
        if self._face_enabled and self._face_recognizer and self._face_recognizer.is_active and len(detections) > 0:
            try:
                person_bboxes = [tuple(map(int, detections.xyxy[i])) for i in range(len(detections))]
                recognized = self._face_recognizer.process_frame(frame, person_bboxes)
            except Exception as e:
                logger.warning(f"Face rec error: {e}")

        return {
            "people_inside": zone_count,
            "people_tracked": len(detections),
            "recognized": recognized,
        }

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI polygon + bounding boxes + labels."""
        annotated = frame.copy()
        detections = self._last_detections

        # Draw ROI polygon
        if self._roi_points:
            pts = np.array(self._roi_points, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            # Semi-transparent fill
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255, 40))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Label "ROI" at top-left of polygon
            cx = int(np.mean([p[0] for p in self._roi_points]))
            cy = int(np.mean([p[1] for p in self._roi_points]))
            cv2.putText(annotated, "ROI", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Overlay count
        count_text = f"Inside ROI: {len(detections) if self.polygon_zone is None else '?'}"
        cv2.putText(annotated, count_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Bounding boxes
        if detections is not None and len(detections) > 0:
            palette = ColorPalette.DEFAULT

            # Check which are inside ROI
            inside_mask = None
            if self.polygon_zone is not None:
                try:
                    inside_mask = self.polygon_zone.trigger(detections)
                except:
                    pass

            for i in range(len(detections)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                tid = detections.tracker_id[i] if detections.tracker_id is not None else None
                is_inside = inside_mask[i] if inside_mask is not None else True

                color = (0, 255, 0) if is_inside else (100, 100, 100)  # green if inside, grey if outside
                label = f"ID:{tid}" if tid is not None else "person"

                # Try face name
                if self._face_enabled and self._face_recognizer and is_inside:
                    try:
                        import face_recognition as fr_
                        rgb_crop = cv2.cvtColor(frame[y1:y1+(y2-y1)//2, x1:x2], cv2.COLOR_BGR2RGB)
                        locs = fr_.face_locations(rgb_crop)
                        if locs:
                            t, r, b, l = max(locs, key=lambda f: (f[2]-f[0])*(f[1]-f[3]))
                            name = self._face_recognizer.recognize(rgb_crop[t:b, l:r])
                            if name != "Unknown":
                                label = name
                    except:
                        pass

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self._last_annotated_frame = annotated
        return annotated

    # ── Internal ────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> Detections:
        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[detections.class_id == PERSON_CLASS_ID]
        detections = self.tracker.update_with_detections(detections)
        return detections
