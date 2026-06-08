"""
Feature-based processor: counting | face_rec | heatmap | trajectory
"""

import cv2
import logging
import numpy as np
import time
from typing import Optional, Dict, Any
from supervision import Detections, PolygonZone, ByteTrack
from supervision.geometry.core import Point, Polygon
from ultralytics import YOLO

logger = logging.getLogger(__name__)
PERSON_CLASS_ID = 0
MAX_TRACK_AGE = 3.0  # seconds before a track is considered "lost"


class PeopleCounter:
    def __init__(self, feature: str = "counting", model_path: str = "yolov8n.pt"):
        self.feature = feature
        self.model = YOLO(model_path)
        self.tracker = ByteTrack(minimum_matching_threshold=0.8)
        self.frame_size = (640, 480)

        # Shared
        self._roi_points = []
        self.polygon_zone: Optional[PolygonZone] = None

        # Face rec
        self._face_recognizer = None
        if feature == "face_rec":
            try:
                from face_rec import FaceRecognizer
                self._face_recognizer = FaceRecognizer()
                self._face_recognizer.load_known_faces()
                logger.info("FaceRec loaded.")
            except Exception as e:
                logger.warning(f"FaceRec failed: {e}")

        # Heatmap
        self.heatmap_grid = np.zeros((24, 32), dtype=np.float32)
        self.heatmap_frames = 0

        # Trajectory (OD Matrix)
        self.trajectory_zones: Dict[str, PolygonZone] = {}
        self.trajectory_labels: list[str] = []
        self.person_history: Dict[int, dict] = {}
        self.od_matrix: Dict[str, Dict[str, int]] = {}

    # ── Config ──────────────────────────────────────────────────

    def set_roi(self, points: list):
        self._roi_points = points
        polygon = Polygon([Point(x, y) for x, y in points])
        self.polygon_zone = PolygonZone(polygon=polygon, frame_resolution_wh=self.frame_size)
        logger.info(f"ROI set ({len(points)} pts)")

    def set_trajectory_zones(self, zones: Dict[str, list]):
        """zones = {"North": [[x,y], ...], "South": [[x,y], ...]}"""
        self.trajectory_zones = {}
        self.trajectory_labels = list(zones.keys())
        self.od_matrix = {o: {d: 0 for d in self.trajectory_labels} for o in self.trajectory_labels}
        for label, pts in zones.items():
            polygon = Polygon([Point(x, y) for x, y in pts])
            self.trajectory_zones[label] = PolygonZone(polygon=polygon, frame_resolution_wh=self.frame_size)
        logger.info(f"Trajectory zones set: {self.trajectory_labels}")

    def set_frame_size(self, w, h):
        self.frame_size = (w, h)
        if self._roi_points:
            self.set_roi(self._roi_points)
        if self.trajectory_zones:
            zones = {k: v.polygon for k, v in self.trajectory_zones.items()}
            self.set_trajectory_zones(zones)

    # ── Processing ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[detections.class_id == PERSON_CLASS_ID]
        detections = self.tracker.update_with_detections(detections)

        now = time.time()

        # ── FEATURE: TRAJECTORY ──────────────────────────────
        if self.feature == "trajectory" and self.trajectory_labels:
            tracked_ids = set()
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                tracked_ids.add(tid)
                if tid not in self.person_history:
                    self.person_history[tid] = {"origin": None, "last_zone": None, "last_seen": now}

                # Find current zone
                current_zone = None
                for label in self.trajectory_labels:
                    zone = self.trajectory_zones[label]
                    try:
                        mask = zone.trigger(detections[i])
                        if mask:
                            current_zone = label
                            break
                    except:
                        pass

                if current_zone:
                    if self.person_history[tid]["origin"] is None:
                        self.person_history[tid]["origin"] = current_zone
                    self.person_history[tid]["last_zone"] = current_zone
                self.person_history[tid]["last_seen"] = now

            # Cleanup expired tracks & record OD
            expired = [tid for tid, rec in self.person_history.items()
                       if tid not in tracked_ids and now - rec["last_seen"] > MAX_TRACK_AGE]
            for tid in expired:
                rec = self.person_history.pop(tid)
                origin = rec["origin"]
                dest = rec["last_zone"]
                if origin and dest and origin != dest:
                    if origin in self.od_matrix and dest in self.od_matrix[origin]:
                        self.od_matrix[origin][dest] += 1

            return {"trajectory_labels": self.trajectory_labels, "od_matrix": self.od_matrix,
                    "people_tracked": len(tracked_ids)}

        # ── FEATURE: COUNTING ────────────────────────────────
        if self.feature == "counting":
            count = 0
            if self.polygon_zone and len(detections) > 0:
                mask = self.polygon_zone.trigger(detections)
                count = len(detections[mask])
            return {"people_inside": count, "total_detected": len(detections)}

        # ── FEATURE: FACE REC ────────────────────────────────
        if self.feature == "face_rec":
            recognized = {}
            if self._face_recognizer and len(detections) > 0:
                bboxes = [tuple(map(int, detections.xyxy[i])) for i in range(len(detections))]
                recognized = self._face_recognizer.process_frame(frame, bboxes)
            return {"recognized": recognized, "total_detected": len(detections)}

        # ── FEATURE: HEATMAP ─────────────────────────────────
        if self.feature == "heatmap":
            if len(detections) > 0:
                for bbox in detections.xyxy:
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, y2
                    grid_x = int(np.clip(cx / self.frame_size[0] * 32, 0, 31))
                    grid_y = int(np.clip(cy / self.frame_size[1] * 24, 0, 23))
                    self.heatmap_grid[grid_y, grid_x] += 1
                self.heatmap_frames += 1

            snapshot = None
            if self.heatmap_frames >= 100:
                mx = np.max(self.heatmap_grid)
                snapshot = (self.heatmap_grid / mx).tolist() if mx > 0 else self.heatmap_grid.tolist()
                self.heatmap_frames = 0

            return {"heatmap": snapshot, "total_detected": len(detections)}

        return {"total_detected": len(detections)}
