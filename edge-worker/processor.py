"""
People counting processor using Supervision + YOLO.
Feature-Based Implementation: 1 RTSP = 1 Feature.
Features: 'counting', 'face_rec', 'heatmap'
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

class PeopleCounter:
    def __init__(
        self,
        feature: str = "counting",
        model_path: str = "yolov8n.pt",
        face_recognition_enabled: bool = False,
    ):
        self.feature = feature
        logger.info(f"Initializing processor with feature: {feature}")
        
        self.model = YOLO(model_path)
        self.tracker = ByteTrack(minimum_matching_threshold=0.8)
        self.frame_size = (640, 480)
        
        # ROI for 'counting'
        self.polygon_zone: Optional[PolygonZone] = None
        self._roi_points = []

        # Heatmap for 'heatmap' (Grid 32x24)
        self.heatmap_grid = np.zeros((24, 32), dtype=np.float32)
        self.heatmap_update_count = 0

        # Face Recognition for 'face_rec'
        self._face_recognizer = None
        if feature == "face_rec":
            try:
                from face_rec import FaceRecognizer
                self._face_recognizer = FaceRecognizer()
                self._face_recognizer.load_known_faces()
                logger.info("Face recognition module loaded.")
            except Exception as e:
                logger.warning(f"Face rec load failed: {e}")

    def set_roi(self, points: list):
        self._roi_points = points
        polygon = Polygon([Point(x, y) for x, y in points])
        self.polygon_zone = PolygonZone(polygon=polygon, frame_resolution_wh=self.frame_size)
        logger.info(f"ROI set for {self.feature} mode")

    def set_frame_size(self, w, h):
        self.frame_size = (w, h)
        if self._roi_points:
            self.set_roi(self._roi_points)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        # Base detection for all features
        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[detections.class_id == PERSON_CLASS_ID]
        detections = self.tracker.update_with_detections(detections)

        # --- FEATURE: COUNTING ---
        if self.feature == "counting":
            count = 0
            if self.polygon_zone and len(detections) > 0:
                mask = self.polygon_zone.trigger(detections)
                count = len(detections[mask])
            return {"people_inside": count, "total_detected": len(detections)}

        # --- FEATURE: FACE REC ---
        elif self.feature == "face_rec":
            recognized = {}
            if self._face_recognizer and len(detections) > 0:
                bboxes = [tuple(map(int, detections.xyxy[i])) for i in range(len(detections))]
                recognized = self._face_recognizer.process_frame(frame, bboxes)
            return {"recognized": recognized, "total_detected": len(detections)}

        # --- FEATURE: HEATMAP ---
        elif self.feature == "heatmap":
            if len(detections) > 0:
                for bbox in detections.xyxy:
                    # Use bottom-center point (feet)
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, y2
                    
                    # Map to 32x24 grid
                    grid_x = int(np.clip(cx / self.frame_size[0] * 32, 0, 31))
                    grid_y = int(np.clip(cy / self.frame_size[1] * 24, 0, 23))
                    self.heatmap_grid[grid_y, grid_x] += 1
                
                self.heatmap_update_count += 1
            
            # Send heatmap snapshot every 100 frames to save bandwidth
            snapshot = None
            if self.heatmap_update_count >= 100:
                # Normalize grid 0-1
                max_val = np.max(self.heatmap_grid) if np.max(self.heatmap_grid) > 0 else 1
                snapshot = (self.heatmap_grid / max_val).tolist()
                self.heatmap_update_count = 0
            
            return {"heatmap": snapshot, "total_detected": len(detections)}

        return {"total_detected": len(detections)}
