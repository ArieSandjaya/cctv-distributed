"""
Face recognition module using face_recognition (dlib).
Loads known faces from known_faces/ directory.

Image naming convention:
    known_faces/Nama Orang.jpg  → will recognize as "Nama Orang"

If no known faces are loaded, returns "Unknown" for all detections.
"""

import os
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")


class FaceRecognizer:
    def __init__(self, tolerance: float = 0.5):
        """
        tolerance: lower = stricter match (0.4-0.5 is typical).
        """
        self.tolerance = tolerance
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._loaded = False

    def load_known_faces(self, directory: str | None = None):
        """Scan a directory for face images and encode them."""
        directory = directory or KNOWN_FACES_DIR
        if not os.path.isdir(directory):
            logger.warning(f"Known faces directory not found: {directory}")
            return

        try:
            import face_recognition
        except ImportError:
            logger.error("face_recognition not installed. Face recognition disabled.")
            return

        encodings = []
        names = []

        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            name, _ = os.path.splitext(fname)

            try:
                image = face_recognition.load_image_file(fpath)
                face_encs = face_recognition.face_encodings(image)
                if not face_encs:
                    logger.warning(f"No face found in {fname}, skipping")
                    continue
                encodings.append(face_encs[0])
                names.append(name)
                logger.info(f"Loaded face: {name} ({len(face_encs[0])}d embedding)")
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")

        self._known_encodings = encodings
        self._known_names = names
        self._loaded = True
        logger.info(f"Loaded {len(names)} known faces: {names}")

    def recognize(self, face_crop_rgb: np.ndarray) -> str:
        """
        Given a face crop (RGB), return the name or "Unknown".
        """
        if not self._loaded or not self._known_encodings:
            return "Unknown"

        try:
            import face_recognition

            # Encode the face crop
            face_encs = face_recognition.face_encodings(face_crop_rgb)
            if not face_encs:
                return "Unknown"

            # Compare against known faces
            matches = face_recognition.compare_faces(
                self._known_encodings, face_encs[0], tolerance=self.tolerance
            )
            if not any(matches):
                return "Unknown"

            # Get the first match (can be enhanced with voting)
            idx = matches.index(True)
            return self._known_names[idx]

        except Exception as e:
            logger.warning(f"Face recognition error: {e}")
            return "Unknown"

    # ── High-level: process a full frame ────────────────────────

    def process_frame(self, frame_bgr: np.ndarray, person_bboxes: list[tuple]) -> dict:
        """
        Accept a BGR frame and a list of (x1,y1,x2,y2) bounding boxes for persons.
        Returns {name: count} for unique recognized people.
        """
        try:
            import face_recognition

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detected_names: dict[str, int] = {}

            for x1, y1, x2, y2 in person_bboxes:
                # Crop face area (upper portion of bounding box)
                face_crop = rgb[y1:y1 + (y2 - y1) // 2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Try to find a face within the crop
                face_locs = face_recognition.face_locations(face_crop)
                if not face_locs:
                    continue

                # Take the largest face in the crop
                top, right, bottom, left = max(face_locs, key=lambda f: (f[2]-f[0])*(f[1]-f[3]))
                face_img = face_crop[top:bottom, left:right]

                name = self.recognize(face_img)
                detected_names[name] = detected_names.get(name, 0) + 1

            return detected_names

        except Exception as e:
            logger.warning(f"process_frame error: {e}")
            return {}

    @property
    def known_names(self) -> list[str]:
        return self._known_names

    @property
    def is_active(self) -> bool:
        return self._loaded and len(self._known_encodings) > 0

    @property
    def face_count(self) -> int:
        return len(self._known_names)


# Lazy singleton (optional)
default_recognizer = FaceRecognizer()
