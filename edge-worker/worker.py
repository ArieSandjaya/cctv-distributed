"""
Edge Worker Multi-Stream: Handles multiple CCTV cameras concurrently
with PolygonZone (ROI) people counting. Each camera runs in its own thread.
"""

import os
import sys
import json
import cv2
import time
import logging
import threading
import signal
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).parent))
from processor import PeopleCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("edge-worker")

# ── Config ──────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "cameras.json"))
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
FACE_ENABLED = os.environ.get("FACE_ENABLED", "true").lower() == "true"
ANALYTICS_INTERVAL = 0.5  # seconds per camera report

# ── Globals ─────────────────────────────────────────────────────
running = True
active_cameras: dict[str, dict] = {}
mqtt_client: mqtt.Client | None = None


# ── Camera Thread ───────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self, cam_id: str, cam_url: str, name: str = "", roi: list | None = None):
        super().__init__(name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self.cam_url = cam_url
        self.cam_name = name or cam_id
        self.default_roi = roi
        self.daemon = True

        self.counter: PeopleCounter | None = None
        self.cap: cv2.VideoCapture | None = None
        self.config = {}

    def run(self):
        global running, mqtt_client

        logger.info(f"[{self.cam_name}] Starting...")

        self.counter = PeopleCounter(model_path=MODEL_PATH, face_recognition_enabled=FACE_ENABLED)
        self.cap = cv2.VideoCapture(self.cam_url)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"[{self.cam_name}] Cannot open stream")
            return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.counter.set_frame_size(w, h)

        # Apply default ROI from cameras.json
        if self.default_roi:
            self.counter.set_roi(self.default_roi)
        else:
            # Fallback: full frame as polygon
            self.counter.set_roi([[0, 0], [w, 0], [w, h], [0, h]])

        self.counter._face_enabled = FACE_ENABLED
        logger.info(f"[{self.cam_name}] Started: {w}x{h}")

        active_cameras[self.cam_id] = {"thread": self, "status": "running"}

        while running:
            if not self.cap:
                break

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[{self.cam_name}] Stream lost, reconnecting...")
                time.sleep(3)
                self._reconnect()
                continue

            try:
                analytics = self.counter.process_frame(frame)

                payload = {
                    "cam_id": self.cam_id,
                    "cam_name": self.cam_name,
                    "timestamp": time.time(),
                    **analytics,
                }

                if mqtt_client and mqtt_client.is_connected():
                    mqtt_client.publish(
                        f"analytics/{self.cam_id}",
                        json.dumps(payload),
                        qos=1,
                    )
            except Exception as e:
                logger.error(f"[{self.cam_name}] Error: {e}")

            time.sleep(ANALYTICS_INTERVAL)

        self._cleanup()

    def _reconnect(self):
        if self.cap:
            self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.cam_url)

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        active_cameras.pop(self.cam_id, None)
        logger.info(f"[{self.cam_name}] Stopped")

    def update_config(self, config: dict):
        """Update ROI or other settings (called from MQTT callback)."""
        self.config = config
        roi = config.get("roi") or config.get("roi_points")
        if roi and self.counter:
            self.counter.set_roi(roi)
            logger.info(f"[{self.cam_name}] ROI updated: {len(roi)} points")
        else:
            logger.warning(f"[{self.cam_name}] Config received but no ROI: {config}")


# ── MQTT Callbacks ──────────────────────────────────────────────

def on_mqtt_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc})")
    client.subscribe("config/#")
    logger.info("Subscribed to config/#")


def on_mqtt_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split("/")
        if len(topic_parts) < 2:
            return
        cam_id = topic_parts[1]
        payload = json.loads(msg.payload.decode())

        if cam_id in active_cameras:
            active_cameras[cam_id]["thread"].update_config(payload)
            logger.info(f"Config sent to {cam_id}: {payload}")
    except Exception as e:
        logger.warning(f"MQTT error: {e}")


# ── Load Config ─────────────────────────────────────────────────

def load_camera_config(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        return []

    with open(path, "r") as f:
        config = json.load(f)

    cameras = config.get("cameras", [])
    logger.info(f"Loaded {len(cameras)} cameras")
    return cameras


# ── Main ────────────────────────────────────────────────────────

def signal_handler(sig, frame):
    global running
    logger.info("Shutting down...")
    running = False


def main():
    global mqtt_client

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cameras = load_camera_config(CONFIG_PATH)
    if not cameras:
        sys.exit(1)

    # MQTT
    mqtt_client = mqtt.Client(client_id=f"edge_{os.environ.get('HOSTNAME', 'edge')}")
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        logger.warning(f"MQTT failed: {e}")

    # Start threads
    threads = []
    for cam in cameras:
        cid = cam.get("cam_id") or cam.get("id")
        url = cam.get("cam_url") or cam.get("url")
        name = cam.get("name", cid)
        roi = cam.get("roi") or cam.get("roi_points")

        if not cid or not url:
            continue

        t = CameraThread(cam_id=cid, cam_url=url, name=name, roi=roi)
        t.start()
        threads.append(t)

    logger.info(f"Running {len(threads)} camera threads")

    while running:
        time.sleep(1)

    if mqtt_client:
        mqtt_client.loop_stop()

    logger.info("Edge Worker stopped.")


if __name__ == "__main__":
    main()
