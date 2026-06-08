"""
Edge Worker Multi-Stream: Handles multiple CCTV cameras concurrently
using threading. Each camera runs in its own thread.

Config: edge-worker/cameras.json
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
    format="%(asctime)s [%(threadName)s] %(message)s",
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
    """
    Thread untuk satu kamera.
    Loop: read frame → detect → publish ke MQTT.
    """

    def __init__(self, cam_id: str, cam_url: str, name: str = ""):
        super().__init__(name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self.cam_url = cam_url
        self.cam_name = name or cam_id
        self.daemon = True

        self.counter: PeopleCounter | None = None
        self.cap: cv2.VideoCapture | None = None
        self.config = {}  # remote config (line position, etc.)

    def run(self):
        global running, mqtt_client

        logger.info(f"[{self.cam_name}] Starting camera: {self.cam_url}")

        # Init processor
        self.counter = PeopleCounter(model_path=MODEL_PATH, face_recognition_enabled=FACE_ENABLED)
        self.cap = cv2.VideoCapture(self.cam_url)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"[{self.cam_name}] Cannot open stream: {self.cam_url}")
            return

        # Set default counting line (center vertical)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.counter.set_frame_size(w, h)
        self.counter.set_line(w // 2, 0, w // 2, h)

        logger.info(f"[{self.cam_name}] Started: {w}x{h}")
        active_cameras[self.cam_id] = {"thread": self, "status": "running"}

        while running:
            if not self.cap:
                break

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[{self.cam_name}] Stream lost, reconnecting in 3s...")
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
                logger.error(f"[{self.cam_name}] Processing error: {e}")

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
        """Update counting line or other settings (called from MQTT callback)."""
        self.config = config
        if self.counter and all(k in config for k in ("start_x", "start_y", "end_x", "end_y")):
            self.counter.set_line(
                config["start_x"], config["start_y"],
                config["end_x"], config["end_y"]
            )
            logger.info(f"[{self.cam_name}] Line updated")


# ── MQTT Callbacks ──────────────────────────────────────────────

def on_mqtt_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc})")
    # Subscribe to all config topics
    client.subscribe("config/#")
    logger.info("Subscribed to config/#")


def on_mqtt_message(client, userdata, msg):
    """Route config updates to the correct camera thread."""
    try:
        # topic format: config/{cam_id}
        topic_parts = msg.topic.split("/")
        if len(topic_parts) < 2:
            return
        cam_id = topic_parts[1]
        payload = json.loads(msg.payload.decode())

        if cam_id in active_cameras:
            thread_info = active_cameras[cam_id]
            thread_info["thread"].update_config(payload)
            logger.info(f"Config sent to {cam_id}: {payload}")
    except Exception as e:
        logger.warning(f"MQTT message error: {e}")


# ── Load Camera Config ──────────────────────────────────────────

def load_camera_config(path: str) -> list[dict]:
    """Load camera list from JSON file."""
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        logger.error("Please create cameras.json (see example below)")
        return []

    with open(path, "r") as f:
        config = json.load(f)

    cameras = config.get("cameras", [])
    logger.info(f"Loaded {len(cameras)} camera configs")
    return cameras


# ── Main ────────────────────────────────────────────────────────

def signal_handler(sig, frame):
    global running
    logger.info("Shutting down all cameras...")
    running = False


def main():
    global mqtt_client

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load camera config
    cameras = load_camera_config(CONFIG_PATH)
    if not cameras:
        logger.error("No cameras configured. Exiting.")
        sys.exit(1)

    # Connect MQTT
    mqtt_client = mqtt.Client(client_id=f"edge_{os.environ.get('HOSTNAME', 'unknown')}")
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        logger.info(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        logger.error(f"MQTT connection failed: {e}")
        logger.warning("Running without MQTT (local only)")

    # Start a thread for each camera
    threads = []
    for cam in cameras:
        cam_id = cam.get("cam_id", cam.get("id"))
        cam_url = cam.get("cam_url", cam.get("url"))
        cam_name = cam.get("name", cam_id)

        if not cam_id or not cam_url:
            logger.warning(f"Skipping invalid camera config: {cam}")
            continue

        thread = CameraThread(cam_id=cam_id, cam_url=cam_url, name=cam_name)
        thread.start()
        threads.append(thread)

    logger.info(f"Running {len(threads)} camera threads")

    # Keep main thread alive
    while running:
        time.sleep(1)

    # Cleanup
    for t in threads:
        t.join(timeout=2)

    if mqtt_client:
        mqtt_client.loop_stop()

    logger.info("Edge Worker stopped.")


if __name__ == "__main__":
    main()
