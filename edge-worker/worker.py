"""
Edge Worker — Multi-Stream dengan Feature Selection.
Support: counting, face_rec, heatmap, trajectory
"""

import os
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("edge-worker")

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "cameras.json"))
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

running = True
active_cameras = {}
mqtt_client = None


class CameraThread(threading.Thread):
    def __init__(self, cam_id, cam_url, name="", feature="counting", roi=None, zones=None):
        super().__init__(name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self.cam_url = cam_url
        self.cam_name = name or cam_id
        self.feature = feature
        self.roi = roi
        self.zones = zones
        self.daemon = True
        self.counter = None
        self.cap = None

    def run(self):
        global running, mqtt_client
        logger.info(f"[{self.cam_name}] Starting — feature: {self.feature}")

        self.counter = PeopleCounter(feature=self.feature)
        self.cap = cv2.VideoCapture(self.cam_url)
        if not self.cap.isOpened():
            logger.error(f"[{self.cam_name}] Failed to open stream")
            return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.counter.set_frame_size(w, h)

        if self.feature == "trajectory" and self.zones:
            self.counter.set_trajectory_zones(self.zones)
        elif self.roi:
            self.counter.set_roi(self.roi)

        active_cameras[self.cam_id] = {"thread": self, "status": "running", "feature": self.feature}

        while running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[{self.cam_name}] Stream lost, reconnecting...")
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.cam_url)
                continue

            try:
                analytics = self.counter.process_frame(frame)
                payload = {
                    "cam_id": self.cam_id,
                    "feature": self.feature,
                    "timestamp": time.time(),
                    **analytics,
                }
                if mqtt_client and mqtt_client.is_connected():
                    mqtt_client.publish(f"analytics/{self.cam_id}", json.dumps(payload), qos=1)
            except Exception as e:
                logger.error(f"[{self.cam_name}] Error: {e}")

            time.sleep(0.2)

        self.cap.release()
        active_cameras.pop(self.cam_id, None)

    def update_config(self, config: dict):
        new_feature = config.get("feature")
        if new_feature and new_feature != self.feature:
            logger.info(f"[{self.cam_name}] Feature change: {self.feature} -> {new_feature}")
            self.feature = new_feature
            self.counter = PeopleCounter(feature=self.feature)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            self.counter.set_frame_size(w, h)

        zones = config.get("zones")
        if zones and self.feature == "trajectory":
            self.zones = zones
            self.counter.set_trajectory_zones(zones)

        roi = config.get("roi")
        if roi and self.feature == "counting":
            self.roi = roi
            self.counter.set_roi(roi)

        active_cameras[self.cam_id] = {"thread": self, "status": "running", "feature": self.feature}


def on_mqtt_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc})")
    client.subscribe("config/#")


def on_mqtt_message(client, userdata, msg):
    try:
        cam_id = msg.topic.split("/")[1]
        payload = json.loads(msg.payload.decode())
        if cam_id in active_cameras:
            logger.info(f"Config received for {cam_id}: {list(payload.keys())}")
            active_cameras[cam_id]["thread"].update_config(payload)
    except Exception as e:
        logger.warning(f"MQTT error: {e}")


def load_camera_config(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.error(f"Config not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)["cameras"]


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
        return

    mqtt_client = mqtt.Client(client_id=f"edge_{os.environ.get('HOSTNAME', 'edge')}")
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        logger.warning(f"MQTT offline: {e}")

    threads = []
    for cam in cameras:
        t = CameraThread(
            cam_id=cam["cam_id"],
            cam_url=cam["cam_url"],
            name=cam.get("name"),
            feature=cam.get("feature", "counting"),
            roi=cam.get("roi"),
            zones=cam.get("zones"),
        )
        t.start()
        threads.append(t)

    logger.info(f"Running {len(threads)} cameras")
    while running:
        time.sleep(1)

    if mqtt_client:
        mqtt_client.loop_stop()

    logger.info("Stopped.")


if __name__ == "__main__":
    main()
