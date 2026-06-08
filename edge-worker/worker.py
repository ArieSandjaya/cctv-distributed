"""
Edge Worker: Processes CCTV stream, detects people + faces,
and publishes analytics to Central Server via MQTT.
"""

import os
import sys
import json
import cv2
import time
import logging
import signal
import threading
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).parent))
from processor import PeopleCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EDGE] %(message)s",
)
logger = logging.getLogger("edge-worker")

# ── Config from env ────────────────────────────────────────────
CAM_ID = os.environ.get("CAM_ID", "cam_01")
CAM_URL = os.environ.get("CAM_URL", "people-walking.mp4")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
FACE_ENABLED = os.environ.get("FACE_ENABLED", "true").lower() == "true"
ANALYTICS_INTERVAL = 0.5  # seconds between reports

# ── Globals ─────────────────────────────────────────────────────
counter: PeopleCounter | None = None
cap: cv2.VideoCapture | None = None
mqtt_client: mqtt.Client | None = None
running = True
current_config = {}


# ── MQTT Callbacks ──────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected to MQTT broker (rc={rc})")
    # Subscribe to config for this camera
    client.subscribe(f"config/{CAM_ID}")
    logger.info(f"Subscribed to config/{CAM_ID}")


def on_message(client, userdata, msg):
    """Handle config updates from Central Server."""
    global current_config
    try:
        payload = json.loads(msg.payload.decode())
        logger.info(f"Received config update: {payload}")

        # Update counting line if provided
        if counter and all(k in payload for k in ("start_x", "start_y", "end_x", "end_y")):
            counter.set_line(
                payload["start_x"], payload["start_y"],
                payload["end_x"], payload["end_y"]
            )
            logger.info(f"Line updated to: ({payload['start_x']},{payload['start_y']}) -> ({payload['end_x']},{payload['end_y']})")

        current_config = payload
    except Exception as e:
        logger.warning(f"Config parse error: {e}")


# ── Main Loop ──────────────────────────────────────────────────

def publish_analytics():
    """Main processing loop: read frame → detect → publish."""
    global counter, cap, running

    counter = PeopleCounter(model_path=MODEL_PATH, face_recognition_enabled=FACE_ENABLED)
    cap = cv2.VideoCapture(CAM_URL)

    if not cap.isOpened():
        logger.error(f"Cannot open camera: {CAM_URL}")
        return

    # Set default line (center vertical)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    counter.set_frame_size(w, h)
    counter.set_line(w // 2, 0, w // 2, h)
    logger.info(f"Started: {CAM_ID} @ {CAM_URL} ({w}x{h})")

    while running:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Stream lost, reconnecting in 3s...")
            time.sleep(3)
            cap.release()
            cap = cv2.VideoCapture(CAM_URL)
            continue

        try:
            analytics = counter.process_frame(frame)

            # Add camera metadata
            payload = {
                "cam_id": CAM_ID,
                "timestamp": time.time(),
                **analytics,
            }

            # Publish to MQTT
            if mqtt_client and mqtt_client.is_connected():
                mqtt_client.publish(
                    f"analytics/{CAM_ID}",
                    json.dumps(payload),
                    qos=1,
                )

        except Exception as e:
            logger.error(f"Processing error: {e}")

        time.sleep(ANALYTICS_INTERVAL)

    cap.release()


# ── Signal Handling ─────────────────────────────────────────────

def signal_handler(sig, frame):
    global running
    logger.info("Shutting down...")
    running = False
    if mqtt_client:
        mqtt_client.disconnect()


# ── Entry Point ─────────────────────────────────────────────────

def main():
    global mqtt_client

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Connect MQTT
    mqtt_client = mqtt.Client(client_id=f"edge_{CAM_ID}")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        logger.error(f"MQTT connection failed: {e}")
        logger.warning("Running without MQTT (local only)")

    # Run processing (blocking)
    publish_analytics()

    if mqtt_client:
        mqtt_client.loop_stop()


if __name__ == "__main__":
    main()
