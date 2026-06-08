"""
Edge Worker Multi-Stream with Feature Selection.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("edge-worker")

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "cameras.json"))
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

running = True
active_cameras = {}
mqtt_client = None

class CameraThread(threading.Thread):
    def __init__(self, cam_id, cam_url, name="", feature="counting", roi=None):
        super().__init__(name=f"cam-{cam_id}")
        self.cam_id = cam_id
        self.cam_url = cam_url
        self.cam_name = name or cam_id
        self.feature = feature
        self.roi = roi
        self.daemon = True
        self.counter = None
        self.cap = None

    def run(self):
        global running, mqtt_client
        logger.info(f"[{self.cam_name}] Starting with feature: {self.feature}")
        
        self.counter = PeopleCounter(feature=self.feature)
        self.cap = cv2.VideoCapture(self.cam_url)
        if not self.cap.isOpened(): return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.counter.set_frame_size(w, h)
        if self.roi: self.counter.set_roi(self.roi)

        active_cameras[self.cam_id] = {"thread": self, "status": "running"}

        while running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.cam_url)
                continue

            analytics = self.counter.process_frame(frame)
            payload = {"cam_id": self.cam_id, "feature": self.feature, "timestamp": time.time(), **analytics}
            
            if mqtt_client and mqtt_client.is_connected():
                mqtt_client.publish(f"analytics/{self.cam_id}", json.dumps(payload))
            
            time.sleep(0.2)

        self.cap.release()

    def update_config(self, config):
        # Change Feature? Restart Processor
        new_feature = config.get("feature")
        if new_feature and new_feature != self.feature:
            logger.info(f"[{self.cam_name}] Changing feature {self.feature} -> {new_feature}")
            self.feature = new_feature
            self.counter = PeopleCounter(feature=self.feature)
            # Restore frame size and ROI
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            self.counter.set_frame_size(w, h)
            if self.roi: self.counter.set_roi(self.roi)

        roi = config.get("roi")
        if roi:
            self.roi = roi
            if self.counter: self.counter.set_roi(roi)

def on_mqtt_message(client, userdata, msg):
    try:
        cam_id = msg.topic.split("/")[1]
        payload = json.loads(msg.payload.decode())
        if cam_id in active_cameras:
            active_cameras[cam_id]["thread"].update_config(payload)
    except: pass

def main():
    global mqtt_client
    cameras = json.load(open(CONFIG_PATH))["cameras"]
    
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
    mqtt_client.subscribe("config/#")
    mqtt_client.loop_start()

    threads = []
    for cam in cameras:
        t = CameraThread(cam["cam_id"], cam["cam_url"], cam.get("name"), cam.get("feature", "counting"), cam.get("roi"))
        t.start()
        threads.append(t)

    while True: time.sleep(1)

if __name__ == "__main__":
    main()
