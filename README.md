# CCTV Distributed System

Edge Workers (GPU servers) → MQTT (Mosquitto) → Central Server (FastAPI + Dashboard)

## Quick Start

```bash
docker compose up -d
```

Open http://localhost:8000

## Edge Workers

On each GPU server:

```bash
docker run -d \
  -e CAM_ID=cam_01 \
  -e CAM_URL=rtsp://192.168.1.100:554/stream1 \
  -e MQTT_BROKER=<central-ip> \
  -e MQTT_PORT=1883 \
  edge-worker
```

## Structure

```
central-server/    # The Brain: FastAPI + SQLite + Dashboard
edge-worker/       # The Muscle: YOLO + Supervision + Face Rec
docker-compose.yml # Full stack (Mosquitto + Central + Workers)
```
