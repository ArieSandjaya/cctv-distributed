# CCTV Distributed System

Sistem CCTV **People Counting + Face Recognition** untuk skala enterprise.
Arsitektur **Hub-and-Spoke**: 1 Central Server 🧠 memantau banyak Edge Server 💪.

## 🏗️ Arsitektur

```
┌───────────────────────────────────────────────┐
│          CENTRAL SERVER (1 server)             │
│  FastAPI + SQLite + Dashboard + MQTT Broker    │
└──────────┬────────────────────────────────────┘
           │ MQTT (data JSON kecil & ringan)
      ┌────┼────┬────┬────┐
      ▼    ▼    ▼    ▼    ▼
┌──────────────────────────────────┐
│        EDGE SERVER #1            │
│  Worker (Multi-Stream)           │
│  ├─ cam_01 (lantai1_pintu)      │
│  ├─ cam_02 (lantai1_lorong)     │
│  ├─ cam_03 ...                   │
│  └─ cam_10                       │
│  YOLOv8 + Supervision + FaceRec  │
│  GPU NVIDIA (wajib)              │
└──────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Central Server

```bash
cd central-server
pip install -r requirements.txt

# Install MQTT broker (Mosquitto)
# Windows: https://mosquitto.org/download/
# Linux: sudo apt install mosquitto mosquitto-clients

python main.py
# → Buka http://localhost:8000
```

### 2. Edge Worker (Bare Metal — rekomendasi)

Di tiap server GPU, clone repo ini lalu:

```bash
cd edge-worker
pip install -r requirements.txt

# Edit file cameras.json — isi daftar kamera lo
nano cameras.json

# Jalankan
python worker.py
```

### 3. Edge Worker (Docker — alternatif)

```bash
docker build -t cctv-edge ./edge-worker

docker run -d \
  --name edge_worker \
  --gpus all \
  -e MQTT_BROKER="<IP_CENTRAL>" \
  -v $(pwd)/cameras.json:/app/cameras.json \
  -v $(pwd)/known_faces:/app/known_faces \
  cctv-edge
```

---

## 📁 Struktur Project

```
cctv-distributed/
├── central-server/
│   ├── main.py              # FastAPI + SQLite + Dashboard HTML
│   ├── requirements.txt
│   └── Dockerfile
├── edge-worker/
│   ├── worker.py             # Multi-Stream (handle 10+ kamera)
│   ├── processor.py          # YOLO + Supervision logic
│   ├── face_rec.py           # Face recognition module
│   ├── cameras.json          # Daftar kamera (edit sesuai kebutuhan)
│   ├── known_faces/          # Foto wajah (.jpg, .png)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml        # Docker full stack (opsional)
├── mosquitto.conf
└── README.md
```

---

## ⚙️ Konfigurasi Kamera (`cameras.json`)

Edit file `cameras.json` untuk menambahkan kamera. Contoh:

```json
{
  "cameras": [
    {
      "cam_id": "lantai1_cam01",
      "name": "Lantai 1 - Pintu Utama",
      "cam_url": "rtsp://admin:pass@192.168.1.101:554/stream1"
    },
    {
      "cam_id": "lantai1_cam02",
      "name": "Lantai 1 - Lorong Timur",
      "cam_url": "rtsp://admin:pass@192.168.1.102:554/stream1"
    }
  ]
}
```

---

## 🔌 API Endpoints (Central Server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard HTML |
| GET | `/health` | Status server + jumlah kamera |
| GET | `/api/cameras` | Semua kamera & status online/offline |
| GET | `/api/cameras/{cam_id}` | Detail + history kamera |
| POST | `/api/config/{cam_id}` | Update konfigurasi remote |
| GET | `/api/summary` | Total agregat seluruh kamera |
| WS | `/ws` | WebSocket real-time |

---

## 🎯 Fitur

- **Multi-Camera**: 1 server handle 10+ kamera sekaligus (threading)
- **People Counting**: IN/OUT/INSIDE per kamera + total agregat
- **Face Recognition**: Otomatis kenali wajah yang sudah di-enroll
- **Remote Config**: Ubah garis counting dari dashboard pusat
- **Offline Detection**: Tahu kamera mana yang mati/putus
- **No Docker Required**: Berjalan langsung di sistem operasi

## ⚠️ Syarat Hardware

### Edge Server (wajib GPU NVIDIA)
- **OS**: Linux (Ubuntu 22.04 rekomendasi) atau Windows
- **GPU**: NVIDIA dengan minimal 6GB VRAM (GTX 1660 / RTX 3060 ke atas)
- **RAM**: Minimal 8GB (16GB rekomendasi)
- **Storage**: 50GB free space

### Central Server (sangat ringan)
- **CPU**: 2 core saja sudah cukup
- **RAM**: 2-4GB
- **OS**: Linux / Windows / Mac
