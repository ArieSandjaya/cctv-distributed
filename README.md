# CCTV Distributed System ⚡

Sistem CCTV **People Counting + Face Recognition + Heatmap + Trajectory** untuk skala enterprise.
Arsitektur **Hub-and-Spoke**: 1 Central Server 🧠 memantau banyak Edge Server 💪.

**Live Dashboard**: [CCTOps](http://localhost:8000) — WebSocket real-time, grid cards, canvas heatmap.

---

## 🏗️ Arsitektur

```
┌──────────────────────────────────────────────────────┐
│              CENTRAL SERVER (1 server)                │
│  FastAPI + SQLite + CCTOps Dashboard + MQTT Broker    │
└──────────────────────┬───────────────────────────────┘
                       │ MQTT (data JSON kecil & ringan)
             ┌────┬────┼────┬────┬────┐
             ▼    ▼    ▼    ▼    ▼    ▼
┌──────────────────────────────────────────────┐
│              EDGE SERVER #N                   │
│  Worker (Multi-Stream)                        │
│  ┌─ cam_01 (lantai1_pintu) — counting        │
│  ├─ cam_02 (lantai1_lorong) — face_rec       │
│  ├─ cam_03 (lantai2_pintu) — heatmap         │
│  ├─ cam_04 (lantai2_lobby) — trajectory      │
│  └─ cam_05 ...                                │
│  YOLOv8 + Supervision + FaceRec               │
│  GPU NVIDIA (wajib)                           │
└──────────────────────────────────────────────┘
```

Setiap kamera di Edge Server menjalankan **1 fitur spesifik** (counting / face_rec / heatmap / trajectory).
1 Edge Server bisa handle 10+ kamera sekaligus via threading.

---

## 🎯 Fitur

| Fitur | Deskripsi | Output |
|-------|-----------|--------|
| **Counting** 🧮 | Hitung orang IN/OUT per zona polygon | people_inside, people_tracked |
| **Face Recognition** 🧑‍🤝‍🧑 | Kenali wajah yang sudah di-enroll | Nama + jumlah kemunculan |
| **Heatmap** 🔥 | Peta aktivitas berbasis grid | Matrix 2D (densitas per sel) |
| **Trajectory** 🧭 | Lacak pergerakan antar zona (OD Matrix) | Origin-Destination matrix |

---

## 🚀 Quick Start

### 1. Central Server

```bash
cd central-server
pip install -r requirements.txt
python main.py
# → Buka http://localhost:8000
```

> MQTT broker sudah terintegrasi di `main.py` (built-in), tidak perlu install Mosquitto terpisah.

### 2. Edge Worker (Bare Metal)

Di tiap server GPU, clone repo ini lalu:

```bash
cd edge-worker
pip install -r requirements.txt

# Edit cameras.json — isi daftar kamera + fitur
nano cameras.json

# Set IP Central Server — WAJIB!
export MQTT_BROKER="192.168.1.100"  # ganti dengan IP central

# Jalankan
python worker.py
```

> **⚠️ Wajib set `MQTT_BROKER`** — default-nya `localhost`, jadi kalo gak diset, edge worker bakal nyari MQTT broker di mesin sendiri. Bisa via env var atau export di shell.

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

### Full Stack (Docker Compose)

```bash
docker-compose up -d
```

---

## ⚙️ Konfigurasi Kamera (`cameras.json`)

```json
{
  "cameras": [
    {
      "cam_id": "lantai1_cam01",
      "name": "Lantai 1 - Pintu Utama",
      "cam_url": "rtsp://admin:pass@192.168.1.101:554/stream1",
      "feature": "counting",
      "roi": [[100,200], [500,200], [500,600], [100,600]],
      "zones": {
        "zona_a": [[50,50], [200,50], [200,300], [50,300]],
        "zona_b": [[250,50], [400,50], [400,300], [250,300]]
      }
    },
    {
      "cam_id": "lantai1_cam02",
      "name": "Lantai 1 - Lorong Timur",
      "cam_url": "rtsp://admin:pass@192.168.1.102:554/stream1",
      "feature": "face_rec"
    },
    {
      "cam_id": "lantai2_cam03",
      "name": "Lantai 2 - Lobby",
      "cam_url": "rtsp://admin:pass@192.168.1.103:554/stream1",
      "feature": "heatmap"
    },
    {
      "cam_id": "lantai2_cam04",
      "name": "Lantai 2 - Koridor",
      "cam_url": "rtsp://admin:pass@192.168.1.104:554/stream1",
      "feature": "trajectory",
      "zones": {
        "pintu_masuk": [[50,50], [200,50], [200,300], [50,300]],
        "tangga": [[250,50], [400,50], [400,300], [250,300]],
        "lift": [[450,50], [600,50], [600,300], [450,300]]
      }
    }
  ]
}
```

### Penjelasan Field

| Field | Wajib | Keterangan |
|-------|-------|------------|
| `cam_id` | ✅ | ID unik kamera |
| `name` | ✅ | Nama tampilan |
| `cam_url` | ✅ | URL RTSP / IP camera |
| `feature` | ✅ | `counting` / `face_rec` / `heatmap` / `trajectory` |
| `roi` | (counting) | Polygon Region of Interest `[[x,y], ...]` |
| `zones` | (trajectory) | Zona-zona untuk OD Matrix `{"nama": [[x,y], ...]}` |

---

## 🖥️ Dashboard — CCTOps

| Sebelum | Sesudah |
|---------|---------|
| List vertikal | **Grid cards** |
| Heatmap cuma teks "Active" | **Canvas visual** (biru→merah) |
| Semua info di card | Klik card → **Modal detail** |
| — | Tabel **OD Matrix** di modal |
| — | **WebSocket real-time** |

### Tampilan:

- **Card per kamera**: nama, status dot (hijau/merah), feature tag, extra info spesifik fitur
- **Stat bar**: total kamera, online count, total people inside
- **Heatmap**: render pixelated canvas dengan color scale (biru → hijau → kuning → merah)
- **Trajectory**: klik card → modal menampilkan tabel Origin-Destination Matrix
- **Face Recognition**: daftar wajah terdeteksi + jumlah kemunculan

---

## 📁 Struktur Project

```
cctv-distributed/
├── central-server/
│   ├── main.py              # FastAPI + built-in MQTT broker + CCTOps Dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── edge-worker/
│   ├── worker.py             # Multi-Stream (handle 10+ kamera)
│   ├── processor.py          # YOLO + Supervision + per-feature logic
│   ├── face_rec.py           # Face recognition module
│   ├── cameras.json          # Daftar kamera (edit sesuai kebutuhan)
│   ├── known_faces/          # Foto wajah yang dikenali (.jpg, .png)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml        # Full stack (opsional)
├── mosquitto.conf            # (optional — MQTT sudah built-in di main.py)
└── README.md
```

---

## 🔌 API Endpoints (Central Server)

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Dashboard CCTOps |
| GET | `/health` | Status server + jumlah kamera |
| GET | `/api/cameras` | Semua kamera & status online/offline |
| GET | `/api/cameras/{cam_id}` | Detail + 50 history terakhir |
| GET | `/api/summary` | Total agregat seluruh kamera |
| POST | `/api/config/{cam_id}` | Update konfigurasi remote (feature, roi, zones) |
| WS | `/ws` | WebSocket real-time (init + incremental update) |

---

## 🧠 Edge Worker — Detail

### Feature-Based Architecture

Setiap kamera di `cameras.json` punya **1 feature**. Worker membaca field `feature` dan menjalankan pipeline yang sesuai:

| Feature | Pipeline |
|---------|----------|
| `counting` | YOLO → PolygonZone → IN/OUT counter |
| `face_rec` | YOLO → crop face → FaceNet → match known |
| `heatmap` | YOLO → grid accumulator (32×24) |
| `trajectory` | YOLO → zone detection → OD matrix update |

Semua data dikirim ke Central Server via MQTT dalam format JSON.

### PolygonZone (sejak v3)

ROI tidak lagi menggunakan LineZone (garis), tapi **PolygonZone** — area poligon tertutup. Cocok untuk:
- Area persegi / trapesium
- Zona parkir / antrian
- Zona larangan masuk

### Trajectory OD Matrix

Edge worker melacak pergerakan antar zona. Output:
```json
{
  "od_matrix": {
    "pintu_masuk": {"pintu_masuk": 0, "tangga": 5, "lift": 3},
    "tangga": {"pintu_masuk": 2, "tangga": 0, "lift": 8},
    "lift": {"pintu_masuk": 1, "tangga": 6, "lift": 0}
  }
}
```

---

## ⚠️ Syarat Hardware

### Edge Server (wajib GPU NVIDIA)
| Komponen | Minimal | Rekomendasi |
|----------|---------|-------------|
| **GPU** | 6GB VRAM (GTX 1660) | 8GB+ (RTX 3060 / RTX 4070) |
| **RAM** | 8GB | 16GB |
| **Storage** | 50GB | 100GB+ |
| **OS** | Linux (Ubuntu 22.04) | Linux (Ubuntu 22.04) |

### Central Server (sangat ringan)
| Komponen | Spesifikasi |
|----------|-------------|
| **CPU** | 2 core |
| **RAM** | 2-4GB |
| **OS** | Linux / Windows / Mac |

---

## 📦 Dependencies

### Central Server
- FastAPI + Uvicorn
- paho-mqtt
- Python 3.9+

### Edge Worker
- Python 3.9+
- PyTorch + torchvision
- ultralytics (YOLOv8)
- supervision
- opencv-python
- numpy
- paho-mqtt
- face-recognition (dlib)
- CUDA Toolkit (sesuai versi PyTorch)
