"""
Central Server (The Brain).
- Runs MQTT broker (Mosquitto)
- Receives analytics from all Edge Workers
- Stores data in SQLite
- Serves a unified dashboard via FastAPI
- Sends config commands to Edge Workers via MQTT
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Simple SQLite store (no ORM needed for this scale)
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CENTRAL] %(message)s",
)
logger = logging.getLogger("central-server")

# ── Config ──────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "/data/central.db")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "0.0.0.0")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# ── Database Setup ──────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            count_in INTEGER DEFAULT 0,
            count_out INTEGER DEFAULT 0,
            total_inside INTEGER DEFAULT 0,
            people_now INTEGER DEFAULT 0,
            recognized TEXT DEFAULT '{}',  -- JSON dict
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS camera_status (
            cam_id TEXT PRIMARY KEY,
            last_seen REAL,
            last_analytics TEXT,
            config TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    return conn

db = init_db()

# ── In-memory cache (for fast WebSocket pushes) ─────────────────
camera_cache: dict[str, dict] = {}
ws_clients: list[WebSocket] = []


# ── MQTT Client ─────────────────────────────────────────────────

def on_connect_central(client, userdata, flags, rc):
    logger.info(f"Connected to local MQTT broker (rc={rc})")
    client.subscribe("analytics/#")
    logger.info("Subscribed to analytics/#")


def on_message_central(client, userdata, msg):
    """Store incoming analytics from Edge Workers."""
    try:
        payload = json.loads(msg.payload.decode())
        cam_id = payload.get("cam_id", "unknown")
        timestamp = payload.get("timestamp", time.time())

        recognized = payload.get("recognized", {})

        # Save to DB
        db.execute(
            """INSERT INTO analytics (cam_id, timestamp, count_in, count_out,
               total_inside, people_now, recognized)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (cam_id, timestamp,
             payload.get("count_in", 0),
             payload.get("count_out", 0),
             payload.get("total_inside", 0),
             payload.get("people_now", 0),
             json.dumps(recognized))
        )
        db.commit()

        # Update cache
        camera_cache[cam_id] = payload

        # Update status
        db.execute(
            """INSERT OR REPLACE INTO camera_status (cam_id, last_seen, last_analytics)
               VALUES (?, ?, ?)""",
            (cam_id, timestamp, json.dumps(payload))
        )
        db.commit()

        # Push to all connected WebSocket clients
        push_to_ws({"type": "update", "cam_id": cam_id, "data": payload})

    except Exception as e:
        logger.warning(f"MQTT message error: {e}")


def push_to_ws(data):
    """Broadcast to all connected dashboard clients."""
    msg = json.dumps(data)
    for ws in list(ws_clients):
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)
        except Exception:
            try:
                ws_clients.remove(ws)
            except ValueError:
                pass


# Store asyncio loop reference for thread safety
loop = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start MQTT client on boot."""
    global loop
    loop = asyncio.get_event_loop()

    # Start MQTT in background thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()
    logger.info("Central Server started")

    yield

    logger.info("Shutdown complete.")


def start_mqtt():
    client = mqtt.Client(client_id="central_server")
    client.on_connect = on_connect_central
    client.on_message = on_message_central

    # Retry connection until Mosquitto is ready
    for attempt in range(30):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            logger.warning(f"Waiting for MQTT broker... ({e})")
            time.sleep(2)
    else:
        logger.error("Could not connect to MQTT broker")
        return

    client.loop_forever()


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(title="CCTV Central Control", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cameras_online": len(camera_cache),
        "db_path": DB_PATH,
    }


@app.get("/api/cameras")
async def list_cameras():
    """Return all cameras and their latest analytics."""
    # Get camera list from status table
    rows = db.execute(
        "SELECT cam_id, last_seen, last_analytics, config FROM camera_status ORDER BY cam_id"
    ).fetchall()

    cameras = []
    for row in rows:
        cam_id, last_seen, last_analytics, config = row
        analytics = json.loads(last_analytics) if last_analytics else {}
        is_online = (time.time() - last_seen) < 30 if last_seen else False
        cameras.append({
            "cam_id": cam_id,
            "online": is_online,
            "last_seen": last_seen,
            "analytics": analytics,
            "config": json.loads(config) if config else {},
        })

    return {"cameras": cameras, "total": len(cameras), "online": sum(1 for c in cameras if c["online"])}


@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    """Get detailed info for a specific camera."""
    data = camera_cache.get(cam_id, {})
    # Get recent history
    rows = db.execute(
        "SELECT * FROM analytics WHERE cam_id = ? ORDER BY id DESC LIMIT 50",
        (cam_id,)
    ).fetchall()
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[2],
            "count_in": row[3],
            "count_out": row[4],
            "total_inside": row[5],
            "people_now": row[6],
            "recognized": json.loads(row[7]) if row[7] else {},
            "created_at": row[8],
        })
    return {"cam_id": cam_id, "current": data, "history": history}


@app.post("/api/config/{cam_id}")
async def configure_camera(cam_id: str, config: dict):
    """Send config changes to an Edge Worker via MQTT."""
    success = mqtt_client.publish(
        f"config/{cam_id}",
        json.dumps(config),
        qos=1,
    )
    # Save config to DB
    db.execute(
        "UPDATE camera_status SET config = ? WHERE cam_id = ?",
        (json.dumps(config), cam_id)
    )
    db.commit()

    return {
        "status": "sent",
        "cam_id": cam_id,
        "config": config,
        "mqtt_rc": success.rc,
    }


@app.get("/api/summary")
async def get_summary():
    """Aggregate all cameras into a single summary."""
    total_in = 0
    total_out = 0
    total_inside = 0
    online_count = 0

    for cam_id, data in camera_cache.items():
        total_in += data.get("count_in", 0)
        total_out += data.get("count_out", 0)
        total_inside += data.get("total_inside", 0)
        online_count += 1

    return {
        "cameras_online": online_count,
        "total_count_in": total_in,
        "total_count_out": total_out,
        "total_people_inside": total_inside,
        "timestamp": time.time(),
    }


# ── WebSocket (for real-time dashboard) ─────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    logger.info(f"Dashboard client connected (total: {len(ws_clients)})")

    # Send initial snapshot of all cameras
    await ws.send_json({"type": "init", "cameras": camera_cache})

    try:
        while True:
            # Wait for any incoming message (keepalive)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass
        logger.info(f"Dashboard client disconnected (total: {len(ws_clients)})")


# ── Simple Dashboard HTML ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Central Control</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', sans-serif;
                background: #0f172a; color: #e2e8f0; padding: 24px;
            }
            h1 { margin-bottom: 8px; }
            .subtitle { color: #64748b; margin-bottom: 24px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
            .card {
                background: #1e293b; border-radius: 12px; padding: 16px; text-align: center;
                border-left: 4px solid #3b82f6;
            }
            .card .val { font-size: 32px; font-weight: 700; margin-top: 4px; }
            .card .lbl { font-size: 13px; color: #94a3b8; }
            .cameras { display: grid; gap: 12px; }
            .cam-row {
                background: #1e293b; border-radius: 10px; padding: 16px;
                display: flex; justify-content: space-between; align-items: center;
                transition: background 0.3s;
            }
            .cam-row.online { border-left: 4px solid #22c55e; }
            .cam-row.offline { border-left: 4px solid #ef4444; opacity: 0.6; }
            .cam-name { font-weight: 600; }
            .cam-stats { display: flex; gap: 16px; font-size: 14px; }
            .cam-stats span { color: #94a3b8; }
            .cam-stats b { color: #e2e8f0; }
            .badge {
                padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;
            }
            .badge.online { background: #22c55e; color: #fff; }
            .badge.offline { background: #ef4444; color: #fff; }
            #people-list { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
            .person-tag {
                background: #334155; padding: 4px 10px; border-radius: 8px;
                font-size: 12px; color: #e2e8f0;
            }
        </style>
    </head>
    <body>
        <h1>📹 CCTV Central Control</h1>
        <div class="subtitle" id="status">Connecting...</div>

        <div class="stats" id="stats-summary"></div>

        <div class="cameras" id="cameras-list">
            <div style="color: #64748b; text-align: center; padding: 40px;">
                Waiting for Edge Workers to connect...
            </div>
        </div>

        <script>
            const ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');
            const cameras = {};

            function render() {
                const entries = Object.values(cameras);
                const online = entries.filter(e => {
                    const now = Date.now() / 1000;
                    return (now - (e.timestamp || 0)) < 30;
                });

                // Summary stats
                const totalIn = entries.reduce((s, e) => s + (e.count_in || 0), 0);
                const totalOut = entries.reduce((s, e) => s + (e.count_out || 0), 0);
                const totalInside = entries.reduce((s, e) => s + (e.total_inside || 0), 0);
                const peopleNow = entries.reduce((s, e) => s + (e.people_now || 0), 0);

                document.getElementById('stats-summary').innerHTML = [
                    { lbl: 'Total Kamera', val: entries.length, color: '#3b82f6' },
                    { lbl: 'Online', val: online.length, color: '#22c55e' },
                    { lbl: 'Total Masuk', val: totalIn, color: '#22c55e' },
                    { lbl: 'Total Keluar', val: totalOut, color: '#ef4444' },
                    { lbl: 'Di Dalam', val: totalInside, color: '#3b82f6' },
                    { lbl: 'Saat Ini', val: peopleNow, color: '#f59e0b' },
                ].map(c => `<div class="card" style="border-color: ${c.color}">
                    <div class="lbl">${c.lbl}</div>
                    <div class="val" style="color: ${c.color}">${c.val}</div>
                </div>`).join('');

                // Camera rows
                document.getElementById('cameras-list').innerHTML = Object.values(cameras).length === 0 ?
                    '<div style="color: #64748b; text-align: center; padding: 40px;">Waiting for Edge Workers...</div>' :
                    entries.sort((a, b) => a.cam_id > b.cam_id ? 1 : -1).map(c => {
                        const now = Date.now() / 1000;
                        const isOnline = (now - (c.timestamp || 0)) < 30;
                        const recognized = c.recognized || {};
                        const names = Object.keys(recognized);
                        return `<div class="cam-row ${isOnline ? 'online' : 'offline'}">
                            <div>
                                <div class="cam-name">${c.cam_id}</div>
                                <div class="cam-stats">
                                    <span>IN <b>${c.count_in || 0}</b></span>
                                    <span>OUT <b>${c.count_out || 0}</b></span>
                                    <span>INSIDE <b>${c.total_inside || 0}</b></span>
                                    <span>NOW <b>${c.people_now || 0}</b></span>
                                </div>
                                <div id="people-list">
                                    ${names.map(n => `<span class="person-tag">${n} ${recognized[n]}×</span>`).join('')}
                                </div>
                            </div>
                            <div>
                                <span class="badge ${isOnline ? 'online' : 'offline'}">
                                    ${isOnline ? '● ONLINE' : '○ OFFLINE'}
                                </span>
                            </div>
                        </div>`;
                    }).join('');
            }

            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'init') {
                    Object.assign(cameras, msg.cameras);
                } else if (msg.type === 'update') {
                    cameras[msg.cam_id] = msg.data;
                    // Only keep active cameras (last 5 min)
                    const now = Date.now() / 1000;
                    for (const [k, v] of Object.entries(cameras)) {
                        if ((now - (v.timestamp || 0)) > 300) delete cameras[k];
                    }
                }
                render();
                document.getElementById('status').textContent =
                    `${Object.values(cameras).length} camera terdeteksi • ${new Date().toLocaleTimeString('id-ID')}`;
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = '❌ Disconnected. Reconnecting...';
                setTimeout(() => location.reload(), 5000);
            };
        </script>
    </body>
    </html>
    """)
