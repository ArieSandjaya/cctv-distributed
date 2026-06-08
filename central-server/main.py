"""
Central Server (The Brain).
- Receives analytics from all Edge Workers (MQTT)
- Stores data in SQLite
- Sends ROI config commands via MQTT
- Serves unified dashboard
"""

import os
import json
import time
import logging
import threading
from contextlib import asynccontextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CENTRAL] %(message)s")
logger = logging.getLogger("central-server")

# ── Config ──────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "/data/central.db")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "0.0.0.0")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# ── Database ────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            people_inside INTEGER DEFAULT 0,
            people_tracked INTEGER DEFAULT 0,
            recognized TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS camera_config (
            cam_id TEXT PRIMARY KEY,
            roi TEXT DEFAULT '[]',
            name TEXT DEFAULT '',
            cam_url TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS camera_status (
            cam_id TEXT PRIMARY KEY,
            last_seen REAL,
            last_analytics TEXT
        )
    """)
    conn.commit()
    return conn

db = init_db()
camera_cache: dict[str, dict] = {}
ws_clients: list[WebSocket] = []
loop = None


# ── MQTT ────────────────────────────────────────────────────────

mqtt_client_instance: mqtt.Client | None = None

def on_connect_central(client, userdata, flags, rc):
    logger.info(f"MQTT connected (rc={rc})")
    client.subscribe("analytics/#")

def on_message_central(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        cam_id = payload.get("cam_id", "unknown")
        ts = payload.get("timestamp", time.time())
        recognized = payload.get("recognized", {})

        db.execute(
            """INSERT INTO analytics (cam_id, timestamp, people_inside, people_tracked, recognized)
               VALUES (?, ?, ?, ?, ?)""",
            (cam_id, ts, payload.get("people_inside", 0),
             payload.get("people_tracked", 0), json.dumps(recognized))
        )
        db.execute(
            "INSERT OR REPLACE INTO camera_status (cam_id, last_seen, last_analytics) VALUES (?, ?, ?)",
            (cam_id, ts, json.dumps(payload))
        )
        db.commit()

        camera_cache[cam_id] = payload

        # Push to WS clients
        for ws in list(ws_clients):
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "update", "cam_id": cam_id, "data": payload}), loop
                )
            except:
                try:
                    ws_clients.remove(ws)
                except ValueError:
                    pass

    except Exception as e:
        logger.warning(f"MQTT error: {e}")


def start_mqtt():
    global mqtt_client_instance
    c = mqtt.Client(client_id="central_server")
    c.on_connect = on_connect_central
    c.on_message = on_message_central
    for attempt in range(30):
        try:
            c.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except:
            time.sleep(2)
    else:
        logger.error("MQTT connection failed")
        return
    mqtt_client_instance = c
    c.loop_forever()


# ── FastAPI ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    global loop
    loop = asyncio.get_event_loop()
    t = threading.Thread(target=start_mqtt, daemon=True)
    t.start()
    logger.info("Central Server started")
    yield

app = FastAPI(title="CCTV Central Control", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── REST Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "cameras_online": len(camera_cache)}


@app.get("/api/cameras")
async def list_cameras():
    now = time.time()
    rows = db.execute("SELECT c.cam_id, c.last_seen, c.last_analytics, "
                      "COALESCE(cf.name, '') as name, COALESCE(cf.roi, '[]') as roi "
                      "FROM camera_status c LEFT JOIN camera_config cf ON c.cam_id = cf.cam_id").fetchall()
    cameras = []
    for row in rows:
        cam_id, last_seen, last_analytics, name, roi = row
        analytics = json.loads(last_analytics) if last_analytics else {}
        cameras.append({
            "cam_id": cam_id,
            "name": name,
            "online": last_seen and (now - last_seen) < 30,
            "last_seen": last_seen,
            "analytics": analytics,
            "roi": json.loads(roi) if roi else [],
        })
    return {"cameras": cameras, "total": len(cameras),
            "online": sum(1 for c in cameras if c["online"])}


@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    data = camera_cache.get(cam_id, {})
    rows = db.execute(
        "SELECT * FROM analytics WHERE cam_id = ? ORDER BY id DESC LIMIT 50", (cam_id,)
    ).fetchall()
    history = [{
        "id": r[0], "timestamp": r[2], "people_inside": r[3],
        "people_tracked": r[4], "recognized": json.loads(r[5]) if r[5] else {},
    } for r in rows]

    config_row = db.execute("SELECT * FROM camera_config WHERE cam_id = ?", (cam_id,)).fetchone()
    config = {"roi": json.loads(config_row[1]) if config_row and config_row[1] else [],
              "name": config_row[2] if config_row else "",
              "cam_url": config_row[3] if config_row else ""} if config_row else {}

    return {"cam_id": cam_id, "current": data, "history": history, "config": config}


@app.post("/api/config/{cam_id}")
async def configure_camera(cam_id: str, body: dict):
    """Send ROI config to edge worker via MQTT."""
    name = body.get("name", "")
    cam_url = body.get("cam_url", "")
    roi = body.get("roi", body.get("roi_points", []))

    # Save to DB
    db.execute(
        "INSERT OR REPLACE INTO camera_config (cam_id, name, roi, cam_url) VALUES (?, ?, ?, ?)",
        (cam_id, name, json.dumps(roi), cam_url)
    )
    db.commit()

    # Send via MQTT
    msg = {"roi": roi, "timestamp": time.time()}
    if mqtt_client_instance:
        mqtt_client_instance.publish(f"config/{cam_id}", json.dumps(msg), qos=1)
        logger.info(f"ROI config sent to {cam_id}: {len(roi)} points")

    return {"status": "sent", "cam_id": cam_id, "roi_points": len(roi)}


@app.get("/api/cameras/{cam_id}/roi")
async def get_roi(cam_id: str):
    row = db.execute("SELECT roi FROM camera_config WHERE cam_id = ?", (cam_id,)).fetchone()
    if not row:
        return {"cam_id": cam_id, "roi": []}
    return {"cam_id": cam_id, "roi": json.loads(row[0])}


@app.get("/api/summary")
async def summary():
    total_inside = 0
    for data in camera_cache.values():
        total_inside += data.get("people_inside", 0)
    return {
        "cameras_online": len(camera_cache),
        "total_people_inside": total_inside,
        "timestamp": time.time(),
    }


# ── WebSocket ───────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    await ws.send_json({"type": "init", "cameras": camera_cache})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass


# ── Dashboard ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <title>CCTV Central Control</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
        h1 { margin-bottom: 4px; }
        .sub { color: #64748b; margin-bottom: 24px; }
        .grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
        .card {
            background: #1e293b; border-radius: 12px; padding: 16px; min-width: 140px; text-align: center;
            border-left: 4px solid #3b82f6;
        }
        .card .v { font-size: 28px; font-weight: 700; }
        .card .l { font-size: 13px; color: #94a3b8; }
        .cameras { display: flex; flex-direction: column; gap: 10px; }
        .cam-row {
            background: #1e293b; border-radius: 10px; padding: 14px;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap;
        }
        .cam-row.online { border-left: 4px solid #22c55e; }
        .cam-row.offline { border-left: 4px solid #ef4444; opacity: 0.6; }
        .cam-name { font-weight: 600; }
        .cam-stats { display: flex; gap: 14px; font-size: 14px; margin-top: 4px; }
        .cam-stats span { color: #94a3b8; }
        .cam-stats b { color: #e2e8f0; }
        .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; color: #fff; }
        .badge.on { background: #22c55e; }
        .badge.off { background: #ef4444; }
        .person-tag { background: #334155; padding: 3px 8px; border-radius: 6px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>📹 CCTV Central Control</h1>
    <div class="sub" id="status">Connecting...</div>
    <div class="grid" id="stats"></div>
    <div class="cameras" id="cameras-list">
        <div style="color: #64748b; text-align: center; padding: 40px;">Waiting for Edge Workers...</div>
    </div>
    <script>
        const ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');
        const cameras = {};

        function render() {
            const entries = Object.values(cameras);
            const now = Date.now() / 1000;
            const online = entries.filter(e => (now - (e.timestamp || 0)) < 30);

            document.getElementById('stats').innerHTML = [
                ['Total Kamera', entries.length, '#3b82f6'],
                ['Online', online.length, '#22c55e'],
                ['Total Di Dalam', entries.reduce((s, e) => s + (e.people_inside || 0), 0), '#f59e0b'],
            ].map(c => `<div class="card" style="border-color:${c[2]}"><div class="l">${c[0]}</div><div class="v" style="color:${c[2]}">${c[1]}</div></div>`).join('');

            document.getElementById('cameras-list').innerHTML = entries.length === 0 ?
                '<div style="color: #64748b; text-align: center; padding: 40px;">Waiting...</div>' :
                entries.sort((a, b) => a.cam_id > b.cam_id ? 1 : -1).map(c => {
                    const isOn = (now - (c.timestamp || 0)) < 30;
                    const rec = c.recognized || {};
                    const names = Object.keys(rec);
                    return `<div class="cam-row ${isOn ? 'online' : 'offline'}">
                        <div>
                            <div class="cam-name">${c.cam_name || c.cam_id}</div>
                            <div class="cam-stats">
                                <span>Inside <b>${c.people_inside || 0}</b></span>
                                <span>Tracked <b>${c.people_tracked || 0}</b></span>
                            </div>
                            ${names.length ? '<div style="margin-top:6px">' + names.map(n => `<span class="person-tag">${n} ${rec[n]}×</span>`).join('') + '</div>' : ''}
                        </div>
                        <div><span class="badge ${isOn ? 'on' : 'off'}">${isOn ? '● ONLINE' : '○ OFFLINE'}</span></div>
                    </div>`;
                }).join('');
        }

        ws.onmessage = e => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'init') Object.assign(cameras, msg.cameras);
            else if (msg.type === 'update') { cameras[msg.cam_id] = msg.data; }
            render();
            document.getElementById('status').textContent = Object.keys(cameras).length + ' camera(s) • ' + new Date().toLocaleTimeString('id-ID');
        };
        ws.onclose = () => { document.getElementById('status').textContent = '❌ Disconnected'; setTimeout(() => location.reload(), 3000); };
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
