"""
Central Server — MQTT subscriber + FastAPI dashboard + REST API.
Updated to handle trajectory OD matrix.
"""

import os
import json
import time
import logging
import threading
import asyncio
import sqlite3
from contextlib import asynccontextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CENTRAL] %(message)s")
logger = logging.getLogger("central-server")

DB_PATH = os.environ.get("DB_PATH", "/data/central.db")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "0.0.0.0")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, cam_id TEXT NOT NULL, timestamp REAL,
    feature TEXT DEFAULT '', people_inside INTEGER DEFAULT 0, people_tracked INTEGER DEFAULT 0,
    recognized TEXT DEFAULT '{}', od_matrix TEXT DEFAULT '{}', heatmap TEXT,
    created_at TEXT DEFAULT (datetime('now')))""")
db.execute("""CREATE TABLE IF NOT EXISTS camera_status (
    cam_id TEXT PRIMARY KEY, last_seen REAL, feature TEXT DEFAULT '',
    last_analytics TEXT DEFAULT '{}')""")
db.execute("""CREATE TABLE IF NOT EXISTS camera_config (
    cam_id TEXT PRIMARY KEY, feature TEXT DEFAULT 'counting',
    roi TEXT DEFAULT '[]', zones TEXT DEFAULT '{}', name TEXT DEFAULT '', cam_url TEXT DEFAULT '')""")
db.commit()

camera_cache = {}
ws_clients = []
loop = None
mqtt_client_instance = None


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        cam_id = payload.get("cam_id", "unknown")
        ts = payload.get("timestamp", time.time())
        feature = payload.get("feature", "counting")

        od_matrix = json.dumps(payload.get("od_matrix", {}))
        recognized = json.dumps(payload.get("recognized", {}))
        heatmap = json.dumps(payload.get("heatmap"))

        db.execute("""INSERT INTO analytics (cam_id, timestamp, feature, people_inside,
            people_tracked, recognized, od_matrix, heatmap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cam_id, ts, feature, payload.get("people_inside", 0),
             payload.get("people_tracked", 0), recognized, od_matrix, heatmap))
        db.execute("""INSERT OR REPLACE INTO camera_status (cam_id, last_seen, feature, last_analytics)
            VALUES (?, ?, ?, ?)""", (cam_id, ts, feature, json.dumps(payload)))
        db.commit()

        camera_cache[cam_id] = payload

        for ws in ws_clients:
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "update", "cam_id": cam_id, "data": payload}), loop)
    except Exception as e:
        logger.warning(f"Error: {e}")


def start_mqtt():
    global mqtt_client_instance
    c = mqtt.Client(client_id="central_server")
    c.on_message = on_message
    c.on_connect = lambda cl, _, __, rc: (logger.info(f"MQTT rc={rc}"), cl.subscribe("analytics/#"))
    for a in range(30):
        try:
            c.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except:
            time.sleep(2)
    mqtt_client_instance = c
    c.loop_forever()


@asynccontextmanager
async def lifespan(app):
    global loop
    loop = asyncio.get_event_loop()
    threading.Thread(target=start_mqtt, daemon=True).start()
    yield

app = FastAPI(title="CCTV Central Control", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# REST endpoints

@app.get("/health")
async def health():
    return {"status": "ok", "cameras_online": len(camera_cache)}

@app.get("/api/cameras")
async def list_cameras():
    now = time.time()
    rows = db.execute("""SELECT cs.cam_id, cs.last_seen, cs.last_analytics, cs.feature,
        COALESCE(cc.name, '') as name, COALESCE(cc.roi, '[]') as roi,
        COALESCE(cc.zones, '{}') as zones
        FROM camera_status cs LEFT JOIN camera_config cc ON cs.cam_id = cc.cam_id""").fetchall()
    cameras = []
    for r in rows:
        a = json.loads(r[2]) if r[2] else {}
        cameras.append({
            "cam_id": r[0], "last_seen": r[1], "feature": r[3], "name": r[4],
            "online": r[1] and (now - r[1]) < 30,
            "analytics": a, "roi": json.loads(r[5]), "zones": json.loads(r[6]),
        })
    return {"cameras": cameras, "total": len(cameras),
            "online": sum(1 for c in cameras if c["online"])}

@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    data = camera_cache.get(cam_id, {})
    rows = db.execute("SELECT * FROM analytics WHERE cam_id = ? ORDER BY id DESC LIMIT 50", (cam_id,)).fetchall()
    history = []
    for r in rows:
        od = json.loads(r[6]) if r[6] and r[6] != "{}" else {}
        history.append({
            "id": r[0], "timestamp": r[2], "feature": r[3],
            "people_inside": r[4], "people_tracked": r[5],
            "od_matrix": od, "recognized": json.loads(r[6] or "{}") if r[6] else {},
        })
    return {"cam_id": cam_id, "current": data, "history": history}

@app.get("/api/summary")
async def summary():
    total = sum(d.get("people_inside", 0) for d in camera_cache.values())
    return {"cameras_online": len(camera_cache), "total_people_inside": total}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    await ws.send_json({"type": "init", "cameras": dict(camera_cache)})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.remove(ws)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
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
        .card { background: #1e293b; border-radius: 12px; padding: 16px; min-width: 140px; text-align: center; border-left: 4px solid #3b82f6; }
        .card .v { font-size: 28px; font-weight: 700; }
        .card .l { font-size: 13px; color: #94a3b8; }
        .feature-tag { display: inline-block; background: #334155; padding: 2px 10px; border-radius: 8px; font-size: 11px; margin-top: 4px; }
        .cameras { display: flex; flex-direction: column; gap: 10px; }
        .cam-row { background: #1e293b; border-radius: 10px; padding: 14px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; }
        .cam-row.online { border-left: 4px solid #22c55e; }
        .cam-row.offline { border-left: 4px solid #ef4444; opacity: 0.6; }
        .cam-name { font-weight: 600; }
        .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge.on { background: #22c55e; }
        .badge.off { background: #ef4444; }
        table.od { font-size: 13px; border-collapse: collapse; margin-top: 6px; }
        table.od th, table.od td { border: 1px solid #334155; padding: 4px 10px; text-align: center; }
        table.od th { background: #1e293b; color: #94a3b8; }
        .flex-row { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
    </style>
</head>
<body>
    <h1>📹 CCTV Central Control</h1>
    <div class="sub" id="status">Connecting...</div>
    <div class="grid" id="stats"></div>
    <div class="cameras" id="cameras-list">
        <div style="color:#64748b;text-align:center;padding:40px;">Waiting for Edge Workers...</div>
    </div>
    <script>
        const ws = new WebSocket((location.protocol=='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
        const cameras = {};

        function render() {
            const entries = Object.values(cameras);
            const now = Date.now()/1000;
            document.getElementById('stats').innerHTML = [
                ['Total Camera', entries.length, '#3b82f6'],
                ['Online', entries.filter(e=>(now-(e.timestamp||0))<30).length, '#22c55e'],
                ['Inside ROI', entries.reduce((s,e)=>s+(e.people_inside||0),0), '#f59e0b'],
            ].map(c=>`<div class="card" style="border-color:${c[2]}"><div class="l">${c[0]}</div><div class="v" style="color:${c[2]}">${c[1]}</div></div>`).join('');

            document.getElementById('cameras-list').innerHTML = entries.length===0
                ? '<div style="color:#64748b;text-align:center;padding:40px;">Waiting...</div>'
                : entries.sort((a,b)=>a.cam_id>b.cam_id?1:-1).map(c=>{
                    const isOn = (now-(c.timestamp||0))<30;
                    const feat = c.feature||'counting';
                    let extra = '';

                    if (feat==='trajectory') {
                        const od = c.od_matrix || {};
                        const labels = c.trajectory_labels || Object.keys(od);
                        if (labels.length) {
                            let rows = '';
                            labels.forEach(o=>{
                                let cells = `<td><b>${o}</b></td>`;
                                labels.forEach(d=>{
                                    cells += `<td>${(od[o]||{})[d]||0}</td>`;
                                });
                                rows += `<tr>${cells}</tr>`;
                            });
                            let header = '<th></th>'+labels.map(l=>`<th>${l}</th>`).join('');
                            extra = `<table class="od"><tr>${header}</tr>${rows}</table>`;
                        }
                    } else if (feat==='face_rec') {
                        const rec = c.recognized||{};
                        const names = Object.keys(rec);
                        extra = names.length
                            ? names.map(n=>`<span class="feature-tag">${n} ${rec[n]}x</span>`).join('')
                            : '<span class="feature-tag">No face</span>';
                    } else if (feat==='heatmap') {
                        extra = '<span class="feature-tag">🔥 Heatmap active</span>';
                    } else {
                        extra = `<span class="feature-tag">👥 Inside: ${c.people_inside||0}</span>`;
                    }

                    return `<div class="cam-row ${isOn?'online':'offline'}">
                        <div>
                            <div class="cam-name">${c.cam_name||c.cam_id}</div>
                            <div class="flex-row">
                                <span class="feature-tag">${feat}</span>
                                ${extra}
                            </div>
                        </div>
                        <div><span class="badge ${isOn?'on':'off'}">${isOn?'● ONLINE':'○ OFFLINE'}</span></div>
                    </div>`;
                }).join('');
        }

        ws.onmessage = e => {
            const msg = JSON.parse(e.data);
            if (msg.type==='init') Object.assign(cameras, msg.cameras);
            else if (msg.type==='update') cameras[msg.cam_id]=msg.data;
            render();
            document.getElementById('status').textContent = Object.keys(cameras).length+' camera(s) • '+new Date().toLocaleTimeString('id-ID');
        };
        ws.onclose = ()=>{ document.getElementById('status').textContent='❌ Disconnected'; setTimeout(()=>location.reload(),3000); };
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
