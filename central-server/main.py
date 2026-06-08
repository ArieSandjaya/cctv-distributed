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
    <title>CCTOps</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Inter',system-ui,sans-serif;background:#0b1120;color:#e2e8f0;padding:20px;min-height:100vh}
        .header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:8px}
        .header h1{font-size:22px;font-weight:700;letter-spacing:-.3px;display:flex;align-items:center;gap:8px}
        .header h1 span{background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        #status{border:1px solid #1e293b;border-radius:8px;padding:6px 14px;font-size:13px;color:#94a3b8}
        .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:20px}
        .stat-card{background:#131c31;border-radius:10px;padding:14px;text-align:center;border-top:3px solid #3b82f6}
        .stat-card .v{font-size:26px;font-weight:700}
        .stat-card .l{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
        .cam-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
        .cam-card{background:#131c31;border-radius:12px;padding:14px;cursor:pointer;transition:all .15s;border:1px solid transparent}
        .cam-card:hover{border-color:#334155;transform:translateY(-1px)}
        .cam-card .top{display:flex;justify-content:space-between;align-items:start;margin-bottom:8px}
        .cam-card .name{font-weight:600;font-size:14px}
        .cam-card .feature-tag{background:#1e293b;padding:2px 8px;border-radius:6px;font-size:10px;color:#94a3b8}
        .cam-card .status-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
        .cam-card .status-dot.on{background:#22c55e;box-shadow:0 0 6px #22c55e66}
        .cam-card .status-dot.off{background:#ef4444}
        .cam-card .extra{font-size:12px;color:#94a3b8;margin-top:6px;display:flex;gap:6px;flex-wrap:wrap}
        .cam-card .extra .tag{background:#1e293b;padding:2px 8px;border-radius:5px;font-size:11px}
        .cam-card canvas{width:100%;height:64px;border-radius:6px;margin-top:8px;background:#0b1120;image-rendering:pixelated}
        .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(4px);z-index:100;align-items:center;justify-content:center;padding:20px}
        .overlay.open{display:flex}
        .modal{background:#131c31;border-radius:16px;padding:24px;width:100%;max-width:650px;max-height:80vh;overflow-y:auto;border:1px solid #1e293b;position:relative}
        .modal .close{position:absolute;top:12px;right:16px;background:none;border:none;color:#64748b;font-size:22px;cursor:pointer}
        .modal .close:hover{color:#e2e8f0}
        .modal h2{font-size:18px;margin-bottom:4px}
        .modal .sub{color:#64748b;font-size:13px;margin-bottom:14px}
        .modal table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
        .modal td,.modal th{border:1px solid #1e293b;padding:6px 10px;text-align:center}
        .modal th{background:#0b1120;color:#94a3b8;font-weight:600}
        .modal .fl{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
        .modal .fl .fi{background:#1e293b;padding:4px 12px;border-radius:8px;font-size:13px}
        .modal canvas.hm{width:100%;height:120px;border-radius:8px;background:#0b1120;margin-top:8px;image-rendering:pixelated}
    </style>
</head>
<body>
    <div class="header">
        <h1><span>⦿</span> CCTOps</h1>
        <div id="status">Connecting...</div>
    </div>
    <div class="stats" id="stats"></div>
    <div class="cam-grid" id="camGrid"><div style="color:#64748b;text-align:center;padding:40px;grid-column:1/-1">Waiting...</div></div>
    <div class="overlay" id="overlay"><div class="modal" id="modal"><button class="close" onclick="closeModal()">&times;</button><h2 id="modalTitle">Camera</h2><div class="sub" id="modalFeature">feature</div><div id="modalBody"></div></div></div>
    <script>
        const ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
        const cameras={};

        function hc(v){v=Math.max(0,Math.min(1,v));if(v<.25)return 'hsl(220,80%,'+(30+v*60)+'%)';if(v<.5)return 'hsl('+(220-(v-.25)*300)+',80%,50%)';if(v<.75)return 'hsl('+(145-(v-.5)*300)+',80%,50%)';return 'hsl('+(45-(v-.75)*30)+',90%,50%)'}

        function dhm(can,grid,w,h){if(!can||!grid||!grid.length)return;const ctx=can.getContext('2d');can.width=w;can.height=h;const bw=Math.ceil(w/grid[0].length),bh=Math.ceil(h/grid.length);for(let y=0;y<grid.length;y++){for(let x=0;x<grid[y].length;x++){ctx.fillStyle=hc(grid[y][x]);ctx.fillRect(x*bw,y*bh,bw,bh)}}}

        function render(){
            const e=Object.values(cameras);const now=Date.now()/1000;const on=e.filter(c=>(now-(c.timestamp||0))<30);const inside=on.reduce((s,c)=>s+(c.people_inside||0),0);
            document.getElementById('stats').innerHTML=[['Camera',e.length,'#3b82f6'],[(e.length?'Online of':'Wait')+(e.length?'/'+e.length:''),e.length?on.length:0,'#22c55e'],['Inside',inside,'#f59e0b']].map(c=>`<div class="stat-card" style="border-color:${c[2]}"><div class="v" style="color:${c[2]}">${c[1]}</div><div class="l">${c[0]}</div></div>`).join('');
            const gr=document.getElementById('camGrid');
            if(!e.length){gr.innerHTML='<div style="color:#64748b;text-align:center;padding:40px;grid-column:1/-1">Waiting...</div>';return}
            gr.innerHTML=e.sort((a,b)=>(a.cam_name||a.cam_id)>(b.cam_name||b.cam_id)?1:-1).map(c=>{
                const i=now-(c.timestamp||0)<30;const f=c.feature||'counting';let x='',cid='';
                if(f==='trajectory'){const od=c.od_matrix||{};const lb=c.trajectory_labels||Object.keys(od);const t=lb.reduce((s,l)=>s+Object.values(od[l]||{}).reduce((a,b)=>a+b,0),0);x='<span class="tag">'+t+' mov</span>'}
                else if(f==='face_rec'){const r=c.recognized||{};const n=Object.keys(r);x=n.length?n.map(n=>'<span class="tag">'+n+' '+r[n]+'x</span>').join(''):'<span class="tag">—</span>'}
                else if(f==='heatmap'){cid='hm-'+c.cam_id;x='<span class="tag">🔥</span>'}
                else{x='<span class="tag">'+((c.people_inside||0)+' inside')+'</span>'}
                return '<div class="cam-card" onclick="openCam(\''+c.cam_id+'\')"><div class="top"><div><div class="name">'+(c.cam_name||c.cam_id)+'</div><span class="feature-tag">'+f+'</span></div><span class="status-dot '+(i?'on':'off')+'"></span></div><div class="extra">'+x+'</div>'+(cid?'<canvas id="'+cid+'"></canvas>':'')+'</div>'
            }).join('');
            e.forEach(c=>{if(c.feature==='heatmap'){const el=document.getElementById('hm-'+c.cam_id);if(el&&c.heatmap)dhm(el,c.heatmap,32,24)}})
        }

        function openCam(id){
            const c=cameras[id];if(!c)return;
            document.getElementById('modalTitle').textContent=c.cam_name||id;
            document.getElementById('modalFeature').textContent='Feature: '+c.feature;
            const b=document.getElementById('modalBody');let h='';
            if(c.feature==='trajectory'){
                const od=c.od_matrix||{};const lb=c.trajectory_labels||Object.keys(od);
                if(lb.length){let ro='';lb.forEach(o=>{let ce='<td><b>'+o+'</b></td>';lb.forEach(d=>ce+='<td>'+(od[o]?.[d]||0)+'</td>');ro+='<tr>'+ce+'</tr>'});h='<table><tr><th></th>'+lb.map(l=>'<th>'+l+'</th>').join('')+'</tr>'+ro+'</table>'}
            }else if(c.feature==='face_rec'){
                const r=c.recognized||{};const n=Object.keys(r);
                h=n.length?'<div class="fl">'+n.map(n=>'<span class="fi">'+n+' <b>'+r[n]+'x</b></span>').join('')+'</div>':'<p style="color:#64748b">No faces</p>'
            }else if(c.feature==='heatmap'){
                h='<canvas class="hm" id="dhm"></canvas>';setTimeout(()=>{const el=document.getElementById('dhm');if(el&&c.heatmap)dhm(el,c.heatmap,64,48)},50)
            }else{h='<p style="color:#94a3b8">Inside: <b>'+(c.people_inside||0)+'</b> | Tracked: <b>'+(c.people_tracked||0)+'</b></p>'}
            b.innerHTML=h||'<p style="color:#64748b">No data</p>';
            document.getElementById('overlay').classList.add('open')
        }
        function closeModal(){document.getElementById('overlay').classList.remove('open')}
        document.getElementById('overlay').onclick=e=>{if(e.target===e.currentTarget)closeModal()};
        ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='init')Object.assign(cameras,m.cameras);else if(m.type==='update')cameras[m.cam_id]=m.data;render();document.getElementById('status').textContent=Object.keys(cameras).length+' cam • '+new Date().toLocaleTimeString('id-ID')};
        ws.onclose=()=>{document.getElementById('status').textContent='\u274c Disconnected';setTimeout(()=>location.reload(),3000)};
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
