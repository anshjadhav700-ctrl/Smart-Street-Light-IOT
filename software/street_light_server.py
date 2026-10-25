import sys
import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import torch
from flask import Flask, Response, request, jsonify
from pyngrok import ngrok
import requests
from collections import deque

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
# ---- FILL THESE IN BEFORE RUNNING -----------------------------------------
FIREBASE_URL      = ""   # TODO: your Firebase Realtime Database URL, no trailing slash
                         #       e.g. https://your-project-default-rtdb.firebaseio.com
NGROK_AUTH_TOKEN  = ""   # TODO (optional): your ngrok authtoken for a public URL.
                         #       Leave blank to use the dashboard on your local network only.
# ----------------------------------------------------------------------------
FIREBASE_URL = FIREBASE_URL.strip().rstrip("/")
if not FIREBASE_URL:
    sys.exit("ERROR: FIREBASE_URL is empty. Open this file and fill it in at the top (see README).")

NIGHT_START       = 18
NIGHT_END         = 6
DETECTION_HOLD    = 5      # secs light stays ON after last detection
CROWD_THRESHOLD   = 5      # persons count to trigger crowd alert
FIRE_FRAME_THRESH = 4      # consecutive fire frames needed
FIRE_PIXEL_THRESH = 8000   # HSV pixels to consider fire present
ACCIDENT_OVERLAP  = 4000   # px² bbox overlap to flag accident
CONF_THRESHOLD    = 0.45

# ═══════════════════════════════════════════════════════════
#  NGROK + MODEL
# ═══════════════════════════════════════════════════════════
if NGROK_AUTH_TOKEN.strip():
    ngrok.set_auth_token(NGROK_AUTH_TOKEN.strip())
    public_url = ngrok.connect(5000)
    print("🌍 Public URL:", public_url)
else:
    print("ℹ️  NGROK_AUTH_TOKEN is blank -> dashboard on local network only: http://<this-pc-ip>:5000")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚡ Device: {device.upper()}")
model = YOLO("yolov8n.pt")
model.to(device)

# ═══════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ═══════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════
lock          = threading.Lock()
raw_frame     = None   # 320×320 for detection
display_frame = None   # full-res
latest_frame  = None   # annotated → MJPEG

persons    = 0
vehicles   = 0
boxes      = []        # vehicle bboxes for accident check
draw_boxes = []        # (x1,y1,x2,y2,color,label)

last_detection_time = 0
fire_frames         = 0
emergency_sent      = False
_fire_state         = False
_crowd_state        = False
_accident_state     = False

mode_web       = "Auto"
light_web      = "OFF"
last_light_sent = None

fps_times = deque(maxlen=30)

# ═══════════════════════════════════════════════════════════
#  FIREBASE
# ═══════════════════════════════════════════════════════════
def send_light(state):
    global last_light_sent
    if state == last_light_sent:
        return
    try:
        requests.put(f"{FIREBASE_URL}/light.json", json=state, timeout=2)
        print(f"[Firebase] light → {state}")
        last_light_sent = state
    except Exception as e:
        print(f"[Firebase] light error: {e}")

def send_emergency(state: bool):
    """Push emergency flag so ESP32 can force light ON."""
    try:
        requests.put(f"{FIREBASE_URL}/emergency.json", json=state, timeout=2)
    except Exception as e:
        print(f"[Firebase] emergency error: {e}")

# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def is_night():
    h = datetime.now().hour
    return h >= NIGHT_START or h < NIGHT_END

def detect_fire(frame_bgr):
    """
    Run on the 320×320 detection frame — much cheaper than 1280×720.
    Dual HSV range + morph cleanup + gradual counter.
    """
    global fire_frames
    small = cv2.resize(frame_bgr, (320, 320)) if frame_bgr.shape[0] > 320 else frame_bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Range 1: bright red-orange core
    m1 = cv2.inRange(hsv, np.array([0,  150, 200]), np.array([20, 255, 255]))
    # Range 2: yellow-orange outer flame
    m2 = cv2.inRange(hsv, np.array([20, 120, 180]), np.array([38, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    pix  = cv2.countNonZero(mask)
    fire_frames = fire_frames + 1 if pix > FIRE_PIXEL_THRESH else max(0, fire_frames - 1)
    return fire_frames >= FIRE_FRAME_THRESH

def detect_accident(vehicle_boxes):
    """
    Bounding-box intersection area check. Ignores tiny boxes.
    """
    if len(vehicle_boxes) < 2:
        return False
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            x1,y1,x2,y2 = vehicle_boxes[i]
            x3,y3,x4,y4 = vehicle_boxes[j]
            ix = max(0, min(x2,x4) - max(x1,x3))
            iy = max(0, min(y2,y4) - max(y1,y3))
            if ix * iy > ACCIDENT_OVERLAP:
                return True
    return False

# ═══════════════════════════════════════════════════════════
#  THREADS
# ═══════════════════════════════════════════════════════════
def camera_thread():
    global raw_frame, display_frame
    while True:
        ret, f = cap.read()
        if ret:
            small = cv2.resize(f, (320, 320))   # do heavy work OUTSIDE lock
            display_frame = f                    # GIL makes these atomic enough
            raw_frame     = small
        # NO sleep — grab as fast as camera allows

def detection_thread():
    global persons, vehicles, boxes, draw_boxes
    while True:
        fa = raw_frame        # snapshot — no lock needed for reads
        fd = display_frame
        if fa is None or fd is None:
            time.sleep(0.01)
            continue
        try:
            results = model(fa, device=device, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"[Detect] {e}")
            continue

        p, v, b, d = 0, 0, [], []
        fh, fw = fd.shape[:2]
        hs, ws = fh/320, fw/320

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                name = model.names[cls]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1=int(x1*ws); x2=int(x2*ws)
                y1=int(y1*hs); y2=int(y2*hs)
                if (x2-x1)<20 or (y2-y1)<20:
                    continue
                if name == "person":
                    p += 1
                    d.append((x1,y1,x2,y2,(0,230,80),f"Person {conf:.0%}"))
                elif name in ["car","motorcycle","truck","bus"]:
                    v += 1
                    b.append((x1,y1,x2,y2))
                    d.append((x1,y1,x2,y2,(30,150,255),f"{name.title()} {conf:.0%}"))

        persons, vehicles, boxes, draw_boxes = p, v, b, d

        # ── All detections run here in the detection thread, NOT in main loop ──
        global _fire_state, _crowd_state, _accident_state, emergency_sent
        _fire_state     = detect_fire(fa)
        _crowd_state    = p >= CROWD_THRESHOLD
        _accident_state = detect_accident(b)
        em = _fire_state or _crowd_state or _accident_state
        if em and not emergency_sent:
            emergency_sent = True
            threading.Thread(target=send_emergency, args=(True,),  daemon=True).start()
        elif not em and emergency_sent:
            emergency_sent = False
            threading.Thread(target=send_emergency, args=(False,), daemon=True).start()

# ═══════════════════════════════════════════════════════════
#  FLASK
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)

def generate_frames():
    while True:
        f = latest_frame      # atomic read, no lock needed
        if f is None:
            time.sleep(0.01)
            continue
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global mode_web, light_web
    data = request.json or {}
    if "mode"  in data: mode_web  = data["mode"]
    if "light" in data:
        light_web = data["light"]
        send_light(light_web)
    return jsonify({"status": "ok"})

@app.route('/status')
def status():
    with lock:
        p, v = persons, vehicles
        em   = emergency_sent
    fps = round(len(fps_times) /
                max(fps_times[-1]-fps_times[0], 0.001), 1) \
          if len(fps_times) > 1 else 0
    return jsonify({
        "emergency": em,
        "persons":   p,
        "vehicles":  v,
        "night":     is_night(),
        "light":     last_light_sent or "OFF",
        "mode":      mode_web,
        "fps":       fps
    })

@app.route('/alerts')
def alerts():
    return jsonify({
        "fire":     _fire_state,
        "crowd":    _crowd_state,
        "accident": _accident_state
    })

@app.route('/')
def index():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OrchaLight · Smart Solar Street Light</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060a10;--surface:#0c1118;--card:#111822;--border:#1e2d3d;
  --accent:#00d4ff;--accent2:#0088ff;--green:#00ff88;
  --red:#ff3355;--yellow:#ffcc00;--text:#c8daea;--muted:#4a6070;
  --font-ui:'Rajdhani',sans-serif;--font-mono:'JetBrains Mono',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-ui);overflow-x:hidden}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px);}

/* HEADER */
header{display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px;border-bottom:1px solid var(--border);
  background:rgba(6,10,16,.92);backdrop-filter:blur(12px);
  position:sticky;top:0;z-index:100;}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:grid;place-items:center;font-size:18px;
  box-shadow:0 0 20px rgba(0,212,255,.4);}
.logo-text{font-size:20px;font-weight:700;letter-spacing:2px;color:#fff}
.logo-sub{font-size:10px;font-family:var(--font-mono);color:var(--muted);letter-spacing:3px}
.hdr-right{display:flex;align-items:center;gap:18px}
.sys-time{font-family:var(--font-mono);font-size:13px;color:var(--accent);letter-spacing:1px}
.sys-ip{font-family:var(--font-mono);font-size:11px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:blink 2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* LAYOUT */
.layout{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px 20px;min-height:calc(100vh - 64px)}
@media(max-width:960px){.layout{grid-template-columns:1fr}}

/* VIDEO */
.video-wrap{position:relative;border-radius:12px;overflow:hidden;
  border:1px solid var(--border);background:#000;
  box-shadow:0 0 40px rgba(0,136,255,.12);}
.video-wrap img{width:100%;display:block;min-height:240px;object-fit:cover}
.vid-hud{position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(to bottom,rgba(6,10,16,.5) 0%,transparent 18%,transparent 76%,rgba(6,10,16,.6) 100%);}
.corner{position:absolute;width:22px;height:22px;border-color:var(--accent);border-style:solid;opacity:.7}
.corner.tl{top:12px;left:12px;border-width:2px 0 0 2px}
.corner.tr{top:12px;right:12px;border-width:2px 2px 0 0}
.corner.bl{bottom:12px;left:12px;border-width:0 0 2px 2px}
.corner.br{bottom:12px;right:12px;border-width:0 2px 2px 0}
.vid-label{position:absolute;top:14px;left:50%;transform:translateX(-50%);
  font-family:var(--font-mono);font-size:10px;letter-spacing:3px;color:var(--accent);opacity:.8;}
.fps-badge{position:absolute;top:12px;right:14px;
  font-family:var(--font-mono);font-size:11px;color:var(--green);
  background:rgba(0,0,0,.6);padding:3px 8px;border-radius:4px;}

/* METRICS */
.metrics-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px;}
.metric{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;text-align:center;transition:border-color .3s,box-shadow .3s;}
.metric.active{border-color:var(--accent);box-shadow:0 0 14px rgba(0,212,255,.18)}
.metric.alert{border-color:var(--red)!important;box-shadow:0 0 16px rgba(255,51,85,.28)!important}
.metric-val{font-size:26px;font-weight:700;font-family:var(--font-mono);color:#fff;line-height:1}
.metric-lbl{font-size:11px;color:var(--muted);letter-spacing:1px;margin-top:4px;text-transform:uppercase}

/* SIDE */
.side{display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;}
.card-title{font-size:11px;letter-spacing:3px;text-transform:uppercase;
  color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:8px;}
.card-title::before{content:'';width:3px;height:12px;background:var(--accent);border-radius:2px}

/* MODE BUTTONS */
.mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mode-btn{padding:12px;border-radius:8px;border:1px solid var(--border);
  background:transparent;color:var(--muted);font-family:var(--font-ui);
  font-size:14px;font-weight:600;letter-spacing:1px;cursor:pointer;transition:all .2s;}
.mode-btn:hover{border-color:var(--accent);color:var(--text)}
.mode-btn.active{background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(0,136,255,.15));
  border-color:var(--accent);color:var(--accent);box-shadow:0 0 14px rgba(0,212,255,.2);}

/* TOGGLE */
.toggle-row{display:flex;align-items:center;justify-content:space-between;padding:4px 0}
.toggle-row span{font-size:14px;color:var(--text)}
.tog{position:relative;display:inline-block;width:52px;height:28px;cursor:pointer}
.tog input{display:none}
.tog-track{position:absolute;inset:0;border-radius:28px;background:#1a2535;
  border:1px solid var(--border);transition:.25s;}
.tog-thumb{position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;
  background:var(--muted);transition:.25s;}
.tog input:checked + .tog-track{background:rgba(0,255,136,.15);border-color:var(--green)}
.tog input:checked ~ .tog-thumb{background:var(--green);transform:translateX(24px);box-shadow:0 0 8px var(--green)}

/* BULB */
.light-display{text-align:center;padding:8px 0}
.bulb{font-size:46px;display:block;filter:grayscale(1) brightness(.4);
  transition:filter .4s,transform .3s;}
.bulb.on{filter:grayscale(0) brightness(1.1) drop-shadow(0 0 22px #ffcc00);transform:scale(1.1)}
.light-state{font-family:var(--font-mono);font-size:12px;letter-spacing:2px;
  margin-top:6px;color:var(--muted);transition:color .3s}
.light-state.on{color:var(--yellow)}

/* ALERTS */
.alert-list{display:flex;flex-direction:column;gap:8px}
.alert-item{display:flex;align-items:center;gap:10px;
  padding:10px 12px;border-radius:8px;background:rgba(255,255,255,.02);
  border:1px solid var(--border);font-size:13px;font-weight:600;
  letter-spacing:.5px;transition:all .3s;color:#3a5060;}
.alert-item.triggered{background:rgba(255,51,85,.08);border-color:var(--red);
  color:var(--red);animation:aPulse 1.2s ease-in-out infinite;}
@keyframes aPulse{0%,100%{box-shadow:none}50%{box-shadow:0 0 18px rgba(255,51,85,.3)}}
.a-icon{font-size:18px;width:24px;text-align:center}
.a-dot{width:6px;height:6px;border-radius:50%;background:currentColor;margin-left:auto;flex-shrink:0}

/* LOG */
.log-box{height:118px;overflow-y:auto;font-family:var(--font-mono);
  font-size:11px;line-height:1.8;color:var(--muted);
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;}
.log-entry.warn{color:var(--yellow)}.log-entry.crit{color:var(--red)}.log-entry.info{color:var(--accent)}

/* EMERGENCY */
.e-overlay{display:none;position:fixed;inset:0;z-index:1000;pointer-events:none;
  animation:flashBg 1s ease-in-out infinite;}
@keyframes flashBg{0%,100%{background:rgba(255,30,50,.04)}50%{background:rgba(255,30,50,.14)}}
.e-banner{display:none;position:fixed;top:80px;left:50%;transform:translateX(-50%);
  background:var(--red);color:#fff;font-family:var(--font-mono);
  font-size:13px;letter-spacing:3px;padding:10px 28px;border-radius:6px;
  z-index:1001;box-shadow:0 0 40px rgba(255,51,85,.6);
  animation:bannerPop .5s cubic-bezier(.36,.07,.19,.97) both;}
@keyframes bannerPop{0%{transform:translateX(-50%) scale(.7);opacity:0}100%{transform:translateX(-50%) scale(1);opacity:1}}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>
<div class="e-overlay" id="eOverlay"></div>
<div class="e-banner"  id="eBanner">⚠ EMERGENCY DETECTED</div>

<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div>
      <div class="logo-text">ORCHALIGHT</div>
      <div class="logo-sub">SMART SOLAR STREET LIGHT v2.0</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="sys-ip" id="sysIP">--</span>
    <span class="sys-time" id="sysTime">--:--:--</span>
    <div class="dot"></div>
  </div>
</header>

<div class="layout">
  <!-- LEFT -->
  <div>
    <div class="video-wrap">
      <img src="/video" alt="Live Feed">
      <div class="vid-hud">
        <div class="corner tl"></div><div class="corner tr"></div>
        <div class="corner bl"></div><div class="corner br"></div>
        <div class="vid-label">LIVE · CAM-01</div>
        <div class="fps-badge" id="fpsBadge">-- FPS</div>
      </div>
    </div>
    <div class="metrics-bar">
      <div class="metric" id="mPersons">
        <div class="metric-val" id="valPersons">0</div>
        <div class="metric-lbl">👤 Persons</div>
      </div>
      <div class="metric" id="mVehicles">
        <div class="metric-val" id="valVehicles">0</div>
        <div class="metric-lbl">🚗 Vehicles</div>
      </div>
      <div class="metric" id="mMode">
        <div class="metric-val" id="valMode" style="font-size:16px;padding-top:5px">Auto</div>
        <div class="metric-lbl">⚙ Mode</div>
      </div>
      <div class="metric" id="mTime">
        <div class="metric-val" id="valTime" style="font-size:16px;padding-top:5px">--</div>
        <div class="metric-lbl">🌙 Period</div>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="side">
    <div class="card">
      <div class="card-title">Operating Mode</div>
      <div class="mode-grid">
        <button class="mode-btn active" id="btnAuto"     onclick="setMode('Auto')">Auto Mode</button>
        <button class="mode-btn"        id="btnManual" onclick="setMode('Manual')">Manual</button>
      </div>
    </div>

    <div class="card" id="manualCard" style="display:none">
      <div class="card-title">Manual Light Control</div>
      <div class="light-display">
        <span class="bulb" id="mBulb">💡</span>
        <div class="light-state" id="mLightState">STANDBY</div>
      </div>
      <div class="toggle-row" style="margin-top:12px">
        <span>Street Light</span>
        <label class="tog">
          <input type="checkbox" id="lightToggle" onchange="toggleLight()">
          <div class="tog-track"></div>
          <div class="tog-thumb"></div>
        </label>
      </div>
    </div>

    <div class="card" id="autoLightCard">
      <div class="card-title">Auto Light Status</div>
      <div class="light-display">
        <span class="bulb" id="autoBulb">💡</span>
        <div class="light-state" id="autoLightState">STANDBY</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Detection Alerts</div>
      <div class="alert-list">
        <div class="alert-item" id="alertFire">
          <span class="a-icon">🔥</span> Fire Detected <span class="a-dot"></span>
        </div>
        <div class="alert-item" id="alertCrowd">
          <span class="a-icon">👥</span> Crowd Alert <span class="a-dot"></span>
        </div>
        <div class="alert-item" id="alertAccident">
          <span class="a-icon">💥</span> Accident Detected <span class="a-dot"></span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">System Log</div>
      <div class="log-box" id="logBox"></div>
    </div>
  </div>
</div>

<script>
// CLOCK
setInterval(()=>{
  document.getElementById('sysTime').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});
},1000);

// IP
fetch('/status').then(r=>r.json()).then(d=>{
  document.getElementById('sysIP').textContent='MODE:'+d.mode;
}).catch(()=>{});

// LOG
function addLog(msg,type=''){
  const b=document.getElementById('logBox');
  const e=document.createElement('div');
  e.className='log-entry '+(type||'');
  e.textContent='['+new Date().toLocaleTimeString('en-GB',{hour12:false})+'] '+msg;
  b.insertBefore(e,b.firstChild);
  while(b.children.length>80)b.removeChild(b.lastChild);
}

// MODE
function setMode(m){
  fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  const isM=m==='Manual';
  document.getElementById('manualCard').style.display=isM?'':'none';
  document.getElementById('autoLightCard').style.display=isM?'none':'';
  document.getElementById('btnAuto').classList.toggle('active',!isM);
  document.getElementById('btnManual').classList.toggle('active',isM);
  document.getElementById('valMode').textContent=m;
  addLog('Mode → '+m,'info');
}

// LIGHT TOGGLE
function toggleLight(){
  const on=document.getElementById('lightToggle').checked;
  const s=on?'ON':'OFF';
  fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({light:s})});
  setBulb('mBulb','mLightState',on);
  addLog('Manual light → '+s,on?'warn':'');
}

function setBulb(bid,sid,on){
  const b=document.getElementById(bid),s=document.getElementById(sid);
  b.classList.toggle('on',on);
  s.textContent=on?'ACTIVE':'STANDBY';
  s.className='light-state '+(on?'on':'');
}

// ALERTS
let prevAlerts={fire:false,crowd:false,accident:false};
function setAlert(id,active,label){
  const el=document.getElementById(id);
  const was=el.classList.contains('triggered');
  el.classList.toggle('triggered',active);
  if(active&&!was) addLog('⚠ '+label+' DETECTED','crit');
  if(!active&&was) addLog('✓ '+label+' cleared','info');
}

// STATUS POLL
let prevEM=false,prevP=-1;
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('valPersons').textContent=d.persons||0;
    document.getElementById('valVehicles').textContent=d.vehicles||0;
    document.getElementById('valTime').textContent=d.night?'Night':'Day';
    document.getElementById('fpsBadge').textContent=(d.fps||0)+' FPS';
    document.getElementById('mPersons').classList.toggle('active',(d.persons||0)>0);
    document.getElementById('mVehicles').classList.toggle('active',(d.vehicles||0)>0);
    setBulb('autoBulb','autoLightState',d.light==='ON');
    if((d.persons||0)!==prevP&&prevP!==-1&&(d.persons||0)>(prevP||0))
      addLog((d.persons||0)+' person(s) in frame');
    prevP=d.persons||0;
    const em=d.emergency;
    document.getElementById('eOverlay').style.display=em?'block':'none';
    document.getElementById('eBanner').style.display=em?'block':'none';
    if(em&&!prevEM) addLog('🚨 EMERGENCY – ESP32 light forced ON','crit');
    if(!em&&prevEM) addLog('✓ Emergency cleared','info');
    prevEM=em;
  }).catch(()=>{});
},1000);

// ALERT POLL
setInterval(()=>{
  fetch('/alerts').then(r=>r.json()).then(d=>{
    setAlert('alertFire',    d.fire,    'Fire');
    setAlert('alertCrowd',   d.crowd,   'Crowd');
    setAlert('alertAccident',d.accident,'Accident');
    document.getElementById('mPersons').classList.toggle('alert',d.crowd);
    document.getElementById('mVehicles').classList.toggle('alert',d.accident);
  }).catch(()=>{});
},800);

addLog('System initialised','info');
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════
#  START THREADS
# ═══════════════════════════════════════════════════════════
threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=detection_thread,     daemon=True).start()
threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
    daemon=True
).start()

# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════
while True:
    df  = display_frame
    dbs = draw_boxes
    p   = persons
    v   = vehicles
    vb  = boxes

    if df is None:
        time.sleep(0.01)
        continue

    fps_times.append(time.time())
    frame_show = df.copy()

    # Draw boxes
    for (x1,y1,x2,y2,color,label) in dbs:
        cv2.rectangle(frame_show,(x1,y1),(x2,y2),color,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        cv2.rectangle(frame_show,(x1,y1-th-8),(x1+tw+6,y1),color,-1)
        cv2.putText(frame_show,label,(x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),1,cv2.LINE_AA)

    night     = is_night()
    fire      = _fire_state
    crowd     = _crowd_state
    accident  = _accident_state
    emergency = emergency_sent   # already managed in detection_thread

    cur = time.time()
    if p > 0 or v > 0:
        last_detection_time = cur

    # Light logic
    if mode_web == "Auto":
        if night and (cur - last_detection_time < DETECTION_HOLD):
            send_light("ON");  light_status = "ON"
        else:
            send_light("OFF"); light_status = "OFF"
    else:
        on = (light_web == "ON")
        send_light("ON" if on else "OFF")
        light_status = "ON (Manual)" if on else "OFF (Manual)"

    # HUD bar
    overlay = frame_show.copy()
    cv2.rectangle(overlay,(0,0),(frame_show.shape[1],34),(0,0,0),-1)
    cv2.addWeighted(overlay,0.55,frame_show,0.45,0,frame_show)
    tags = []
    if fire:     tags.append("FIRE")
    if crowd:    tags.append("CROWD")
    if accident: tags.append("ACCIDENT")
    hud = f"{'NIGHT' if night else 'DAY'} | LIGHT:{light_status} | P:{p} V:{v}"
    if tags: hud += "  !! " + "  ".join(tags)
    cv2.putText(frame_show, hud,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,230,255),1,cv2.LINE_AA)

    if emergency:
        h,w = frame_show.shape[:2]
        cv2.rectangle(frame_show,(0,0),(w,h),(0,0,200),4)

    latest_frame = frame_show   # atomic write