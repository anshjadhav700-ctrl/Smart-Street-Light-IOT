# Smart Solar Street Light (Camera Detection + ESP32 + Firebase)

A solar-powered street light that turns on **only when people or vehicles are present at night**, and switches to **emergency mode** (light forced ON and a dashboard alert) when it detects **fire, a crowd, or a possible accident**.

- **PC / laptop (Python):** reads a webcam, detects people and vehicles with YOLOv8, and serves a live web dashboard.
- **ESP32:** reads a PIR motion sensor, listens to commands from Firebase, and switches a relay.
- **Firebase Realtime Database:** passes the `light` and `emergency` commands from the PC to the ESP32.

---

## Features

| Feature | How it works |
|---|---|
| Person / vehicle detection | YOLOv8n detects `person`, `car`, `motorcycle`, `truck`, `bus` |
| Night-only smart lighting | Light turns ON if it is night **and** something was seen in the last 5 s |
| PIR backup | The ESP32 PIR sensor turns the light ON locally, even if the PC is off |
| Fire alert | HSV colour-based flame detection over several consecutive frames |
| Crowd alert | Triggers when 5 or more people are in frame |
| Accident alert | Triggers when two vehicle bounding boxes overlap heavily |
| Emergency override | Any alert sets `emergency=true` in Firebase, and the ESP32 forces the light ON |
| Live dashboard | Video stream, counters, alerts, log, Auto / Manual mode, remote light toggle |
| Remote access | Optional public URL through ngrok |

---

## System architecture

```
   Webcam ──► PC running street_light_server.py ──► Firebase Realtime DB
              (YOLOv8 + Flask dashboard)              /light       "ON" | "OFF"
                     │                                /emergency   true | false
                     ▼                                      │
              Browser dashboard                             ▼ (polled every 1.5 s)
              (local or via ngrok)                    ESP32 ◄── PIR sensor
                                                         │
                                                         ▼
                                                   Relay ► Street light (LED)
```

---

## Hardware required

| # | Component | Notes |
|---|---|---|
| 1 | ESP32 DevKit (30/38-pin) | Any ESP32 dev board with WiFi |
| 2 | 1-channel relay module | **Active-LOW** type (the firmware assumes LOW = ON) |
| 3 | PIR motion sensor (HC-SR501) | |
| 4 | LED street-light / LED strip | Use **low-voltage (12 V) DC**, see safety note |
| 5 | Solar panel + charge controller + battery | e.g. 12 V panel, PWM/MPPT controller, 12 V battery |
| 6 | Buck converter (12 V to 5 V) or USB supply | To power the ESP32 from the battery |
| 7 | USB webcam | Connected to the PC |
| 8 | PC / laptop | Python 3.9 - 3.12. An NVIDIA GPU is optional but makes detection faster |
| 9 | Jumper wires, breadboard | |

> **Safety:** Do **not** switch 230 V mains with a hobby relay unless you know how to do it safely. A 12 V DC LED load powered from the battery is strongly recommended.

---

## Wiring

| ESP32 pin | Connects to |
|---|---|
| `GPIO 2` | Relay module **IN** |
| `GPIO 4` | PIR module **OUT** |
| `VIN / 5V` | Relay **VCC** and PIR **VCC** |
| `GND` | Relay **GND** and PIR **GND** |

Relay output side (typical solar wiring):

```
Battery (+) ──► Relay COM
Relay NO    ──► LED (+)
LED (-)     ──► Battery (-)
```

Solar panel to charge controller to battery is wired according to your charge controller's manual.

> Pins are defined at the top of `firmware/smart_street_light/smart_street_light.ino` (`LIGHT_PIN`, `PIR_PIN`) if you want to change them.
> The PIR sensor needs about 30-60 seconds to warm up after power-on.

---

## Circuit diagram

![Circuit diagram](docs/circuit_diagram.png)

Vector version: [docs/circuit_diagram.svg](docs/circuit_diagram.svg)

---

## Repository structure

```
smart-solar-street-light/
├── README.md
├── .gitignore
├── docs/
│   ├── circuit_diagram.png
│   └── circuit_diagram.svg
├── software/
│   ├── street_light_server.py   # detection + Flask dashboard + Firebase sender
│   └── requirements.txt
└── firmware/
    └── smart_street_light/
        └── smart_street_light.ino   # ESP32 firmware
```

---

## Step-by-step build guide

### Step 1 - Get the code

```bash
git clone https://github.com/<your-username>/smart-solar-street-light.git
cd smart-solar-street-light
```

### Step 2 - Set up Firebase (free)

1. Go to <https://console.firebase.google.com> and click **Add project** (Analytics is not needed).
2. In the left menu: **Build > Realtime Database > Create Database**. Pick a location and start in **test mode**.
3. Open the **Rules** tab, paste the following for development, then **Publish**:
   ```json
   {
     "rules": {
       ".read": true,
       ".write": true
     }
   }
   ```
4. Open the **Data** tab. Hover on the root node, click **+**, and add two keys:
   - `light` = `OFF` (string)
   - `emergency` = `false` (boolean)
5. Copy your database URL from the top of the Data tab. It looks like
   `https://your-project-default-rtdb.firebaseio.com` (**no trailing slash**).
6. Quick test: open `https://your-project-default-rtdb.firebaseio.com/light.json` in a browser. You should see `"OFF"`.

> **Security note:** these open rules let *anyone who knows your URL* read and change your data. That is fine for a demo; for anything real, add authentication (Firebase Auth / database secret) and tighten the rules. Do not share your database URL publicly.

### Step 3 - Flash the ESP32

1. Install the **Arduino IDE** (<https://www.arduino.cc/en/software>).
2. Add ESP32 support: *File > Preferences > Additional Boards Manager URLs*, add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`,
   then *Tools > Board > Boards Manager*, search **esp32** and install **"esp32 by Espressif Systems"**.
3. Open `firmware/smart_street_light/smart_street_light.ino`.
4. Near the top of the sketch, find the **"FILL THESE IN BEFORE UPLOADING"** section and enter your own details between the quotes:
   ```cpp
   const char* SSID     = "";   // your WiFi name (2.4 GHz)
   const char* PASSWORD = "";   // your WiFi password
   const String FIREBASE_URL = "";   // e.g. https://your-project-default-rtdb.firebaseio.com
   ```
5. *Tools > Board* > **ESP32 Dev Module**; select the correct **Port**.
6. Click **Upload**. (If it hangs at "Connecting...", hold the **BOOT** button on the ESP32 until the upload starts.)
7. Open *Tools > Serial Monitor* at **115200 baud**. You should see `[WiFi] Connected  IP: ...`.

> The ESP32 only works on a **2.4 GHz** WiFi network (not 5 GHz).

### Step 4 - Install the Python software

Install **Python 3.9 - 3.12**, then from the repo folder:

```bash
cd software
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the `yolov8n.pt` model (about 6 MB) automatically.

### Step 5 - Add your details and run

1. Open `software/street_light_server.py` in any text editor.
2. At the top, find the **"FILL THESE IN BEFORE RUNNING"** section and enter your own values between the quotes:
   ```python
   FIREBASE_URL      = ""   # your Firebase Realtime Database URL, no trailing slash
   NGROK_AUTH_TOKEN  = ""   # optional: your ngrok authtoken (leave blank for local network only)
   ```
   - `FIREBASE_URL` is the same URL you used in the ESP32 sketch (Step 2.5).
   - `NGROK_AUTH_TOKEN` is optional. Create a free account at <https://ngrok.com> and copy your **authtoken** from its dashboard. If you leave it blank, the dashboard only works on your local network.
3. Run the program:
   ```bash
   python street_light_server.py
   ```
4. Open the dashboard at **http://localhost:5000**. If you entered an ngrok token, the public URL is printed in the terminal on startup.

> **Keep your details private:** the WiFi password, Firebase URL and ngrok token are personal. If you publish your own copy of this project (for example on GitHub), set these fields back to empty quotes `""` before you upload.

### Step 6 - Wire everything up and test

Follow the **Wiring** section and the **Circuit diagram** above, power the ESP32, and wait about a minute for the PIR to warm up.

| Test | What to do | Expected result |
|---|---|---|
| Camera | Open the dashboard | Live video with FPS counter |
| Detection | Stand in front of the camera | Green box around you, **Persons** counter goes up |
| Manual mode | Click **Manual**, then toggle the light | Relay clicks, `light` in Firebase changes, ESP32 prints `[FB] Command -> ON` |
| Auto mode (night) | Switch to **Auto**, walk in front of the camera | Light turns ON and stays ON for 5 s after you leave |
| PIR | Wave your hand in front of the PIR | Light turns ON immediately (local, no PC needed) |
| Emergency | Show 5 or more people / a video of fire | Red banner appears, `emergency=true`, light forced ON |

> **Testing during the day:** Auto mode only switches the light at night (18:00 - 06:00 by default). To test in daylight, set `NIGHT_START = 0` in `street_light_server.py`, or use Manual mode.

---

## Configuration

Edit the constants at the top of `software/street_light_server.py`:

| Setting | Default | Meaning |
|---|---|---|
| `NIGHT_START` / `NIGHT_END` | `18` / `6` | Hours (24 h clock) that count as "night" |
| `DETECTION_HOLD` | `5` | Seconds the light stays ON after the last detection |
| `CROWD_THRESHOLD` | `5` | Number of people that triggers a crowd alert |
| `FIRE_FRAME_THRESH` | `4` | Consecutive "fire-like" frames needed |
| `FIRE_PIXEL_THRESH` | `8000` | Number of fire-coloured pixels needed per frame |
| `ACCIDENT_OVERLAP` | `4000` | Overlap area (px²) between two vehicles to flag an accident |
| `CONF_THRESHOLD` | `0.45` | Minimum detection confidence |

ESP32 settings (top of the `.ino`): `PIR_HOLD_MS`, `FIREBASE_POLL_MS`, `WIFI_RETRY_MS`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: FIREBASE_URL is empty` | Open `street_light_server.py` and fill in `FIREBASE_URL` at the top (Step 5) |
| Video is black / no camera | Change `cv2.VideoCapture(0)` to `1` (or `2`) for an external webcam; close other apps using the camera |
| Very low FPS | Normal on CPU-only PCs. Use an NVIDIA GPU with CUDA PyTorch, or lower the camera resolution |
| ESP32 can't connect to WiFi | Must be 2.4 GHz; check `SSID` and `PASSWORD` at the top of the sketch |
| Serial shows `[CONFIG] Fill in SSID, PASSWORD and FIREBASE_URL` | You left the fields at the top of the sketch empty (Step 3.4) |
| ESP32 prints `[HTTP] Error -1` | No internet, wrong Firebase URL, or Firebase rules don't allow access |
| `[FB] Unknown payload: null` | The `light` key doesn't exist in Firebase yet, add it (Step 2.4) |
| Relay works backwards | Your relay is active-HIGH. Swap `LOW`/`HIGH` in the `LIGHT_ON()` / `LIGHT_OFF()` macros |
| Light turns on by itself in daytime | That is the PIR, which works day and night (see limitations) |
| ngrok error on startup | `NGROK_AUTH_TOKEN` is wrong, or an old ngrok session is still running. Leave it blank to skip ngrok |

---

## Known limitations

- **Fire detection is colour-based.** Orange objects, sunsets or bright lamps can cause false alarms.
- **Accident detection is a simple box-overlap check.** Traffic jams or vehicles close together can cause false alarms.
- The **PIR sensor ignores day/night**, so it can switch the light on in daylight. A light sensor (LDR) would fix this.
- The dashboard and `/control` endpoint have **no login**. Anyone with the ngrok URL can view the camera and control the light.
- The Firebase rules in this guide are **open** (demo only).
- The ESP32 uses **HTTP polling** every 1.5 s, so there is a small delay between detection and the light switching.
- Detection runs on a 320x320 resized frame, so small or distant objects may be missed.

## Future improvements

- Add an LDR sensor for real day/night detection
- Add password protection to the dashboard and Firebase Auth
- Use a dedicated fire/smoke and accident detection model
- Use an ESP32-CAM to remove the need for a PC webcam
- Solar voltage and battery monitoring on the dashboard
- Dimming via PWM instead of just ON/OFF
