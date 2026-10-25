/*  ═══════════════════════════════════════════════════════════════
    NeuralStreet – Smart Solar Street Light  |  ESP32 Firmware
    Firebase Realtime DB  +  PIR Motion  +  Emergency Override
    ═══════════════════════════════════════════════════════════════ */

#include <WiFi.h>
#include <HTTPClient.h>

// ── FILL THESE IN BEFORE UPLOADING ───────────────────────────────
const char* SSID     = "";   // TODO: your WiFi name (must be a 2.4 GHz network)
const char* PASSWORD = "";   // TODO: your WiFi password

// TODO: your Firebase Realtime Database URL, no trailing slash
// e.g. https://your-project-default-rtdb.firebaseio.com
const String FIREBASE_URL = "";

const String FB_LIGHT     = FIREBASE_URL + "/light.json";
const String FB_EMERGENCY = FIREBASE_URL + "/emergency.json";

// ── PINS ─────────────────────────────────────────────────────────
#define LIGHT_PIN   2    // Relay IN  (active-LOW relay)
#define PIR_PIN     4    // PIR OUT

// ── TIMING ───────────────────────────────────────────────────────
#define PIR_HOLD_MS       5000   // light stays ON this long after motion
#define FIREBASE_POLL_MS  1500   // how often we poll Firebase (ms)
#define WIFI_RETRY_MS     5000   // reconnect attempt interval

// ── HELPERS (inverted relay: LOW = ON, HIGH = OFF) ────────────────
#define LIGHT_ON()   digitalWrite(LIGHT_PIN, LOW)
#define LIGHT_OFF()  digitalWrite(LIGHT_PIN, HIGH)

// ── STATE ─────────────────────────────────────────────────────────
unsigned long lastMotionTime  = 0;
unsigned long lastFirebasePoll = 0;
unsigned long lastWifiRetry   = 0;

bool motionActive    = false;
bool emergencyActive = false;
int  lastPIR         = LOW;

// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  if (strlen(SSID) == 0 || FIREBASE_URL.length() == 0) {
    Serial.println("[CONFIG] Fill in SSID, PASSWORD and FIREBASE_URL at the top of the sketch, then re-upload.");
  }
  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(PIR_PIN,   INPUT);
  LIGHT_OFF();   // safe default

  connectWiFi();
}

// ─────────────────────────────────────────────────────────────────
void loop() {

  // ── 1. WiFi watchdog ─────────────────────────────────────────
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWifiRetry > WIFI_RETRY_MS) {
      Serial.println("[WiFi] Lost – reconnecting…");
      WiFi.disconnect();
      WiFi.begin(SSID, PASSWORD);
      lastWifiRetry = millis();
    }
    delay(100);
    return;   // skip logic until reconnected
  }

  // ── 2. PIR motion (edge-triggered, with hold timer) ──────────
  int pirNow = digitalRead(PIR_PIN);

  if (pirNow == HIGH && lastPIR == LOW) {
    Serial.println("[PIR] Motion detected → Light ON");
    LIGHT_ON();
    motionActive  = true;
    lastMotionTime = millis();
  }
  lastPIR = pirNow;

  // Auto-off after hold period (only if not emergency)
  if (motionActive && !emergencyActive &&
      millis() - lastMotionTime > PIR_HOLD_MS) {
    motionActive = false;
    Serial.println("[PIR] Hold expired");
    // Don't force OFF here — let Firebase decide next poll
  }

  // ── 3. Firebase poll ─────────────────────────────────────────
  if (millis() - lastFirebasePoll > FIREBASE_POLL_MS) {
    lastFirebasePoll = millis();

    // Read emergency flag
    String emStr = fbGet(FB_EMERGENCY);
    emergencyActive = (emStr == "true" || emStr == "1");

    // Emergency overrides everything → light ON
    if (emergencyActive) {
      LIGHT_ON();
      Serial.println("[FB] EMERGENCY → Light FORCED ON");
    } else {
      // Only follow light command when no active PIR motion
      if (!motionActive) {
        String lightCmd = fbGet(FB_LIGHT);
        if (lightCmd == "\"ON\"") {
          LIGHT_ON();
          Serial.println("[FB] Command → ON");
        } else if (lightCmd == "\"OFF\"") {
          LIGHT_OFF();
          Serial.println("[FB] Command → OFF");
        } else {
          Serial.println("[FB] Unknown payload: " + lightCmd);
        }
      } else {
        Serial.println("[FB] Motion active – skipping FB light cmd");
      }
    }
  }

  delay(50);   // tight loop, non-blocking
}

// ─────────────────────────────────────────────────────────────────
String fbGet(const String& url) {
  HTTPClient http;
  http.begin(url);
  http.setTimeout(1000);   // 1 s timeout – don't stall the loop
  int code = http.GET();
  String payload = "";
  if (code > 0) {
    payload = http.getString();
    payload.trim();
  } else {
    Serial.println("[HTTP] Error " + String(code) + " on " + url);
  }
  http.end();
  return payload;
}

// ─────────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.print("[WiFi] Connecting to ");
  Serial.print(SSID);
  WiFi.begin(SSID, PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500);
    Serial.print(".");
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected  IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[WiFi] Failed – will retry in loop");
  }
}
