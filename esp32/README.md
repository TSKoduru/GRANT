# ESP32-S3 Bridge

Role in the pipeline:

1. **WiFi access point** — the user connects their phone/laptop to the ESP32's SSID (no router needed) and opens the dashboard in a browser.
2. **HTTP proxy** — all `/scan/*` requests the dashboard makes get forwarded to the AI PC's FastAPI server over a wired or local-WiFi link.
3. **Onshape completion webhook** — when the AI PC POSTs a finished mesh to `/onshape` on the ESP32, the ESP32 relays it to the Onshape REST API using credentials stored in firmware (keeps API keys off the AI PC).

## Wiring

| ESP32 pin | To | Notes |
| --- | --- | --- |
| USB-C | AI PC | Serial + power, only used for firmware flashing |
| GPIO 1 (UART TX) | AI PC UART RX | Optional fallback link if WiFi drops |
| GPIO 2 (UART RX) | AI PC UART TX |   |
| 3V3 / GND | — | Standard |

Default link between the ESP32 and the AI PC is `http://ai-pc.local:8000` over the shared WiFi. Set the AI PC's hostname to `ai-pc` or change `AI_PC_URL` in the firmware.

## Firmware responsibilities (Arduino core)

See `firmware_sketch.ino`. Sketch:

- `setup()` — start WiFi AP, bring up mDNS as `grant.local`, start async web server.
- Routes:
  - `GET /`             — redirect to `http://ai-pc.local:8000/`
  - `GET /scan/*`       — reverse-proxy to `AI_PC_URL + path` (for dashboards on the ESP32's subnet)
  - `POST /onshape`     — accept multipart mesh from AI PC, forward to Onshape
- `loop()` — nothing; everything is async.

## Configuration

Firmware reads these from NVS (preferences) on boot — seed once via serial:

```
SSID=GRANT-Scanner
PSK=<wifi-password>
AI_PC_URL=http://192.168.4.2:8000
ONSHAPE_API_KEY=<key>
ONSHAPE_API_SECRET=<secret>
ONSHAPE_DOCUMENT_ID=<did>
```

## Testing without the ESP32

The AI PC's FastAPI server also serves the dashboard directly on port 8000. During development you can point a browser at `http://localhost:8000/` and skip the bridge entirely. The `onshape.py` module falls back to direct-to-Onshape mode when `ONSHAPE_WEBHOOK_URL` is unset.
