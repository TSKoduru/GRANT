// ESP32-S3 bridge firmware — skeleton.
//
// Build with Arduino IDE + "ESP32" board package (v3.x).
// Libraries: WiFi, ESPAsyncWebServer, AsyncTCP, HTTPClient, Preferences, ESPmDNS.
//
// TODO flags mark everything that needs to be filled in before first boot.

#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <ESPmDNS.h>

static const char* MDNS_HOSTNAME = "grant";
static const uint16_t HTTP_PORT = 80;

AsyncWebServer server(HTTP_PORT);
Preferences prefs;

String AI_PC_URL;          // e.g. "http://192.168.4.2:8000"
String ONSHAPE_API_KEY;
String ONSHAPE_API_SECRET;
String ONSHAPE_DOCUMENT_ID;

// ─── Setup ────────────────────────────────────────────────────────────

void loadConfig() {
  prefs.begin("grant", true);
  AI_PC_URL           = prefs.getString("AI_PC_URL", "http://192.168.4.2:8000");
  ONSHAPE_API_KEY     = prefs.getString("ONSHAPE_API_KEY", "");
  ONSHAPE_API_SECRET  = prefs.getString("ONSHAPE_API_SECRET", "");
  ONSHAPE_DOCUMENT_ID = prefs.getString("ONSHAPE_DOCUMENT_ID", "");
  prefs.end();
}

void startAccessPoint() {
  prefs.begin("grant", true);
  String ssid = prefs.getString("SSID", "GRANT-Scanner");
  String psk  = prefs.getString("PSK",  "grant1234");
  prefs.end();
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid.c_str(), psk.c_str());
  Serial.printf("AP up: SSID=%s IP=%s\n", ssid.c_str(), WiFi.softAPIP().toString().c_str());
}

void startMDNS() {
  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", HTTP_PORT);
    Serial.printf("mDNS: http://%s.local/\n", MDNS_HOSTNAME);
  }
}

// ─── Routes ───────────────────────────────────────────────────────────

void handleRoot(AsyncWebServerRequest* req) {
  // Dashboard lives on the AI PC; redirect.
  String dest = AI_PC_URL + "/";
  req->redirect(dest.c_str());
}

void handleScanProxy(AsyncWebServerRequest* req) {
  // Reverse-proxy any /scan/* request to the AI PC.
  // TODO: async proxy — the naive synchronous HTTPClient blocks the server loop.
  //       For the hackathon this is acceptable; for production switch to AsyncClient.
  HTTPClient http;
  String dest = AI_PC_URL + req->url();
  http.begin(dest);
  int code = http.GET();
  String body = http.getString();
  http.end();
  req->send(code > 0 ? code : 502, http.header("Content-Type"), body);
}

void handleOnshapeUpload(AsyncWebServerRequest* req) {
  // TODO: Receive multipart-encoded mesh, then POST to Onshape REST API:
  //   1. Call /documents/{did}/w/{wid}/blobelements with the .ply bytes
  //   2. Import geometry as a part studio element
  //   3. Respond with JSON { "document_url": "..." }
  req->send(501, "application/json", "{\"error\":\"onshape upload not yet implemented\"}");
}

// ─── Entry ────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  loadConfig();
  startAccessPoint();
  startMDNS();

  server.on("/",         HTTP_GET,  handleRoot);
  server.on("/scan/.*",  HTTP_ANY,  handleScanProxy);
  server.on("/onshape",  HTTP_POST, handleOnshapeUpload);
  server.begin();
}

void loop() {
  // All work is async — nothing to do here.
}
