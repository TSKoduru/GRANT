# quick_ollama_test.py
import ollama
import json

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": "respond with only the JSON: {\"ok\": true}"}],
    format="json",
)
print("Response:", response["message"]["content"])
parsed = json.loads(response["message"]["content"])
print("Parsed:", parsed)