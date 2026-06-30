from fastapi import FastAPI, Request
from pydantic import BaseModel
import statistics
import time
import json
import os
import requests
from typing import Dict, List, Optional

app = FastAPI(title="AI-Unit Core SaaS API", version="3.0")

CONFIG = {
    "market_leader_runtimes": {
        "1": [0.10, 0.14, 0.12], "2": [0.35, 0.42, 0.38],
        "3": [0.95, 1.15, 1.02], "4": [2.10, 2.60, 2.30],
        "5": [4.80, 6.10, 5.40]
    }
}

def _heuristic_fallback(text: str) -> Dict[str, float]:
    words = text.split()
    word_count = max(1, len(words))
    score = min(10.0, (len(set(words)) / word_count) * 8)
    return {"accuracy": score, "clarity": score, "creativity": score, "conciseness": score}

def _call_groq_judge(text: str) -> Optional[Dict[str, float]]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in Environment Variables!", flush=True)
        return None
        
    print(f"ℹ️ Attempting Groq API (Llama 3) | Key starts with: {api_key[:6]}...", flush=True)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        f"Evaluate text: '{text}'.\n"
        f"Return ONLY a raw JSON object matching this exact schema without any markdown wrapping or backticks:\n"
        f"{{\n"
        f"  \"accuracy\": 8.0,\n"
        f"  \"clarity\": 8.0,\n"
        f"  \"creativity\": 8.0,\n"
        f"  \"conciseness\": 8.0\n"
        f"}}"
    )
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            text_response = result['choices'][0]['message']['content'].strip()
            print("✅ Groq API Success!", flush=True)
            return json.loads(text_response)
        else:
            print(f"❌ Groq API Failed! Status: {response.status_code} | Body: {response.text}", flush=True)
    except Exception as e:
        print(f"❌ Groq Connection Exception: {str(e)}", flush=True)
    return None

@app.on_event("startup")
def setup_telegram_webhook():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if token and render_url:
        token_clean = token.strip()
        url_clean = render_url.strip()
        tg_base = "https://api.telegram.org/bot"
        final_webhook_url = f"{tg_base}{token_clean}/setWebhook?url={url_clean}/tg-webhook"
        requests.get(final_webhook_url)

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        
        start_time = time.time()
        # استدعاء المحلف الأول النشط (Llama 3)
        scores = _call_groq_judge(user_text)
        real_eval = True
        
        if not scores:
            scores = _heuristic_fallback(user_text)
            real_eval = False
            
        t_actual = time.time() - start_time
        avg_score = sum(scores.values()) / len(scores)
        
        eval_type = "🤖 حقيقي (المحلف 1: Llama 3)" if real_eval else "🛡️ defensive احتياطي (Local)"
        reply = f"📊 التقرير:\n🔹 نوع التقييم: {eval_type}\n🔹 التقييم: {round(avg_score, 2)}/10\n⏱️ الوقت: {round(t_actual, 3)} ثانية"
        
        tg_send_base = "https://api.telegram.org/bot"
        tg_send_url = f"{tg_send_base}{token}/sendMessage"
        requests.post(tg_send_url, json={"chat_id": chat_id, "text": reply})
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "Active"}
