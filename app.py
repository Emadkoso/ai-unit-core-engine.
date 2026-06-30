from fastapi import FastAPI, Request
from pydantic import BaseModel
import statistics
import time
import json
import random
import os
import requests
from typing import Dict, List, Optional

app = FastAPI(title="AI-Unit Core SaaS API", version="3.0")

# إعدادات المحرك الحركي
CONFIG = {
    "market_leader_runtimes": {
        "1": [0.10, 0.14, 0.12], "2": [0.35, 0.42, 0.38],
        "3": [0.95, 1.15, 1.02], "4": [2.10, 2.60, 2.30],
        "5": [4.80, 6.10, 5.40]
    }
}

class EvaluationRequest(BaseModel):
    model_name: str
    test_suite: List[dict]

class AIUnitEngine:
    def __init__(self):
        self.market_leader_runtimes = CONFIG["market_leader_runtimes"]

    def calculate_speed_factor(self, k: int, t_actual: float) -> float:
        runtimes = self.market_leader_runtimes.get(str(k), [1.0, 1.5, 2.0])
        t_target = statistics.median(runtimes)
        return min(t_target / (t_actual + t_target), 1.0)

def _heuristic_fallback(text: str) -> Dict[str, float]:
    words = text.split()
    word_count = max(1, len(words))
    score = min(10.0, (len(set(words)) / word_count) * 8)
    return {"accuracy": score, "clarity": score, "creativity": score, "conciseness": score}

def _call_gemini_judge(text: str) -> Optional[Dict[str, float]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in Environment Variables!", flush=True)
        return None
        
    # جلب اسم النموذج مرن؛ الافتراضي تم تحديثه لتفادي الـ 404
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")
    print(f"ℹ️ Attempting Gemini API [{model_name}] | Key starts with: {api_key[:6]}...", flush=True)
    
    p1 = "ht" + "tps://generative"
    p2 = f"language.googleapis.com/v1beta/models/{model_name}:generateContent"
    url = f"{p1}{p2}?key={api_key}"
    
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
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            text_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
            
            if text_response.startswith("```"):
                text_response = text_response.replace("```json", "").replace("```", "").strip()
                
            print("✅ Gemini API Success!", flush=True)
            return json.loads(text_response)
        else:
            print(f"❌ Gemini API Failed! Status: {response.status_code} | Body: {response.text}", flush=True)
    except Exception as e:
        print(f"❌ Gemini Connection Exception: {str(e)}", flush=True)
    return None

@app.on_event("startup")
def setup_telegram_webhook():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if token and render_url:
        token_clean = token.strip()
        url_clean = render_url.strip()
        
        tg_base = "ht" + "tps://api.te" + "legram.org/bot"
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
        scores = _call_gemini_judge(user_text)
        real_eval = True
        
        if not scores:
            scores = _heuristic_fallback(user_text)
            real_eval = False
            
        t_actual = time.time() - start_time
        avg_score = sum(scores.values()) / len(scores)
        
        eval_type = "🤖 حقيقي (Gemini)" if real_eval else "🛡️ defensive احتياطي (Local)"
        reply = f"📊 التقرير:\n🔹 نوع التقييم: {eval_type}\n🔹 التقييم: {round(avg_score, 2)}/10\n⏱️ الوقت: {round(t_actual, 3)} ثانية"
        
        tg_send_base = "ht" + "tps://api.te" + "legram.org/bot"
        tg_send_url = f"{tg_send_base}{token}/sendMessage"
        requests.post(tg_send_url, json={"chat_id": chat_id, "text": reply})
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "Active"}
