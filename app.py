from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import statistics
import time
import json
import random
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

class TestCaseInput(BaseModel):
    k: int
    t_actual: float
    raw_output: str

class EvaluationRequest(BaseModel):
    model_name: str
    test_suite: List[TestCaseInput]

class AIUnitEngine:
    def __init__(self):
        self.market_leader_runtimes = CONFIG["market_leader_runtimes"]

    def calculate_difficulty_weight(self, k: int) -> float:
        return float(k ** 2)

    def calculate_speed_factor(self, k: int, t_actual: float) -> float:
        runtimes = self.market_leader_runtimes.get(str(k))
        t_target = statistics.median(runtimes) if runtimes else float(k * 1.5)
        return min(t_target / (t_actual + t_target), 1.0)

def _heuristic_fallback(text: str) -> Dict[str, float]:
    words = text.split()
    word_count = max(1, len(words))
    unique_ratio = len(set(words)) / word_count
    punctuation_density = sum(1 for c in text if c in ".,!?") / word_count
    score = min(10.0, unique_ratio * 5 + punctuation_density * 30)
    return {k: round(score, 2) for k in ["accuracy", "clarity", "creativity", "conciseness"]}

def _call_gemini_judge(text: str) -> Optional[Dict[str, float]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"""Evaluate the following text based on 4 metrics: accuracy, clarity, creativity, conciseness (0.0 to 10.0). Respond ONLY with a valid JSON object matching this schema: {{"accuracy": 0.0, "clarity": 0.0, "creativity": 0.0, "conciseness": 0.0}} \n\nOutput to evaluate:\n"{text}" """
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text_response)
    except Exception:
        pass
    return None

@app.on_event("startup")
def setup_telegram_webhook():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if token and render_url:
        webhook_url = f"{render_url}/tg-webhook"
        requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}")

# --- نقطة استقبال رسايل تليجرام المكشوفة للفحص ---
@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN missing in Environment Variables!", flush=True)
        return {"status": "ok"}
        
    try:
        data = await request.json()
        print(f"📥 Received Payload: {data}", flush=True)
        
        if "message" in data and "text" in data["message"]:
            chat_id = data["message"]["chat"]["id"]
            user_text = data["message"]["text"]
            
            if user_text.startswith("/start"):
                reply = "👋 أهلاً بك في محرك AI-Unit المعماري الحركي!\n\nأرسل لي الآن أي نص قام بتوليده الذكاء الاصطناعي، وسأقوم بفحصه وتقييمه أونلاين عبر سيرفرك السحابي فوراً وتطبيق المعادلات الرياضية الحركية لتحديد كفاءته!"
            else:
                start_time = time.time()
                scores = _call_gemini_judge(user_text)
                real_eval = True
                if not scores:
                    scores = _heuristic_fallback(user_text)
                    real_eval = False
                
                t_actual = time.time() - start_time
                engine = AIUnitEngine()
                consensus_avg = sum(scores.values()) / len(scores)
                w_k = engine.calculate_difficulty_weight(2)
                s_k = engine.calculate_speed_factor(2, t_actual)
                final_score = w_k * s_k * (consensus_avg / 10)
                
                eval_type = "🤖 حقيقي (Gemini)" if real_eval else "🛡️ دفاعي احتياطي (Local)"
                reply = (
                    f"📊 تقرير فحص جودة النص:\n\n"
                    f"🔹 نوع التقييم: {eval_type}\n"
                    f"🔹 متوسط جودة النص: {round(consensus_avg, 2)} / 10\n"
                    f"⏱️ زمن استجابة السيرفر: {round(t_actual, 3)} ثانية\n"
                    f"📈 معامل السرعة الحركي: {round(s_k, 4)}\n"
                    f"🏅 رصيد الـ AI-Unit المحسوب: {round(final_score, 4)} نقطة\n\n"
                    f"⚙️ تم الفحص والمعالجة بالكامل عبر خادمك السحابي بنجاح."
                )
                
            # إرسال الرد ومراقبة استجابة تليجرام
            tg_url = f"https://api.telegram.org/bot{token}/sendMessage"
            tg_res = requests.post(tg_url, json={"chat_id": chat_id, "text": reply})
            print(f"📤 Telegram Sending Status: {tg_res.status_code} | Response: {tg_res.text}", flush=True)
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR INSIDE WEBHOOK: {str(e)}", flush=True)
        
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "Active", "framework": "AI-Unit Core V3.0"}

@app.post("/api/v1/evaluate")
def evaluate_api(request: EvaluationRequest):
    engine = AIUnitEngine()
    total_aiu_score = 0.0
    breakdown = {}
    for run in request.test_suite:
        k = run.k
        raw_output = run.raw_output
        t_actual = run.t_actual
        scores = _call_gemini_judge(raw_output)
        real_eval = True
        if not scores:
            scores = _heuristic_fallback(raw_output)
            real_eval = False
        consensus_avg = sum(scores.values()) / len(scores)
        w_k = engine.calculate_difficulty_weight(k)
        s_k = engine.calculate_speed_factor(k, t_actual)
        tier_score = w_k * s_k * (consensus_avg / 10)
        total_aiu_score += tier_score
        breakdown[f"Level_{k}"] = {
            "Assigned_ID": f"Model_{random.randint(10000, 99999)}",
            "Real_AI_Evaluation": real_eval,
            "Consensus_Avg_Score": round(consensus_avg, 2),
            "Difficulty_Weight_W_k": w_k,
            "Speed_Factor_S_k": round(s_k, 4),
            "Score_Earned": round(tier_score, 4)
        }
    return {"Model_Name": request.model_name, "Consolidated_AI_Unit_Score": round(total_aiu_score, 4), "Total_Tasks": len(request.test_suite), "Evaluation_Breakdown": breakdown}
    
