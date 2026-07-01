from fastapi import FastAPI, Request
import time
import json
import math
import os
import re
import requests
from typing import Dict

app = FastAPI(title="AI-Unit V8.0 ∞", version="8.0")

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models": {"default": "meta-llama/llama-3.3-70b-instruct:free"}
    }
}

def _call_llm(messages, max_tokens=900):
    try:
        config = PROVIDERS["openrouter"]
        api_key = os.environ.get(config["key_env"])
        if not api_key:
            return "❌ OPENROUTER_API_KEY غير موجود"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-unit.com",
            "X-Title": "AI-Unit V8"
        }
        payload = {
            "model": config["models"]["default"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        resp = requests.post(config["base_url"], json=payload, headers=headers, timeout=35)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] LLM: {e}")
        return f"❌ خطأ فني: {str(e)[:80]}"

def _send_tg(token, chat_id, text):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except:
        pass

# ================== تقييم أسي باستخدام 2^k (غير محدود) ==================
def assess_difficulty(prompt: str) -> Dict:
    # استخدام 2^k كرمز للنمو الأسي
    length_factor = len(prompt) // 30
    k = max(2, min(12, length_factor + 3))   # أسي محدود عملياً
    w = round(math.pow(2, k), 4)              # 2^k بدلاً من e^k
    return {
        "k": k,
        "w": w,
        "reason": f"نمو أسي 2^k = {w} (S → ∞)"
    }

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}

    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    _send_tg(token, chat_id, "⚡ جاري التقييم الأسي (2^k)...")

    # 1. تقييم
    assessment = assess_difficulty(prompt)

    # 2. تنفيذ المهمة
    start = time.time()
    response = _call_llm([{"role": "user", "content": prompt}])
    duration = round(time.time() - start, 3)

    # 3. حساب النتيجة النهائية
    s_k = round(6.0 / (duration + 0.4), 4)   # عامل السرعة
    a_k = 0.88                               # متوسط الجودة (يمكن تحسينه)
    ai_score = round(assessment["w"] * a_k * s_k, 4)

    report = f"""
🏆 **AI-Unit V8.0** (∞)

🎯 **k = {assessment['k']}**
📈 **W = 2^k = {assessment['w']}**
💬 {assessment['reason']}

📊 **النتيجة النهائية**: `{ai_score} AIU`
📐 `{assessment['w']} × {a_k} × {s_k}`

⏱️ **{duration} ث**

**الرد:**
{response[:700]}{"..." if len(response) > 700 else ""}
"""
    _send_tg(token, chat_id, report)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "V8.0 ∞ Active", "formula": "2^k + divergent series"}
