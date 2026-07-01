from fastapi import FastAPI, Request
import time
import math
import os
import requests

app = FastAPI(title="AI-Unit V8.1", version="8.1")

# Fallback بين OpenRouter و Groq
def _call_llm(prompt: str):
    # محاولة OpenRouter أولاً
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "ai-unit"}
            payload = {"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=25)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                print("OpenRouter Rate Limit → Fallback to Groq")
    except:
        pass

    # Fallback إلى Groq
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800}
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except:
        pass

    return "❌ جميع الـ APIs وصلت الحد الأقصى. انتظر قليلاً ثم حاول مرة أخرى."

def _send_tg(token, chat_id, text):
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    _send_tg(token, chat_id, "⚡ جاري التقييم...")

    k = max(2, min(10, len(prompt)//35 + 2))
    w = round(math.pow(2, k), 4)

    start = time.time()
    response = _call_llm(prompt)
    duration = round(time.time() - start, 3)

    s_k = round(6.0 / (duration + 0.5), 4)
    ai_score = round(w * 0.9 * s_k, 4)

    report = f"""
🏆 **AI-Unit V8.1**

🎯 k = {k} | W(2^k) = {w}
📊 **النتيجة**: `{ai_score} AIU`
📐 {w} × 0.9 × {s_k}
⏱️ {duration} ث

**الرد:**
{response[:650]}{"..." if len(response or "") > 650 else ""}
"""
    _send_tg(token, chat_id, report)
    return {"status": "ok"}
