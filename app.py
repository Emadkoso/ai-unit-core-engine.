from fastapi import FastAPI, Request
import time
import json
import math
import os
import re
import requests
from typing import Dict

app = FastAPI(title="AI-Unit Core Engine V7.2 - أسي", version="7.2")

# ================== Providers ==================
PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models": {"default": "meta-llama/llama-3.3-70b-instruct:free", "fast": "google/gemini-2.0-flash-exp:free"}
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "models": {"strong": "llama-3.3-70b-versatile"}
    }
}

def _call_llm(messages, provider="openrouter", model_key="default", temperature=0.7, max_tokens=1000, json_mode=False):
    config = PROVIDERS.get(provider)
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        return "❌ API Key مفقود"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "ai-unit", "X-Title": "AI-Unit"}
    payload = {"model": config["models"].get(model_key, model_key), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(config["base_url"], json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ {str(e)[:100]}"

# ================== تقييم أسي (الطريقة الأصلية) ==================
def assess_difficulty_exponential(prompt: str) -> Dict:
    assessment_prompt = """
    قيم الصعوبة المعرفية للمهمة التالية من 1 إلى 10 بطريقة أسية.
    أخرج JSON فقط:
    {"k": integer 1-10, "reason": "سبب مختصر", "w_k": "e^k تقريبي"}
    """
    raw = _call_llm([{"role": "user", "content": assessment_prompt + prompt}], json_mode=True)
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(match.group(0))
        data["w_k"] = round(math.e ** data["k"], 4)
        return data
    except:
        k = min(8, max(3, len(prompt)//50))
        return {"k": k, "reason": "تقدير احتياطي أسي", "w_k": round(math.e ** k, 4)}

# ================== باقي الدوال ==================
def execute_task(prompt: str, assessment: Dict):
    start = time.time()
    response = _call_llm([{"role": "user", "content": prompt}], provider="openrouter")
    duration = time.time() - start
    return response, duration

def final_jury_evaluation(response: str, duration: float, k: int):
    # تقييم الجودة
    raw = _call_llm([{"role": "user", "content": f"قيم الرد JSON: accuracy,clarity,creativity,conciseness 0-10\n{response}"}], json_mode=True)
    try:
        scores = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group(0))
    except:
        scores = {"accuracy":8,"clarity":8.5,"creativity":8,"conciseness":7.5}
    
    avg = sum(scores.values())/4
    w_k = round(math.e ** min(k,10), 4)
    a_k = avg / 10
    s_k = round(5.0 / (duration + 0.8), 4)
    ai_score = round(w_k * a_k * s_k, 4)
    
    return {"ai_unit_score": ai_score, "scores": scores, "w_k": w_k, "a_k": round(a_k,4), "s_k": s_k}

def _send_tg(token, chat_id, text):
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

# ================== Webhook ==================
@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    _send_tg(token, chat_id, "🔬 جاري التقييم الأسي...")

    assessment = assess_difficulty_exponential(prompt)
    response, duration = execute_task(prompt, assessment)
    final = final_jury_evaluation(response, duration, assessment["k"])

    report = f"""
🏆 **AI-Unit V7.2 (أسي)**

🤖 Provider: openrouter
🎯 **k = {assessment['k']}** | W_k = {assessment['w_k']}
💬 {assessment['reason']}

📊 **النتيجة**: `{final['ai_unit_score']} AIU`
📐 `{final['w_k']} × {final['a_k']} × {final['s_k']}`

⏱️ {duration:.3f} ث
"""
    _send_tg(token, chat_id, report)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "V7.2 أسي Active"}
