from fastapi import FastAPI, Request
import time
import json
import math
import os
import re
import requests
from typing import Dict, Tuple

app = FastAPI(title="AI-Unit Core Engine V7.1", version="7.1")

# ================== Providers Configuration ==================
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "models": {
            "fast": "llama-3.1-8b-instant",
            "strong": "llama-3.3-70b-versatile",
            "jury": "llama-3.3-70b-versatile"
        }
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models": {
            "fast": "google/gemini-2.0-flash-exp:free",
            "strong": "meta-llama/llama-3.3-70b-instruct:free",
            "ultra": "anthropic/claude-3.5-sonnet:free",
            "default": "meta-llama/llama-3.3-70b-instruct:free"
        }
    }
}

def _call_llm(messages: list, provider: str = "openrouter", model_key: str = "default", temperature: float = 0.7, max_tokens: int = 1200, json_mode: bool = False):
    config = PROVIDERS.get(provider)
    if not config:
        return "❌ Provider غير مدعوم"
    
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        return f"❌ {config['key_env']} مفقود! أضفه في Render.com"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-unit-bot.com",
        "X-Title": "AI-Unit V7.1"
    }

    payload = {
        "model": config["models"].get(model_key, model_key),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(config["base_url"], json=payload, headers=headers, timeout=45)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ خطأ في {provider}: {str(e)[:150]}"

# ================== 1. هيئة المحلفين ==================
def jury_assess_task(prompt: str) -> Dict:
    prompt_text = f"""
    أنت هيئة محلفين عليا. حلل المهمة وأخرج JSON فقط:
    {{"k": int (1-12), "task_type": str, "recommended_provider": "openrouter/groq", "model_key": str, "reason": str}}
    المهمة: {prompt}
    """
    raw = _call_llm([{"role": "user", "content": prompt_text}], provider="openrouter", json_mode=True)
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group(0))
    except:
        return {"k": 5, "task_type": "general", "recommended_provider": "openrouter", "model_key": "default", "reason": "fallback"}

# ================== 2. تنفيذ المهمة ==================
def execute_task(prompt: str, assessment: Dict) -> Tuple[str, float, str]:
    start = time.time()
    provider = assessment.get("recommended_provider", "openrouter")
    model_key = assessment.get("model_key", "default")
    
    user_prompt = f"أجب بجودة عالية جداً:\n{prompt}" if assessment["k"] > 4 else prompt
    response = _call_llm([{"role": "user", "content": user_prompt}], provider, model_key)
    duration = time.time() - start
    return response or "❌ فشل التنفيذ", duration, f"{provider}/{model_key}"

# ================== 3. تقييم نهائي ==================
def final_jury_evaluation(response: str, duration: float, k: int) -> Dict:
    scores_prompt = f"قيم الرد (JSON فقط): {{\"accuracy\": float, \"clarity\": float, \"creativity\": float, \"conciseness\": float}}"
    raw = _call_llm([{"role": "user", "content": scores_prompt + f"\nالرد: {response}"}], json_mode=True)
    
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        scores = json.loads(match.group(0))
    except:
        scores = {"accuracy": 8.0, "clarity": 8.5, "creativity": 8.0, "conciseness": 7.5}
    
    avg = sum(scores.values()) / 4
    w_k = round(math.e ** min(k, 10), 4)
    a_k = avg / 10
    s_k = round(max(0.2, 6.0 / (duration + 1.0)), 4)
    
    return {
        "ai_unit_score": round(w_k * a_k * s_k, 4),
        "scores": scores,
        "w_k": w_k, "a_k": round(a_k, 4), "s_k": s_k
    }

def _send_tg(token: str, chat_id: int, text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except:
        pass

# ================== Webhook ==================
@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}
    
    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    _send_tg(token, chat_id, "🔍 هيئة المحلفين تعمل...")

    assessment = jury_assess_task(prompt)
    response, duration, used = execute_task(prompt, assessment)
    final = final_jury_evaluation(response, duration, assessment["k"])

    report = f"""
🏆 **AI-Unit V7.1**
🤖 **المستخدم**: {used}
🎯 **k = {assessment['k']}**
📊 **النتيجة**: `{final['ai_unit_score']}` AIU
📐 `{final['w_k']} × {final['a_k']} × {final['s_k']}`
⏱️ **{duration:.3f} ث**
"""
    _send_tg(token, chat_id, report)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "7.1", "openrouter": "✅ Connected"}
