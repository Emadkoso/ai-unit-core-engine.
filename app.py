from fastapi import FastAPI, Request
import time
import json
import math
import os
import re
import requests
from typing import Dict, Tuple, Optional

app = FastAPI(title="AI-Unit Core Engine V7.0", version="7.0")

# ================== إعدادات النماذج ==================
MODELS = {
    "jury": "llama-3.3-70b-versatile",      # هيئة المحلفين العليا
    "strong": "llama-3.3-70b-versatile",    # مهام معقدة
    "fast": "llama-3.1-8b-instant",         # مهام عادية (مع تعزيز جودة)
    "ultra": "mixtral-8x7b-32768"
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _groq_call(messages: list, model: str, temperature: float = 0.7, max_tokens: int = 800, json_mode: bool = False, timeout: int = 30) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY مفقود")
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return None

# ================== 1. هيئة المحلفين العليا (تقييم المهمة) ==================
def jury_assess_task(prompt: str) -> Dict:
    assessment_prompt = """
    أنت هيئة محلفين عليا لنظام ذكاء اصطناعي.
    قم بتحليل المهمة التالية وأخرج JSON فقط:
    {
      "k": <integer 1-12>, 
      "task_type": "brief description",
      "recommended_model": "fast/strong/ultra",
      "reason": "one concise sentence",
      "quality_boost_needed": true/false
    }
    """
    full_prompt = assessment_prompt + f"\nالمهمة:\n\"\"\"\n{prompt}\n\"\"\""
    
    raw = _groq_call([{"role": "user", "content": full_prompt}], MODELS["jury"], temperature=0.1, json_mode=True, timeout=15)
    
    if raw is None:
        return {"k": 5, "task_type": "general", "recommended_model": "fast", "reason": "fallback", "quality_boost_needed": True}
    
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group(0))
    except:
        return {"k": 5, "task_type": "general", "recommended_model": "fast", "reason": "parse fallback", "quality_boost_needed": True}

# ================== 2. تنفيذ المهمة بالنموذج المناسب ==================
def execute_task(prompt: str, assessment: Dict) -> Tuple[str, float, str]:
    start = time.time()
    model_key = assessment["recommended_model"]
    model = MODELS.get(model_key, MODELS["fast"])
    
    user_prompt = prompt
    if assessment.get("quality_boost_needed", True):
        user_prompt = f"أجب بأعلى مستوى من الدقة والإبداع كأنك نموذج 70B+ متقدم جداً:\n{prompt}"
    
    response = _groq_call([{"role": "user", "content": user_prompt}], model, max_tokens=1200)
    duration = time.time() - start
    return response or "❌ فشل في التنفيذ", duration, model

# ================== 3. تقييم نهائي من المحلفين ==================
def final_jury_evaluation(response: str, duration: float, k: int, original_prompt: str) -> Dict:
    jury_prompt = f"""
    قيم الرد التالي فقط (لا تجب على السؤال الأصلي):
    أخرج JSON فقط:
    {{"accuracy": float, "clarity": float, "creativity": float, "conciseness": float}}
    قيم من 0 إلى 10.
    """
    full = jury_prompt + f"\nالرد:\n\"\"\"\n{response}\n\"\"\""
    
    raw = _groq_call([{"role": "user", "content": full}], MODELS["jury"], temperature=0.1, json_mode=True)
    
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        scores = json.loads(match.group(0))
    except:
        scores = {"accuracy": 7.0, "clarity": 8.0, "creativity": 7.5, "conciseness": 7.0}
    
    avg_score = sum(scores.values()) / 4
    w_k = round(math.e ** min(k, 12), 4)  # حد أقصى لتجنب overflow
    a_k = avg_score / 10.0
    s_k = max(0.1, min(1.0, 5.0 / (duration + 0.5)))  # سرعة
    
    ai_unit_score = round(w_k * a_k * s_k, 4)
    
    return {
        "ai_unit_score": ai_unit_score,
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "w_k": w_k,
        "a_k": round(a_k, 4),
        "s_k": round(s_k, 4)
    }

# ================== 4. إرسال رسائل تيليجرام ==================
def _send_tg(token: str, chat_id: int, text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                      timeout=10)
    except:
        pass

# ================== Webhook الرئيسي ==================
@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}
    
    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    
    _send_tg(token, chat_id, "🔍 هيئة المحلفين العليا تقيم المهمة...")
    
    # 1. تقييم المهمة
    assessment = jury_assess_task(prompt)
    
    _send_tg(token, chat_id, f"📊 تم تحديد الصعوبة: k={assessment['k']} | النموذج المقترح: {assessment['recommended_model']}")
    
    # 2. تنفيذ المهمة
    response, duration, used_model = execute_task(prompt, assessment)
    
    # 3. تقييم نهائي
    final = final_jury_evaluation(response, duration, assessment["k"], prompt)
    
    # 4. التقرير النهائي
    report = f"""
🏆 **تقرير AI-Unit V7.0**

🤖 **النموذج المستخدم**: `{used_model}`
🎯 **مستوى الصعوبة**: k={assessment['k']} (أسي)
💬 **السبب**: {assessment['reason']}

📊 **النتيجة النهائية**: `{final['ai_unit_score']} AIU`

📐 **المعادلة**:
  e^k × A_k × S_k
  {final['w_k']} × {final['a_k']} × {final['s_k']}

📋 **تقييم المحلفين**:
• الدقة: {final['scores'].get('accuracy', '?')}/10
• الوضوح: {final['scores'].get('clarity', '?')}/10
• الإبداع: {final['scores'].get('creativity', '?')}/10
• الإيجاز: {final['scores'].get('conciseness', '?')}/10

⏱️ **زمن التنفيذ**: {duration:.3f} ثانية
"""
    _send_tg(token, chat_id, report)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "operational", "version": "7.0", "message": "Jury Panel + Smart Router Active"}
