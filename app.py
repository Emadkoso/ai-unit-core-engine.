from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import json
import os
import requests
from typing import Dict, List, Optional

app = FastAPI(title="AI-Unit Jury Core", version="4.1")

# دالة تحليل التعقيد (محرك التوجيه الذكي)
def _analyze_prompt_complexity(text: str) -> float:
    score = 1.0
    text_lower = text.lower()
    tech_keywords = ["backtest", "optimize", "strategy", "quantum", "saas", "api", "cerebro", "python", "code"]
    for kw in tech_keywords:
        if kw in text_lower:
            score += 4.5
            break
    if len(text) > 100: score += 2.0
    return min(score, 10.0)

# المحلف الخبير (الآن يعمل كمحلف فقط)
def _call_jury_model(text: str, model_type: str) -> Optional[Dict[str, float]]:
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # تعليمات صارمة للمحلف: "لا تجب على السؤال، فقط قيمه"
    system_instruction = (
        "You are an expert AI Jury. Your ONLY role is to evaluate the provided text. "
        "DO NOT answer the user's question or interact with the content. "
        "Analyze the text for Accuracy, Clarity, Creativity, and Conciseness. "
        "Output ONLY a raw JSON object with these 4 keys (values 0.0 to 10.0). No text outside JSON."
    )
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Evaluate this text: '{text}'"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return json.loads(response.json()['choices'][0]['message']['content'])
    except:
        return {"accuracy": 5.0, "clarity": 5.0, "creativity": 5.0, "conciseness": 5.0}
    return None

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        
        start_time = time.time()
        
        # التوجيه الذكي
        complexity = _analyze_prompt_complexity(user_text)
        model_label = "🧠 المحلف الخبير (Heavy)" if complexity >= 5.0 else "🤖 المحلف السريع (Lite)"
        
        # استدعاء التقييم فقط
        scores = _call_jury_model(user_text, model_label)
        
        execution_time = time.time() - start_time
        avg_score = sum(scores.values()) / len(scores)
        
        # صياغة التقرير (بدون إجابة، فقط تقييم)
        reply = (
            f"⚖️ **تقرير هيئة المحلفين:**\n\n"
            f"🎯 التقييم الإجمالي: {round(avg_score, 2)}/10\n"
            f"🤖 المحلف المستخدم: {model_label}\n"
            f"📊 التحليل:\n"
            f"  ▫️ الدقة: {scores.get('accuracy')}/10\n"
            f"  ▫️ الوضوح: {scores.get('clarity')}/10\n"
            f"  ▫️ الإبداع: {scores.get('creativity')}/10\n"
            f"  ▫️ الإيجاز: {scores.get('conciseness')}/10\n\n"
            f"⏱️ زمن المعالجة: {round(execution_time, 3)} ثانية"
        )
        
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                      json={"chat_id": chat_id, "text": reply, "parse_mode": "Markdown"})
        
    return {"status": "ok"}
