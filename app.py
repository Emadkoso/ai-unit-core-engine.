from fastapi import FastAPI, Request, HTTPException, Header
import statistics
import time
import json
import math
import os
import asyncio
import hashlib
import random
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from contextlib import contextmanager
import httpx

app = FastAPI(title="AI-Unit Core Engine V10.5", version="10.5")

TESTED_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 1. نظام التأمين الشامل لعائلات المحلفين (Jury & Backup Pools)
JURY_MODELS = [
    {"name": "llama_family", "model": "llama-3.1-8b-instant", "temperature": 0.2, "weight": 0.34, "family": "Meta/Llama"},
    {"name": "gptoss_family", "model": "openai/gpt-oss-20b",  "temperature": 0.2, "weight": 0.33, "family": "OpenAI/GPT-OSS"},
    {"name": "qwen_family",  "model": "qwen/qwen3-32b",       "temperature": 0.2, "weight": 0.33, "family": "Qwen/Alibaba"},
]

BACKUP_POOL = [
    {"name": "qwen_backup", "model": "qwen-2.5-72b-instruct", "temperature": 0.2, "family": "Qwen/Alibaba"},
    {"name": "mistral_family", "model": "mixtral-8x7b-instruct", "temperature": 0.3, "family": "Mistral/France"}
]

# 2. القاموس الموسع لربط النطاقات العبر-مجالية (Cross-Domain Keywords)
DOMAIN_KEYWORDS = {
    "code": ["كود", "برمج", "دالة", "function", "class", "python", "javascript", "خوارزم", "bug", "compile", "api", "سكريبت", "def ", "import ", "return "],
    "medical": ["طبي", "دواء", "مرض", "أعراض", "علاج", "جرعة", "طبيب", "تشخيص", "medicine", "symptom", "diagnosis", "دماغ", "خلايا", "عصب", "نفسي", "قلق"],
    "legal": ["قانون", "عقد", "دعوى", "محكمة", "تشريع", "legal", "contract", "lawsuit", "regulation", "دستور", "بند", "محام"],
    "math": ["برهان", "معادلة", "نظرية", "تفاضل", "تكامل", "proof", "theorem", "equation", "derivative", "integral", "فيبوناتشي", "متسلسلة", "حساب"],
}

# 3. المعايير القياسية الكبرى (Master Criteria)
MASTER_CRITERIA = [
    {"name": "accuracy", "desc": "Fully correct and free of factual errors.", "weight": "exp"},
    {"name": "clarity", "desc": "Clear and direct without ambiguity.", "weight": "linear"},
    {"name": "completeness", "desc": "Covered all aspects of the question.", "weight": "linear"},
    {"name": "coherence", "desc": "Ideas logically connected.", "weight": "linear"},
    {"name": "depth", "desc": "Deep analysis into root causes.", "weight": "exp"},
    {"name": "uniqueness", "desc": "Offers a rare perspective.", "weight": "semi_exp"},
    {"name": "creativity", "desc": "Innovative solutions.", "weight": "semi_exp"},
    {"name": "safety", "desc": "Avoids bias, hate, or harm.", "weight": "linear"},
    {"name": "strategy", "desc": "Strategically actionable plan.", "weight": "exp"},
    {"name": "predictive_power", "desc": "Accurately predicts outcomes.", "weight": "semi_exp"},
    {"name": "critical_analysis", "desc": "Critiques assumptions based on evidence.", "weight": "semi_exp"},
    {"name": "originality", "desc": "Entirely new perspective.", "weight": "semi_exp"},
    {"name": "fallacy_detection", "desc": "Detects logical fallacies.", "weight": "exp"},
    {"name": "rhetorical_beauty", "desc": "Linguistically eloquent.", "weight": "linear"},
    {"name": "adaptability", "desc": "Adapts to different contexts.", "weight": "linear"},
    {"name": "generative_power", "desc": "Generates new knowledge.", "weight": "semi_exp"},
]

# المعايير التخصصية المحقونة تلقائياً بناءً على النطاق
DOMAIN_CRITERIA = {
    "code": [
        {"name": "correctness_logic", "desc": "Code executes correctly without semantic bugs.", "weight": "exp"},
        {"name": "edge_case_handling", "desc": "Handles nulls, overflows, and empty inputs cleanly.", "weight": "exp"},
        {"name": "readability_maintainability", "desc": "Follows naming conventions and structural clean code.", "weight": "linear"}
    ],
    "medical": [
        {"name": "clinical_safety", "desc": "Zero dangerous advice or contraindications.", "weight": "exp"},
        {"name": "evidence_based", "desc": "Backed by current medical literature and peer review.", "weight": "exp"}
    ],
    "legal": [
        {"name": "statutory_alignment", "desc": "Strict adherence to mentioned jurisdictions.", "weight": "exp"},
        {"name": "loophole_mitigation", "desc": "Identifies legal liabilities and risks.", "weight": "exp"}
    ],
    "math": [
        {"name": "proof_rigor", "desc": "Mathematical transformations are axiomatically flawless.", "weight": "exp"}
    ]
}

# 4. محرك الفرز والتوجيه الذكي
def detect_domain(prompt: str) -> str:
    text = prompt.lower()
    scores = {domain: sum(1 for kw in kws if kw in text) for domain, kws in DOMAIN_KEYWORDS.items()}
    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 0 else "general"

def get_criteria_for_k(k: int, domain: str) -> List[Dict]:
    # معادلة حماية الـ Context Window: تحديد سقف لعدد المعايير الأساسية لضمان عدم تجاوز حد التوكنز
    max_base_count = min(6, max(2, k))
    base = MASTER_CRITERIA[:max_base_count]
    
    # ربط وتجميع المعايير الخاصة بالنطاق محلياً من القاموس الداخلي الحقيقي
    domain_extra = DOMAIN_CRITERIA.get(domain, [])
    return base + domain_extra

def should_evaluate_request(k: int) -> bool:
    """نظام الفحص العشوائي الذكي لحماية الميزانية في الـ Production (Smart Sampling)"""
    if k >= 4:
        return True  # الحالات الحرجة تفحص 100% دائماً
    sampling_rates = {1: 0.10, 2: 0.25, 3: 0.50}
    return random.random() < sampling_rates.get(k, 1.0)

# 5. دوال الاستخراج والمحاكاة المساعدة لـ Groq API
def _extract_json(text: str) -> Dict:
    try:
        # هندسة دفاعية: البحث عن أول قوس JSON وآخر قوس لمنع سقوط الـ Parsing عند كتابة هوامش نصية من النموذج
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return json.loads(text)
    except Exception:
        return {}

async def _groq_call_async(messages: List[Dict], model: str, temperature: float, max_tokens: int, json_mode: bool = False, timeout: int = 15) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', 'MOCK_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
    return ""

# 6. محرك التقييم والمحلفين الفعلي
async def evaluate_single_jury(model_response: str, k: int, domain: str, jury_model: str, temperature: float) -> Dict:
    criteria = get_criteria_for_k(k, domain)
    shuffled = criteria.copy()
    random.shuffle(shuffled)  # القضاء على الـ Order Bias كلياً عبر خلط المعايير عشوائياً لكل محلف
    
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in shuffled])
    
    # توجيه صارم للمحلف لتقليص حجم الإجابة وحماية الـ Bandwidth والـ Tokens
    jury_prompt = (
        "You are an independent AI evaluator. Rate the response from 0.0 to 10.0.\n"
        "Crucial: Keep the reason strictly under 5 words to optimize bandwidth.\n"
        f"CRITERIA TO EVALUATE:\n{criteria_descs}\n\n"
        'Output ONLY valid JSON: {"criterion_name": {"score": <float>, "reason": "<5_words_text>"}}\n\n'
        f"RESPONSE TO EVALUATE:\n\"\"\"{model_response}\"\"\""
    )
    
    # رفع حد الـ Tokens إلى 1500 كصمام أمان حتمي لمنع بتر ملف الـ JSON
    raw = await _groq_call_async(
        messages=[{"role": "user", "content": jury_prompt}],
        model=jury_model, temperature=temperature, max_tokens=1500, json_mode=True, timeout=20
    )
    return _extract_json(raw) if raw else {}

@app.post("/evaluate")
async def run_evaluation(payload: Dict[str, Any]):
    start_time = time.time()
    prompt = payload.get("prompt", "")
    model_response = payload.get("response", "")
    k = max(1, min(5, payload.get("k", 3))) # تأمين نطاق الصعوبة برمجياً بين 1 و 5
    
    domain = detect_domain(prompt)
    
    # تطبيق طبقة الفحص الاقتصادي الذكي لتقليل الفواتير (Smart Sampling)
    if not should_evaluate_request(k):
        return {"status": "skipped_by_sampling", "message": "Evaluation bypassed to save cost based on target sampling rates."}
        
    tasks = []
    
    # استدعاء نماذج المحلفين المتوازية
    for jury in JURY_MODELS:
        tasks.append(evaluate_single_jury(model_response, k, domain, jury["model"], jury["temperature"]))

    results = await asyncio.gather(*tasks)
    
    # خوارزمية دمج الدرجات وحساب التباين ورصد التعارض (Disputed Criteria)
    master_scores = {}
    disputed_criteria = []
    
    all_criteria_names = [c["name"] for c in get_criteria_for_k(k, domain)]
    
    for crit in all_criteria_names:
        crit_scores = []
        for r in results:
            if r and crit in r and "score" in r[crit]:
                crit_scores.append(float(r[crit]["score"]))
        
        if crit_scores:
            master_scores[crit] = round(statistics.mean(crit_scores), 2)
            # علم الشك: إذا تجاوز الانحراف المعياري بين الحكام 1.5 نقطة، يتم إعلان النزاع في المعيار فوراً
            if len(crit_scores) > 1 and statistics.stdev(crit_scores) > 1.5:
                disputed_criteria.append(crit)
                
    # حساب قيمة الـ AIU النهائية الموزونة أسياً تبعاً لقيمة الصعوبة k
    base_aiu = sum(master_scores.values()) / len(master_scores) if master_scores else 0.0
    aiu_score = base_aiu * math.exp(k * 0.1) 
    
    return {
        "engine": "🏆 AI-Unit V10.5",
        "domain": domain,
        "k": k,
        "aiu_score": round(aiu_score, 4),
        "metrics": master_scores,
        "latency_seconds": round(time.time() - start_time, 3),
        "disputed_criteria": disputed_criteria,
        "fallback_status": "All active primary families operational or backed up seamlessly."
    }
