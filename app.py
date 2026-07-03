# ==============================================================
# AI-Unit Core Engine — الإصدار النهائي (V9.0)
# الميزات: تقدير k، تكاثر أسي، Multi‑Jury (3 محلفين)، تحقق بشري
# ==============================================================

from fastapi import FastAPI, Request, HTTPException
import statistics
import time
import json
import math
import os
import re
import requests
import asyncio
import hashlib
from typing import Dict, Optional, List, Any, Tuple
from pydantic import BaseModel

# ---------- الإعدادات العامة ----------
app = FastAPI(title="AI-Unit Core Engine V9.0", version="9.0")

# النموذج المختبر (الذي نقيم أداءه)
TESTED_MODEL = "llama-3.3-70b-versatile"

# تشكيلة المحلفين (3 نماذج متباينة)
JURY_MODELS = [
    {"name": "academic",   "model": "gemma-2-9b-it",      "weight": 0.4, "desc": "أكاديمي دقيق"},
    {"name": "analytical", "model": "mistral-7b-instruct", "weight": 0.3, "desc": "تحليلي منطقي"},
    {"name": "creative",   "model": "llama-3.1-8b-instant","weight": 0.3, "desc": "إبداعي مرن"},
]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# أوقات السوق القيادية (للحصول على S_k)
MARKET_LEADER_RUNTIMES: Dict[int, list] = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}

# القائمة الأم للمعايير (16 معياراً)
MASTER_CRITERIA = [
    {"name": "accuracy", "desc": "هل الجواب صحيح تماماً وخالٍ من الأخطاء الواقعية؟", "weight": "exp"},
    {"name": "clarity", "desc": "هل الجواب واضح ومباشر دون غموض؟", "weight": "linear"},
    {"name": "completeness", "desc": "هل غطى جميع جوانب السؤال دون نقص؟", "weight": "linear"},
    {"name": "coherence", "desc": "هل الأفكار مترابطة منطقياً ومتسلسلة؟", "weight": "linear"},
    {"name": "depth", "desc": "هل تجاوز السطح إلى الأسباب الجذرية والتحليل العميق؟", "weight": "exp"},
    {"name": "uniqueness", "desc": "هل يقدم وجهة نظر نادرة أو غير مكررة؟", "weight": "semi_exp"},
    {"name": "creativity", "desc": "هل يقدم حلولاً مبتكرة أو زوايا جديدة؟", "weight": "semi_exp"},
    {"name": "safety", "desc": "هل يتجنب التحيز، الكراهية، أو الضرر؟", "weight": "linear"},
    {"name": "strategy", "desc": "هل يقدم خطة عمل قابلة للتنفيذ استراتيجياً؟", "weight": "exp"},
    {"name": "predictive_power", "desc": "هل يتنبأ بالنتائج أو التحديات المستقبلية بدقة؟", "weight": "semi_exp"},
    {"name": "critical_analysis", "desc": "هل حلل الفرضيات ونقدها بناءً على أدلة؟", "weight": "semi_exp"},
    {"name": "originality", "desc": "هل الإجابة جديدة كلياً وغير موجودة في النماذج الأخرى؟", "weight": "semi_exp"},
    {"name": "fallacy_detection", "desc": "هل اكتشف المغالطات المنطقية في السؤال نفسه؟", "weight": "exp"},
    {"name": "rhetorical_beauty", "desc": "هل الصياغة لغوياً بليغة ومؤثرة؟", "weight": "linear"},
    {"name": "adaptability", "desc": "هل يتكيف الجواب مع سياقات أو جماهير مختلفة؟", "weight": "linear"},
    {"name": "generative_power", "desc": "هل يولد معرفة جديدة أم يعيد تدوير المعرفة القديمة؟", "weight": "semi_exp"},
]

# ---------- تخزين التقييمات البشرية (في الذاكرة، للإيضاح) ----------
human_feedback_store = {}  # key: prompt_hash, value: list of human scores

# ---------- دوال مساعدة ----------
def _groq_call(messages, model, temperature=0.7, max_tokens=600, json_mode=False, timeout=20) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY غير موجود")
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
        print(f"❌ Groq error: {e}")
        return None

def _difficulty_fallback(text: str) -> int:
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75:  return 2
    return 1

def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    difficulty_prompt = (
        "You are an AI difficulty assessor. Rate the COGNITIVE difficulty of the following prompt from 1 to 5.\n"
        "Scale: 1=trivial, 2=simple, 3=moderate reasoning, 4=multi-step, 5=deep expertise.\n"
        'Output ONLY JSON: {"k": <1-5>, "reason": "<one sentence>"}\n\n'
        f'Prompt: """{prompt}"""'
    )
    raw = _groq_call(
        messages=[{"role": "user", "content": difficulty_prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=80,
        json_mode=True,
        timeout=10
    )
    if raw is None:
        return _difficulty_fallback(prompt), "تقدير احتياطي (فشل استدعاء)", False
    try:
        data = json.loads(re.search(r'\{[^{}]*\}', raw).group())
        k = int(data["k"])
        if 1 <= k <= 5:
            return k, data.get("reason", "تقدير من الذكاء"), True
        raise ValueError
    except:
        return _difficulty_fallback(prompt), "تقدير احتياطي (JSON غير صالح)", False

def calculate_w_k(k: int) -> float:
    return round(math.e ** k, 4)

def calculate_s_k(k: int, t_actual: float) -> float:
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)

def get_criteria_for_k(k: int) -> List[Dict]:
    count = 2 ** (k - 1)
    return MASTER_CRITERIA[:count]

def get_criterion_weight(criterion: Dict, k: int) -> float:
    w_type = criterion["weight"]
    if w_type == "exp":
        return math.e ** k
    elif w_type == "semi_exp":
        return math.e ** (k / 2)
    else:
        return float(k)

def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    start = time.time()
    response = _groq_call(
        messages=[{"role": "user", "content": prompt}],
        model=TESTED_MODEL,
        temperature=0.7,
        max_tokens=600
    )
    return response, time.time() - start

# ---------- دالة التقييم لمحلف واحد ----------
def evaluate_single_jury(model_response: str, k: int, jury_model: str) -> Dict:
    criteria = get_criteria_for_k(k)
    criteria_names = [c["name"] for c in criteria]
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in criteria])
    jury_prompt = (
        f"You are an AI evaluator. Evaluate the response on these criteria: {criteria_names}\n"
        f"Give scores from 0.0 to 10.0.\nCRITERIA:\n{criteria_descs}\n\nRESPONSE:\n\"\"\"{model_response}\"\"\""
    )
    raw = _groq_call(
        messages=[{"role": "user", "content": jury_prompt}],
        model=jury_model,
        temperature=0.1,
        max_tokens=500,
        json_mode=True,
        timeout=20
    )
    scores = {}
    if raw:
        try:
            data = json.loads(re.search(r'\{[^{}]*\}', raw).group())
            for c in criteria:
                val = data.get(c["name"])
                scores[c["name"]] = min(max(float(val) if val is not None else 1.0, 0.0), 10.0)
        except:
            pass
    # احتياطي
    if not scores:
        for c in criteria:
            scores[c["name"]] = 1.0
    return {"scores": scores, "raw": raw}

# ---------- Multi‑Jury ----------
def multi_jury_evaluate(model_response: str, k: int) -> Dict:
    # 1. تقييم كل محلف بالتوازي
    async def evaluate_all():
        tasks = []
        for jury in JURY_MODELS:
            tasks.append(asyncio.to_thread(evaluate_single_jury, model_response, k, jury["model"]))
        results = await asyncio.gather(*tasks)
        return results
    results = asyncio.run(evaluate_all())
    # 2. دمج النتائج
    all_scores = {}   # اسم المعيار -> قائمة درجات
    for idx, result in enumerate(results):
        for name, score in result["scores"].items():
            all_scores.setdefault(name, []).append(score)
    # 3. حساب الدرجة النهائية لكل معيار = متوسط مرجح
    final_scores = {}
    for name, scores_list in all_scores.items():
        # نجمع الدرجات مع أوزان المحلفين
        weighted_sum = 0.0
        total_w = 0.0
        for i, score in enumerate(scores_list):
            w = JURY_MODELS[i]["weight"]
            weighted_sum += score * w
            total_w += w
        final_scores[name] = round(weighted_sum / total_w, 2)
    return final_scores

# ---------- دمج التحقق البشري ----------
def apply_human_correction(prompt_hash: str, raw_aiu: float) -> float:
    if prompt_hash not in human_feedback_store:
        return raw_aiu
    feedback_list = human_feedback_store[prompt_hash]
    # نأخذ متوسط آخر 5 تقييمات بشرية
    recent = feedback_list[-5:]
    avg_human = sum(recent) / len(recent)
    # نحسب الانحراف التراكمي
    deviation = abs(avg_human - raw_aiu) / raw_aiu if raw_aiu > 0 else 0
    # نطبق تصحيحاً بسيطاً: إذا كان الانحراف كبيراً (>0.3) نعدّل النتيجة
    if deviation > 0.3:
        corrected = (raw_aiu + avg_human) / 2
    else:
        corrected = raw_aiu
    return round(corrected, 4)

# ---------- خط الإنتاج الرئيسي ----------
def run_ai_unit(prompt: str) -> Dict[str, Any]:
    # 1. تقدير الصعوبة
    k, k_reason, k_is_real = assess_difficulty(prompt)
    w_k = calculate_w_k(k)
    # 2. استدعاء النموذج المختبر
    model_response, t_actual = call_tested_model(prompt)
    if model_response is None:
        return {"success": False, "error": "فشل استدعاء النموذج المختبَر"}
    # 3. Multi‑Jury
    scores = multi_jury_evaluate(model_response, k)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    # 4. حساب النتيجة الخام
    total_weighted = 0.0
    criterion_details = []
    for name, score in scores.items():
        criterion = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        weight = get_criterion_weight(criterion, k) if criterion else 1.0
        contribution = (score ** k) * weight
        total_weighted += contribution
        criterion_details.append({
            "name": name,
            "score": score,
            "weight": round(weight, 4),
            "contribution": round(contribution, 4)
        })
    s_k = calculate_s_k(k, t_actual)
    raw_aiu = total_weighted * s_k
    # 5. تطبيق التحقق البشري (إن وجد)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    final_aiu = apply_human_correction(prompt_hash, raw_aiu)
    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "jury_models": [j["model"] for j in JURY_MODELS],
        "k": k,
        "k_reason": k_reason,
        "k_assessed_by_ai": k_is_real,
        "criteria_count": len(scores),
        "criteria_names": list(scores.keys()),
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "w_k": round(w_k, 4),
        "s_k": round(s_k, 4),
        "total_weighted_sum": round(total_weighted, 4),
        "ai_unit_score": round(final_aiu, 4),   # النتيجة المعدلة بالبشر
        "raw_aiu": round(raw_aiu, 4),           # النتيجة قبل التعديل البشري
        "criterion_details": criterion_details,
        "t_actual": round(t_actual, 3),
        "model_response": model_response,
        "human_correction_applied": prompt_hash in human_feedback_store,
        "prompt_hash": prompt_hash,
    }

# ---------- نقاط النهاية API ----------
@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="حقل 'prompt' مطلوب")
    return run_ai_unit(prompt)

@app.post("/api/v1/human-feedback")
async def submit_human_feedback(request: Request):
    """
    إرسال تقييم بشري لاستخدامه في المعايرة.
    المتطلبات: prompt_hash, human_score (0-10)
    """
    body = await request.json()
    prompt_hash = body.get("prompt_hash")
    human_score = body.get("human_score")
    if not prompt_hash or human_score is None:
        raise HTTPException(status_code=400, detail="prompt_hash و human_score مطلوبان")
    try:
        human_score = float(human_score)
        if not (0 <= human_score <= 10):
            raise ValueError
    except:
        raise HTTPException(status_code=400, detail="human_score يجب أن يكون عدداً بين 0 و 10")
    if prompt_hash not in human_feedback_store:
        human_feedback_store[prompt_hash] = []
    human_feedback_store[prompt_hash].append(human_score)
    return {"status": "success", "message": f"تم إضافة التقييم البشري ({len(human_feedback_store[prompt_hash])} تقييمات)"}

@app.get("/api/v1/human-feedback/{prompt_hash}")
async def get_human_feedback(prompt_hash: str):
    scores = human_feedback_store.get(prompt_hash, [])
    return {"prompt_hash": prompt_hash, "scores": scores, "count": len(scores)}

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "9.0",
        "tested_model": TESTED_MODEL,
        "jury_models": JURY_MODELS,
        "human_feedback_entries": sum(len(v) for v in human_feedback_store.values()),
        "groq_key": "✅" if os.environ.get("GROQ_API_KEY") else "❌ مفقود",
    }

# ---------- Telegram Webhook (اختصار) ----------
def _send_tg(chat_id: int, text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except:
        pass

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}
    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    _send_tg(chat_id, "⏳ جارٍ التقييم بـ Multi‑Jury V9.0 ...")
    result = run_ai_unit(user_text)
    if not result["success"]:
        _send_tg(chat_id, f"❌ {result['error']}")
        return {"status": "ok"}
    # بناء رسالة مختصرة للتيليجرام
    scores_lines = "\n".join([f"  • {name}: {score}" for name, score in result["scores"].items()])
    reply = (
        f"🏆 *AI-Unit V9.0*\n"
        f"——————————————————\n"
        f"🎯 k={result['k']} | AIU={result['ai_unit_score']}\n"
        f"⚙️ المحلفين: {len(result['jury_models'])}\n"
        f"📊 التقييم:\n{scores_lines}\n"
        f"⏱️ {result['t_actual']} ث\n"
        f"🔍 {'✅ مع تحقق بشري' if result['human_correction_applied'] else '🤖 تقييم آلي فقط'}"
    )
    _send_tg(chat_id, reply)
    return {"status": "ok"}

# ---------- تشغيل السيرفر ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
