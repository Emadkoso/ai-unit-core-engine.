# ==============================================================
# AI-Unit Core Engine — الإصدار المُصلَّح (V9.2)
# الإصلاحات الرئيسية عن V9.1:
#   1) إزالة الـ deadlock القاتل في multi_jury_evaluate (كان يجمّد كل طلب)
#   2) توحيد معادلة الدرجات بحيث تكون قابلة للمقارنة بين مستويات k المختلفة
#   3) إصلاح التحقق البشري ليعمل فعلياً (كان بلا أي تأثير عملي)
#   4) استخدام httpx async بدل requests المتزامنة (حل جذر مشكلة الحجب)
#   5) حماية /tg-webhook بـ secret token و /api/v1/evaluate بمفتاح API اختياري
#   6) الإبقاء على مرجع لمهام asyncio.create_task لمنع garbage collection
#   7) وضع علامة is_fallback عند فشل تحليل JSON من أحد المحلفين بدل الصمت
#   8) حد أدنى معيارين حتى عند k=1
#   9) تحليل JSON أكثر متانة (raw_decode بدل regex هش)
#  10) حفظ التقييمات البشرية على القرص (JSON) لتنجو من إعادة التشغيل
# ==============================================================

from fastapi import FastAPI, Request, HTTPException, Header
import statistics
import time
import json
import math
import os
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import httpx

# ---------- الإعدادات العامة ----------
app = FastAPI(title="AI-Unit Core Engine V9.2", version="9.2")

TESTED_MODEL = "llama-3.3-70b-versatile"

# ملاحظة: تحقق من توفر هذه النماذج فعلياً على Groq قبل النشر —
# قوائم النماذج المتاحة على Groq تتغيّر بشكل متكرر.
JURY_MODELS = [
    {"name": "academic",   "model": "gemma2-9b-it",        "weight": 0.4, "desc": "أكاديمي دقيق"},
    {"name": "analytical", "model": "llama-3.1-8b-instant", "weight": 0.3, "desc": "تحليلي منطقي"},
    {"name": "creative",   "model": "llama-3.3-70b-versatile", "weight": 0.3, "desc": "إبداعي مرن"},
]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# حماية اختيارية للـ endpoints
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "").strip()
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

MARKET_LEADER_RUNTIMES: Dict[int, list] = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}

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

# ---------- تخزين التقييمات البشرية (في الذاكرة + على القرص) ----------
HUMAN_FEEDBACK_FILE = Path("/tmp/human_feedback_store.json")
human_feedback_store: Dict[str, list] = {}

def _load_human_feedback():
    global human_feedback_store
    if HUMAN_FEEDBACK_FILE.exists():
        try:
            human_feedback_store = json.loads(HUMAN_FEEDBACK_FILE.read_text())
        except Exception as e:
            print(f"⚠️ فشل تحميل ملف التقييم البشري: {e}")
            human_feedback_store = {}

def _save_human_feedback():
    try:
        HUMAN_FEEDBACK_FILE.write_text(json.dumps(human_feedback_store))
    except Exception as e:
        print(f"⚠️ فشل حفظ ملف التقييم البشري: {e}")

_load_human_feedback()

# مرجع لمهام الخلفية حتى لا يتم جمعها (garbage collected) قبل انتهائها
_background_tasks: set = set()

# ---------- عميل HTTP غير متزامن مشترك ----------
_http_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def _shutdown():
    if _http_client:
        await _http_client.aclose()

# ---------- دوال مساعدة ----------
def _extract_json(raw: str) -> Optional[dict]:
    """تحليل JSON بشكل متين باستخدام raw_decode بدل regex هش."""
    if not raw:
        return None
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(raw, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = raw.find("{", start + 1)
    return None

async def _groq_call_async(messages, model, temperature=0.7, max_tokens=600,
                            json_mode=False, timeout=20) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY غير موجود")
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = await _http_client.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Groq error ({model}): {e}")
        return None

def _difficulty_fallback(text: str) -> int:
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75:  return 2
    return 1

async def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    difficulty_prompt = (
        "You are an AI difficulty assessor. Rate the COGNITIVE difficulty of the following prompt from 1 to 5.\n"
        "Scale: 1=trivial, 2=simple, 3=moderate reasoning, 4=multi-step, 5=deep expertise.\n"
        'Output ONLY JSON: {"k": <1-5>, "reason": "<one sentence>"}\n\n'
        f'Prompt: """{prompt}"""'
    )
    raw = await _groq_call_async(
        messages=[{"role": "user", "content": difficulty_prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=80,
        json_mode=True,
        timeout=10,
    )
    if raw is None:
        return _difficulty_fallback(prompt), "تقدير احتياطي (فشل استدعاء)", False
    data = _extract_json(raw)
    if data:
        try:
            k = int(data["k"])
            if 1 <= k <= 5:
                return k, data.get("reason", "تقدير من الذكاء"), True
        except (KeyError, ValueError, TypeError):
            pass
    return _difficulty_fallback(prompt), "تقدير احتياطي (JSON غير صالح)", False

def calculate_w_k(k: int) -> float:
    return round(math.e ** k, 4)

def calculate_s_k(k: int, t_actual: float) -> float:
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)

def get_criteria_for_k(k: int) -> List[Dict]:
    # حد أدنى معيارين حتى عند k=1 لتجنّب تقييم ضيق جداً
    count = max(2, 2 ** (k - 1))
    return MASTER_CRITERIA[:count]

def get_criterion_weight(criterion: Dict, k: int) -> float:
    w_type = criterion["weight"]
    if w_type == "exp":
        return math.e ** k
    elif w_type == "semi_exp":
        return math.e ** (k / 2)
    else:
        return float(k)

async def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    start = time.time()
    response = await _groq_call_async(
        messages=[{"role": "user", "content": prompt}],
        model=TESTED_MODEL,
        temperature=0.7,
        max_tokens=600,
    )
    return response, time.time() - start

# ---------- دالة التقييم لمحلف واحد ----------
async def evaluate_single_jury(model_response: str, k: int, jury_model: str) -> Dict:
    criteria = get_criteria_for_k(k)
    criteria_names = [c["name"] for c in criteria]
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in criteria])
    jury_prompt = (
        f"You are an AI evaluator. Evaluate the response on these criteria: {criteria_names}\n"
        f"Give scores from 0.0 to 10.0.\nCRITERIA:\n{criteria_descs}\n\n"
        f'Output ONLY a JSON object mapping each criterion name to a numeric score.\n\n'
        f"RESPONSE:\n\"\"\"{model_response}\"\"\""
    )
    raw = await _groq_call_async(
        messages=[{"role": "user", "content": jury_prompt}],
        model=jury_model,
        temperature=0.1,
        max_tokens=500,
        json_mode=True,
        timeout=20,
    )
    scores = {}
    is_fallback = True
    data = _extract_json(raw) if raw else None
    if data:
        for c in criteria:
            val = data.get(c["name"])
            if val is not None:
                try:
                    scores[c["name"]] = min(max(float(val), 0.0), 10.0)
                except (ValueError, TypeError):
                    pass
        if len(scores) == len(criteria):
            is_fallback = False
    # احتياطي لأي معيار لم يُرجعه المحلف
    for c in criteria:
        scores.setdefault(c["name"], 1.0)
    return {"scores": scores, "is_fallback": is_fallback}

# ---------- Multi-Jury (تصحيح: async مباشر بلا أي event-loop nesting) ----------
async def multi_jury_evaluate(model_response: str, k: int) -> Tuple[Dict, List[str]]:
    tasks = [
        evaluate_single_jury(model_response, k, jury["model"])
        for jury in JURY_MODELS
    ]
    results = await asyncio.gather(*tasks)

    all_scores: Dict[str, list] = {}
    fallback_juries = []
    for idx, result in enumerate(results):
        if result["is_fallback"]:
            fallback_juries.append(JURY_MODELS[idx]["name"])
        for name, score in result["scores"].items():
            all_scores.setdefault(name, []).append(score)

    final_scores = {}
    for name, scores_list in all_scores.items():
        weighted_sum = 0.0
        total_w = 0.0
        for i, score in enumerate(scores_list):
            w = JURY_MODELS[i]["weight"]
            weighted_sum += score * w
            total_w += w
        final_scores[name] = round(weighted_sum / total_w, 2) if total_w else 0.0

    return final_scores, fallback_juries

# ---------- التحقق البشري (مُصلَّح: يعمل على نفس مقياس avg_score 0-10) ----------
def apply_human_correction(prompt_hash: str, avg_score: float) -> Tuple[float, bool]:
    """
    يُطبَّق التصحيح على متوسط الدرجات (مقياس 0-10) وليس على AIU الخام الضخم،
    لأن الأخير قد يكون بمراتب أكبر بكثير مما يجعل المقارنة عديمة المعنى.
    """
    if prompt_hash not in human_feedback_store or not human_feedback_store[prompt_hash]:
        return avg_score, False
    recent = human_feedback_store[prompt_hash][-5:]
    avg_human = sum(recent) / len(recent)
    deviation = abs(avg_human - avg_score) / avg_score if avg_score > 0 else 1.0
    if deviation > 0.3:
        corrected = (avg_score + avg_human) / 2
    else:
        corrected = avg_score
    return round(corrected, 4), True

# ---------- خط الإنتاج الرئيسي ----------
async def run_ai_unit(prompt: str) -> Dict[str, Any]:
    # 1. تقدير الصعوبة
    k, k_reason, k_is_real = await assess_difficulty(prompt)
    w_k = calculate_w_k(k)

    # 2. استدعاء النموذج المختبر
    model_response, t_actual = await call_tested_model(prompt)
    if model_response is None:
        return {"success": False, "error": "فشل استدعاء النموذج المختبَر"}

    # 3. Multi-Jury
    scores, fallback_juries = await multi_jury_evaluate(model_response, k)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    # 4. التصحيح البشري يُطبَّق على avg_score (مقياس 0-10) قبل التضخيم الأسي
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    corrected_avg_score, human_applied = apply_human_correction(prompt_hash, avg_score)

    # 5. حساب النتيجة النهائية — مُطبَّعة لتكون قابلة للمقارنة بين مستويات k
    #    بدل (score ** k) الذي كان يتفاقم أسياً بشكل يُبطل أي مقارنة،
    #    نحسب متوسطاً مرجحاً بمقياس 0-10 ثم نطبّق مضاعف الصعوبة w_k بشكل صريح ومحكوم.
    criterion_details = []
    total_weight_sum = 0.0
    weighted_score_sum = 0.0
    for name, score in scores.items():
        criterion = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        weight = get_criterion_weight(criterion, k) if criterion else 1.0
        # نطبّق التصحيح البشري كتعديل نسبي موحّد على كل الدرجات
        adj_score = score
        if human_applied and avg_score > 0:
            adj_score = round(score * (corrected_avg_score / avg_score), 2)
            adj_score = min(max(adj_score, 0.0), 10.0)
        weighted_score_sum += adj_score * weight
        total_weight_sum += weight
        criterion_details.append({
            "name": name,
            "score": score,
            "adjusted_score": adj_score,
            "weight": round(weight, 4),
        })

    # متوسط مرجّح بمقياس 0-10، ثم يُضرب بمضاعف الصعوبة w_k وبعامل السرعة s_k
    normalized_weighted_avg = weighted_score_sum / total_weight_sum if total_weight_sum else 0.0
    s_k = calculate_s_k(k, t_actual)
    ai_unit_score = round(normalized_weighted_avg * w_k * s_k, 4)

    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "jury_models": [j["model"] for j in JURY_MODELS],
        "jury_fallback_used": fallback_juries,  # أي محلف فشل تحليل JSON له (بدل الصمت)
        "k": k,
        "k_reason": k_reason,
        "k_assessed_by_ai": k_is_real,
        "criteria_count": len(scores),
        "criteria_names": list(scores.keys()),
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "corrected_avg_score": corrected_avg_score,
        "human_correction_applied": human_applied,
        "w_k": round(w_k, 4),
        "s_k": round(s_k, 4),
        "normalized_weighted_avg": round(normalized_weighted_avg, 4),
        "ai_unit_score": ai_unit_score,
        "criterion_details": criterion_details,
        "t_actual": round(t_actual, 3),
        "model_response": model_response,
        "prompt_hash": prompt_hash,
    }

# ---------- دوال Telegram المساعدة ----------
async def _send_tg(chat_id: int, text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    try:
        await _http_client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"❌ فشل إرسال Telegram: {e}")

async def process_and_reply(chat_id: int, user_text: str):
    """المهمة الخلفية التي تُقيّم وترد."""
    try:
        await _send_tg(chat_id, "⏳ جارٍ التقييم بـ Multi-Jury V9.2 ...")
        result = await run_ai_unit(user_text)
        if not result["success"]:
            await _send_tg(chat_id, f"❌ {result['error']}")
            return
        scores_lines = "\n".join([f"  • {name}: {score}" for name, score in result["scores"].items()])
        fb_note = ""
        if result["jury_fallback_used"]:
            fb_note = f"\n⚠️ محلفون استخدموا قيمة احتياطية: {', '.join(result['jury_fallback_used'])}"
        reply = (
            f"🏆 *AI-Unit V9.2*\n"
            f"——————————————————\n"
            f"🎯 k={result['k']} | AIU={result['ai_unit_score']}\n"
            f"⚙️ المحلفين: {len(result['jury_models'])}\n"
            f"📊 التقييم:\n{scores_lines}\n"
            f"⏱️ {result['t_actual']} ث"
            f"{fb_note}\n"
            f"🔍 {'✅ مع تحقق بشري' if result['human_correction_applied'] else '🤖 تقييم آلي فقط'}"
        )
        await _send_tg(chat_id, reply)
    except Exception as e:
        await _send_tg(chat_id, f"❌ خطأ داخلي: {str(e)[:100]}")

# ---------- نقاط النهاية API ----------
def _check_api_key(x_api_key: Optional[str]):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح")

@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="حقل 'prompt' مطلوب")
    return await run_ai_unit(prompt)

@app.post("/api/v1/human-feedback")
async def submit_human_feedback(request: Request, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    body = await request.json()
    prompt_hash = body.get("prompt_hash")
    human_score = body.get("human_score")
    if not prompt_hash or human_score is None:
        raise HTTPException(status_code=400, detail="prompt_hash و human_score مطلوبان")
    try:
        human_score = float(human_score)
        if not (0 <= human_score <= 10):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="human_score يجب أن يكون عدداً بين 0 و 10")
    human_feedback_store.setdefault(prompt_hash, []).append(human_score)
    _save_human_feedback()
    return {"status": "success", "message": f"تم إضافة التقييم البشري ({len(human_feedback_store[prompt_hash])} تقييمات)"}

@app.get("/api/v1/human-feedback/{prompt_hash}")
async def get_human_feedback(prompt_hash: str):
    scores = human_feedback_store.get(prompt_hash, [])
    return {"prompt_hash": prompt_hash, "scores": scores, "count": len(scores)}

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "9.2",
        "tested_model": TESTED_MODEL,
        "jury_models": JURY_MODELS,
        "human_feedback_entries": sum(len(v) for v in human_feedback_store.values()),
        "groq_key": "✅" if os.environ.get("GROQ_API_KEY") else "❌ مفقود",
        "tg_token": "✅" if os.environ.get("TELEGRAM_BOT_TOKEN") else "❌ مفقود",
        "api_key_protection": "✅ مفعّلة" if API_SECRET_KEY else "❌ غير مفعّلة",
        "webhook_secret_protection": "✅ مفعّلة" if TELEGRAM_WEBHOOK_SECRET else "❌ غير مفعّلة",
    }

# ---------- Telegram Webhook (مع تحقق من secret token) ----------
@app.post("/tg-webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    # تحقق من الـ secret token إن كان مُفعَّلاً (يمنع أي طرف من استدعاء الـ webhook زوراً)
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="secret token غير صالح")

    try:
        data = await request.json()
    except Exception:
        return {"status": "ok"}

    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        print("❌ TELEGRAM_BOT_TOKEN مفقود")
        return {"status": "ok"}

    # أطلق المهمة الخلفية مع الاحتفاظ بمرجع لها لمنع الـ garbage collection
    task = asyncio.create_task(process_and_reply(chat_id, user_text))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"st
