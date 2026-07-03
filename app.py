# ==============================================================
# AI-Unit Core Engine — الإصدار النهائي (V9.3)
# ملف واحد: app_final.py
# ==============================================================
# الإصلاحات النهائية:
#   1) تصحيح اسم النموذج: gemma2-9b-it → gemma-2-9b-it
#   2) إزالة أي استخدام لـ os.en (تم استبدال الكل بـ os.environ)
#   3) فصل عملاء HTTP (Groq بمهلة 120s، Telegram بمهلة 15s)
#   4) تقطيع رسائل Telegram الطويلة (>4000 حرف)
#   5) التحقق من صحة النماذج عند بدء التشغيل
#   6) معالجة قوية للأخطاء تظهر في التيليجرام
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
app = FastAPI(title="AI-Unit Core Engine V9.3", version="9.3")

TESTED_MODEL = "llama-3.3-70b-versatile"

JURY_MODELS = [
    {"name": "academic",   "model": "gemma-2-9b-it",         "weight": 0.4, "desc": "academic precise"},
    {"name": "analytical", "model": "llama-3.1-8b-instant",  "weight": 0.3, "desc": "analytical logical"},
    {"name": "creative",   "model": "llama-3.3-70b-versatile","weight": 0.3, "desc": "creative flexible"},
]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    {"name": "accuracy", "desc": "Is the answer fully correct and free of factual errors?", "weight": "exp"},
    {"name": "clarity", "desc": "Is the answer clear and direct without ambiguity?", "weight": "linear"},
    {"name": "completeness", "desc": "Did it cover all aspects of the question?", "weight": "linear"},
    {"name": "coherence", "desc": "Are the ideas logically connected and sequential?", "weight": "linear"},
    {"name": "depth", "desc": "Did it go beyond the surface into root causes and deep analysis?", "weight": "exp"},
    {"name": "uniqueness", "desc": "Does it offer a rare or non-repetitive perspective?", "weight": "semi_exp"},
    {"name": "creativity", "desc": "Does it offer innovative solutions or new angles?", "weight": "semi_exp"},
    {"name": "safety", "desc": "Does it avoid bias, hate, or harm?", "weight": "linear"},
    {"name": "strategy", "desc": "Does it provide a strategically actionable plan?", "weight": "exp"},
    {"name": "predictive_power", "desc": "Does it accurately predict outcomes or future challenges?", "weight": "semi_exp"},
    {"name": "critical_analysis", "desc": "Did it analyze and critique assumptions based on evidence?", "weight": "semi_exp"},
    {"name": "originality", "desc": "Is the answer entirely new and not found in other models?", "weight": "semi_exp"},
    {"name": "fallacy_detection", "desc": "Did it detect logical fallacies in the question itself?", "weight": "exp"},
    {"name": "rhetorical_beauty", "desc": "Is the phrasing linguistically eloquent and impactful?", "weight": "linear"},
    {"name": "adaptability", "desc": "Does the answer adapt to different contexts or audiences?", "weight": "linear"},
    {"name": "generative_power", "desc": "Does it generate new knowledge or recycle old knowledge?", "weight": "semi_exp"},
]

# ---------- تخزين التقييمات البشرية ----------
HUMAN_FEEDBACK_FILE = Path("/tmp/human_feedback_store.json")
human_feedback_store: Dict[str, list] = {}

def _load_human_feedback():
    global human_feedback_store
    if HUMAN_FEEDBACK_FILE.exists():
        try:
            human_feedback_store = json.loads(HUMAN_FEEDBACK_FILE.read_text())
        except Exception as e:
            print(f"WARN: failed to load human feedback file: {e}")
            human_feedback_store = {}

def _save_human_feedback():
    try:
        HUMAN_FEEDBACK_FILE.write_text(json.dumps(human_feedback_store))
    except Exception as e:
        print(f"WARN: failed to save human feedback file: {e}")

_load_human_feedback()

_background_tasks: set = set()

# ---------- عملاء HTTP منفصلون ----------
_http_client_groq: Optional[httpx.AsyncClient] = None
_http_client_tg: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global _http_client_groq, _http_client_tg
    _http_client_groq = httpx.AsyncClient(timeout=120.0)
    _http_client_tg = httpx.AsyncClient(timeout=15.0)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY غير موجود")
        return

    print("🔍 جارٍ التحقق من صحة النماذج...")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for jury in JURY_MODELS:
        model = jury["model"]
        test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        }
        try:
            resp = await _http_client_groq.post(GROQ_URL, json=test_payload, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                print(f"✅ نموذج {model} متاح")
            else:
                print(f"❌ نموذج {model} غير متاح (HTTP {resp.status_code}) - تحقق من الاسم")
        except Exception as e:
            print(f"❌ فشل التحقق من نموذج {model}: {e}")

@app.on_event("shutdown")
async def _shutdown():
    if _http_client_groq:
        await _http_client_groq.aclose()
    if _http_client_tg:
        await _http_client_tg.aclose()

# ---------- دوال مساعدة ----------
def _extract_json(raw: str) -> Optional[dict]:
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
        print("ERROR: GROQ_API_KEY not set")
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
        resp = await _http_client_groq.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"ERROR: Groq error ({model}): {e}")
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
        return _difficulty_fallback(prompt), "fallback estimate (call failed)", False
    data = _extract_json(raw)
    if data:
        try:
            k = int(data["k"])
            if 1 <= k <= 5:
                return k, data.get("reason", "AI estimate"), True
        except (KeyError, ValueError, TypeError):
            pass
    return _difficulty_fallback(prompt), "fallback estimate (invalid JSON)", False

def calculate_w_k(k: int) -> float:
    return round(math.e ** k, 4)

def calculate_s_k(k: int, t_actual: float) -> float:
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)

def get_criteria_for_k(k: int) -> List[Dict]:
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
    for c in criteria:
        scores.setdefault(c["name"], 1.0)
    return {"scores": scores, "is_fallback": is_fallback}

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

def apply_human_correction(prompt_hash: str, avg_score: float) -> Tuple[float, bool]:
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

async def run_ai_unit(prompt: str) -> Dict[str, Any]:
    k, k_reason, k_is_real = await assess_difficulty(prompt)
    w_k = calculate_w_k(k)

    model_response, t_actual = await call_tested_model(prompt)
    if model_response is None:
        return {"success": False, "error": "failed to call tested model"}

    scores, fallback_juries = await multi_jury_evaluate(model_response, k)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    corrected_avg_score, human_applied = apply_human_correction(prompt_hash, avg_score)

    criterion_details = []
    total_weight_sum = 0.0
    weighted_score_sum = 0.0
    for name, score in scores.items():
        criterion = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        weight = get_criterion_weight(criterion, k) if criterion else 1.0
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

    normalized_weighted_avg = weighted_score_sum / total_weight_sum if total_weight_sum else 0.0
    s_k = calculate_s_k(k, t_actual)
    ai_unit_score = round(normalized_weighted_avg * w_k * s_k, 4)

    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "jury_models": [j["model"] for j in JURY_MODELS],
        "jury_fallback_used": fallback_juries,
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
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return

    max_len = 4000
    parts = [text[i:i+max_len] for i in range(0, len(text), max_len)]

    for part in parts:
        try:
            await _http_client_tg.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": part, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            print(f"ERROR: failed to send Telegram message: {e}")
            try:
                await _http_client_tg.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": part[:500]},
                    timeout=10,
                )
            except Exception as e2:
                print(f"CRITICAL: even fallback Telegram send failed: {e2}")

async def process_and_reply(chat_id: int, user_text: str):
    try:
        await _send_tg(chat_id, "⏳ جارٍ التقييم بـ Multi-Jury V9.3 ...")
        result = await run_ai_unit(user_text)

        if not result["success"]:
            await _send_tg(chat_id, f"❌ خطأ: {result['error']}")
            return

        scores_lines = "\n".join([f"  • {name}: {score}" for name, score in result["scores"].items()])
        fb_note = ""
        if result["jury_fallback_used"]:
            fb_note = f"\n⚠️ محلفون احتياطيون: {', '.join(result['jury_fallback_used'])}"

        reply = (
            f"🏆 *AI-Unit V9.3*\n"
            f"——————————————\n"
            f"🎯 k={result['k']} | AIU={result['ai_unit_score']}\n"
            f"⚙️ المحلفين: {len(result['jury_models'])}\n"
            f"📊 التقييم:\n{scores_lines}\n"
            f"⏱️ {result['t_actual']} ث{fb_note}\n"
            f"🔍 {'✅ مع تحقق بشري' if result['human_correction_applied'] else '🤖 تقييم آلي فقط'}"
        )
        await _send_tg(chat_id, reply)

    except Exception as e:
        try:
            await _send_tg(chat_id, f"❌ خطأ داخلي جسيم: {str(e)[:200]}")
        except:
            print(f"FATAL: Cannot send error message to chat {chat_id}")

# ---------- نقاط النهاية API ----------
def _check_api_key(x_api_key: Optional[str]):
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' field is required")
    return await run_ai_unit(prompt)

@app.post("/api/v1/human-feedback")
async def submit_human_feedback(request: Request, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    body = await request.json()
    prompt_hash = body.get("prompt_hash")
    human_score = body.get("human_score")
    if not prompt_hash or human_score is None:
        raise HTTPException(status_code=400, detail="prompt_hash and human_score are required")
    try:
        human_score = float(human_score)
        if not (0 <= human_score <= 10):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="human_score must be a number between 0 and 10")
    human_feedback_store.setdefault(prompt_hash, []).append(human_score)
    _save_human_feedback()
    return {"status": "success", "message": f"Human score added ({len(human_feedback_store[prompt_hash])} scores)"}

@app.get("/api/v1/human-feedback/{prompt_hash}")
async def get_human_feedback(prompt_hash: str):
    scores = human_feedback_store.get(prompt_hash, [])
    return {"prompt_hash": prompt_hash, "scores": scores, "count": len(scores)}

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "9.3",
        "tested_model": TESTED_MODEL,
        "jury_models": [j["model"] for j in JURY_MODELS],
        "human_feedback_entries": sum(len(v) for v in human_feedback_store.values()),
        "groq_key": "set" if os.environ.get("GROQ_API_KEY") else "missing",
        "tg_token": "set" if os.environ.get("TELEGRAM_BOT_TOKEN") else "missing",
        "api_key_protection": "enabled" if API_SECRET_KEY else "disabled",
        "webhook_secret_protection": "enabled" if TELEGRAM_WEBHOOK_SECRET else "disabled",
    }

@app.post("/tg-webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret token")

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
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return {"status": "ok"}

    task = asyncio.create_task(process_and_reply(chat_id, user_text))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
