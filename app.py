# ==============================================================
# AI-Unit Core Engine — بوت تيليجرام (V9.0 - Stabilized & Multi-Jury)
# الملف: app.py
# ==============================================================

import asyncio
import logging
import math
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

# ---------- الإعدادات العامة ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ai_unit")

app = FastAPI(title="AI-Unit Core Engine", version="9.0")

TESTED_MODEL = "llama-3.3-70b-versatile"
JURY_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TELEGRAM_MAX_LEN = 4096
MAX_USER_PROMPT_LEN = 8000
NUM_JURIES = 3
CALIBRATION_FACTOR = 1.15

_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=40)
    return _http_client

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

# ---------- 1. Groq Call ----------
async def _groq_call(
    messages: list,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
    timeout: int = 20,
) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY غير موجود")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    client = get_http_client()
    try:
        resp = await client.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Groq error: {str(e)[:200]}")
        return None

# ---------- 2. تقدير الصعوبة ----------
def _difficulty_fallback(text: str) -> int:
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75: return 2
    return 1

async def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    safe_prompt = prompt[:MAX_USER_PROMPT_LEN]
    difficulty_prompt = (
        "You are an AI difficulty assessor for an LLM benchmarking system.\n"
        "Rate the COGNITIVE difficulty of the user-provided text below, from 1 to 5.\n\n"
        "Scale:\n"
        "  1 = Greeting or trivial\n"
        "  2 = Simple factual\n"
        "  3 = Moderate reasoning\n"
        "  4 = Multi-step analysis\n"
        "  5 = Deep expertise or original thinking\n\n"
        "Output ONLY this JSON: {\"k\": <integer 1-5>, \"reason\": \"<one concise sentence>\"}\n"
        f"<user_data>\n{safe_prompt}\n</user_data>"
    )

    raw = await _groq_call(
        [{"role": "user", "content": difficulty_prompt}],
        JURY_MODEL, temperature=0.1, max_tokens=100, json_mode=True, timeout=10
    )

    if raw is None:
        k = _difficulty_fallback(prompt)
        return k, "تقدير احتياطي", False

    for candidate in re.findall(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            import json
            data = json.loads(candidate)
            k = int(data["k"])
            reason = str(data.get("reason", "")).strip()
            if 1 <= k <= 5:
                return k, reason, True
        except:
            continue
    k = _difficulty_fallback(prompt)
    return k, "تقدير احتياطي", False

# ---------- 3. الحسابات الرياضية ----------
def calculate_w_k(k: int) -> float:
    return round(math.e**k, 4)

def calculate_s_k(k: int, t_actual: float) -> float:
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (max(t_actual, 0.0) + t_target), 1.0)

def calculate_normalized_contribution(score: float, k: int, weight: float) -> float:
    norm_score = min(max(score / 10.0, 0.0), 1.0)
    return (norm_score ** k) * weight

# ---------- 4. استدعاء النموذج المختبر ----------
async def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    safe_prompt = prompt[:MAX_USER_PROMPT_LEN]
    start = time.monotonic()
    response = await _groq_call(
        [{"role": "user", "content": safe_prompt}],
        TESTED_MODEL, temperature=0.7, max_tokens=600, timeout=40
    )
    return response, time.monotonic() - start

# ---------- 5. Jury Functions ----------
async def call_jury_exponential(model_response: str, k: int) -> Dict:
    criteria = MASTER_CRITERIA[:min(2 ** (k - 1), len(MASTER_CRITERIA))]
    criteria_names = [c["name"] for c in criteria]
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in criteria])
    safe_response = model_response[:6000]

    jury_prompt = (
        f"You are the ULTIMATE AI Jury.\n"
        f"Evaluate the response based on EXACTLY {len(criteria)} criteria.\n"
        f"Output RAW JSON with keys: {criteria_names}\n"
        f"Values: float 0.0 to 10.0. Be harsh.\n"
        f"<response_data>\n{safe_response}\n</response_data>"
    )

    raw = await _groq_call(
        [{"role": "user", "content": jury_prompt}],
        JURY_MODEL, temperature=0.1, max_tokens=1000, json_mode=True, timeout=20
    )

    if raw is None:
        return {"scores": {c["name"]: 5.0 for c in criteria}, "is_real": False, "error": "فشل"}

    try:
        import json
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_str = raw[json_start:json_end]
        scores = json.loads(json_str)
        final_scores = {}
        for c in criteria:
            name = c["name"]
            val = float(scores.get(name, 5.0))
            final_scores[name] = min(max(val, 0.0), 10.0)
        return {"scores": final_scores, "is_real": True, "error": None}
    except Exception as e:
        log.warning(f"Jury JSON error: {e}")
        return {"scores": {c["name"]: 5.0 for c in criteria}, "is_real": False, "error": str(e)}

async def call_multi_jury(model_response: str, k: int) -> Dict:
    tasks = [call_jury_exponential(model_response, k) for _ in range(NUM_JURIES)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_scores = [res["scores"] for res in results if isinstance(res, dict) and "scores" in res]
    if not valid_scores:
        return await call_jury_exponential(model_response, k)
    
    final_scores = {}
    criteria = list(valid_scores[0].keys())
    for crit in criteria:
        vals = [s[crit] for s in valid_scores if crit in s]
        final_scores[crit] = statistics.mean(vals)
    
    return {
        "scores": final_scores,
        "is_real": True,
        "error": None,
        "jury_count": len(valid_scores)
    }

# ---------- 6. الخط الرئيسي ----------
async def run_ai_unit(prompt: str) -> Dict[str, Any]:
    if not prompt or not prompt.strip():
        return {"success": False, "error": "النص فارغ"}

    log.info(f"AI-Unit V9.0 | prompt_len={len(prompt)}")

    k, k_reason, k_is_real = await assess_difficulty(prompt)
    model_response, t_actual = await call_tested_model(prompt)
    if model_response is None:
        return {"success": False, "error": "فشل استدعاء النموذج"}

    jury = await call_multi_jury(model_response, k)
    scores = jury["scores"]
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    total_weighted_score = 0.0
    criterion_details = []
    for name, score in scores.items():
        criterion_data = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        weight = calculate_w_k(k) if criterion_data and criterion_data.get("weight") == "exp" else float(k)
        contribution = calculate_normalized_contribution(score, k, weight)
        total_weighted_score += contribution
        criterion_details.append({
            "name": name,
            "score": round(score, 2),
            "weight": round(weight, 4),
            "contribution": round(contribution, 4)
        })

    s_k = calculate_s_k(k, t_actual)
    ai_unit_score = round(total_weighted_score * s_k * CALIBRATION_FACTOR * 100, 2)

    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "k": k,
        "k_reason": k_reason,
        "k_assessed_by_ai": k_is_real,
        "criteria_count": len(scores),
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "ai_unit_score": ai_unit_score,
        "criterion_details": criterion_details,
        "t_actual": round(t_actual, 3),
        "model_response": model_response[:2000],  # اختصار للرد
        "jury_count": jury.get("jury_count", 1)
    }

# ---------- 7. Telegram Helpers ----------
def _escape_markdown_v1(text: str) -> str:
    for ch in ["_", "*", "`", "["]:
        text = text.replace(ch, f"\\{ch}")
    return text

def _split_telegram_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts

async def _send_tg(token: str, chat_id: int, text: str) -> None:
    client = get_http_client()
    for chunk in _split_telegram_message(text):
        try:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
            if resp.status_code != 200:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                    timeout=10,
                )
        except Exception as e:
            log.error(f"Telegram send error: {e}")

@app.post("/tg-webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    expected_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if expected_secret and x_telegram_bot_api_secret_token != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        return {"status": "ok"}

    await _send_tg(token, chat_id, "⏳ جارٍ التقييم بـ AI-Unit V9.0 ...")

    try:
        result = await run_ai_unit(user_text)
    except Exception as e:
        log.exception("خطأ في run_ai_unit")
        await _send_tg(token, chat_id, "❌ حدث خطأ داخلي.")
        return {"status": "ok"}

    if not result["success"]:
        await _send_tg(token, chat_id, f"❌ {result['error']}")
        return {"status": "ok"}

    # بناء الرد
    criteria_lines = "\n".join([f"  • {name}: {score:.1f}/10" for name, score in result["scores"].items()])
    reply = (
        f"🏆 *AI-Unit V9.0*\n"
        f"——————————————————\n"
        f"🎯 k = {result['k']} (صعوبة)\n"
        f"📊 عدد المحلّفين: {result['jury_count']}\n"
        f"📋 المعايير:\n{criteria_lines}\n\n"
        f"🏅 *النتيجة النهائية:* `{result['ai_unit_score']}` AIU\n"
        f"⏱️ الزمن: {result['t_actual']} ث\n"
        f"——————————————————\n"
        f"⚠️ AIU رقم داخلي (غير معياري عالمياً)"
    )
    await _send_tg(token, chat_id, reply)
    return {"status": "ok"}

# ---------- Endpoints إضافية ----------
@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt مطلوب")
    return await run_ai_unit(prompt)

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "9.0",
        "tested_model": TESTED_MODEL,
        "multi_jury": NUM_JURIES,
        "calibration": CALIBRATION_FACTOR
    }

@app.on_event("shutdown")
async def shutdown_event():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
