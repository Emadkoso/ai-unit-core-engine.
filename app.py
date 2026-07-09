# ==============================================================
# AI-Unit V11 — الإصدار التجاري (مجاني بالكامل) + واجهة بصرية
# التغييرات الجوهرية:
#   1) تخزين دائم عبر Supabase (مجاني) بدل SQLite.
#   2) نظام كاش ذكي لتوفير 90% من حصص Groq.
#   3) محلفان سريعان فقط → زمن استجابة ~8 ثوانٍ.
#   4) أمر /compare لمقارنة نموذجين (المنتج التسويقي).
#   5) واجهة بصرية (Dashboard) تعمل على الرابط الأساسي / 
# ==============================================================

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles  # <-- إضافة جديدة للواجهة
from fastapi.responses import FileResponse
import statistics
import time
import json
import math
import os
import asyncio
import hashlib
import random
from typing import Dict, Optional, List, Any, Tuple

import httpx
from supabase import create_client, Client

# ---------- الإعدادات العامة ----------
app = FastAPI(title="AI-Unit Commercial V11", version="11.0")
TESTED_MODEL = "llama-3.3-70b-versatile"

# ---------- متغيرات البيئة الإلزامية ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "").strip()
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ تحذير: Supabase غير مهيأ. سيتم التخزين في الذاكرة فقط (يُفقد عند إعادة التشغيل).")

# ---------- تهيئة Supabase ----------
_supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ تم الاتصال بـ Supabase بنجاح (تخزين دائم).")
    except Exception as e:
        print(f"❌ فشل الاتصال بـ Supabase: {e}. التبديل للتخزين المؤقت بالذاكرة.")

# ---------- طبقة التخزين المتسامحة مع الأخطاء (ذاكرة احتياطية) ----------
_memory_store: Dict[str, list] = {}
_memory_cache: Dict[str, dict] = {}

def _is_supabase_ready() -> bool:
    return _supabase is not None

# ================= دوال التخزين الدائم والكاش =================

def save_human_score(prompt_hash: str, score: float):
    if _is_supabase_ready():
        try:
            _supabase.table("human_feedback").insert({
                "prompt_hash": prompt_hash,
                "score": score
            }).execute()
            return
        except Exception as e:
            print(f"⚠️ فشل كتابة Supabase (سجل): {e}")
    _memory_store.setdefault(prompt_hash, []).append(score)

def get_human_scores(prompt_hash: str, limit: int = 5) -> List[float]:
    if _is_supabase_ready():
        try:
            res = _supabase.table("human_feedback") \
                .select("score") \
                .eq("prompt_hash", prompt_hash) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return [r["score"] for r in res.data]
        except Exception as e:
            print(f"⚠️ فشل قراءة Supabase (سجل): {e}")
    return _memory_store.get(prompt_hash, [])[-limit:]

def count_human_scores() -> int:
    if _is_supabase_ready():
        try:
            res = _supabase.table("human_feedback").select("id", count="exact").execute()
            return res.count or 0
        except Exception:
            pass
    return sum(len(v) for v in _memory_store.values())

# ---------- دوال الكاش (الأهم لتوفير المال) ----------
def get_cached_result(prompt_hash: str) -> Optional[dict]:
    if _is_supabase_ready():
        try:
            res = _supabase.table("response_cache") \
                .select("result_json") \
                .eq("prompt_hash", prompt_hash) \
                .maybe_single() \
                .execute()
            if res.data:
                return res.data["result_json"]
        except Exception as e:
            print(f"⚠️ فشل قراءة الكاش: {e}")
    return _memory_cache.get(prompt_hash)

def save_cached_result(prompt_hash: str, result: dict):
    if _is_supabase_ready():
        try:
            _supabase.table("response_cache").upsert({
                "prompt_hash": prompt_hash,
                "result_json": result
            }).execute()
            return
        except Exception as e:
            print(f"⚠️ فشل كتابة الكاش: {e}")
    _memory_cache[prompt_hash] = result

# ================= باقي دوال النظام الأساسية =================

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_http_client_groq: Optional[httpx.AsyncClient] = None
_http_client_tg: Optional[httpx.AsyncClient] = None

# محلفان مستقلان وسريعان فقط
JURY_MODELS = [
    {"name": "llama_fast", "model": "llama-3.1-8b-instant", "temperature": 0.3, "weight": 0.5, "family": "Meta/Llama"},
    {"name": "gptoss_fast", "model": "openai/gpt-oss-20b", "temperature": 0.3, "weight": 0.5, "family": "OpenAI/GPT-OSS"},
]

# معايير التقييم العامة
MASTER_CRITERIA = [
    {"name": "accuracy", "desc": "Is the answer fully correct and free of factual errors?", "weight": "exp"},
    {"name": "clarity", "desc": "Is the answer clear and direct without ambiguity?", "weight": "linear"},
    {"name": "completeness", "desc": "Did it cover all aspects of the question?", "weight": "linear"},
    {"name": "coherence", "desc": "Are the ideas logically connected and sequential?", "weight": "linear"},
    {"name": "depth", "desc": "Did it go beyond the surface into root causes?", "weight": "exp"},
    {"name": "uniqueness", "desc": "Does it offer a non-repetitive perspective?", "weight": "semi_exp"},
    {"name": "safety", "desc": "Does it avoid bias, hate, or harm?", "weight": "linear"},
    {"name": "strategy", "desc": "Does it provide a strategically actionable plan?", "weight": "exp"},
]

def get_criteria_for_k(k: int) -> List[Dict]:
    count = max(2, 2 ** (k - 1))
    return MASTER_CRITERIA[:count]

def _extract_json(raw: str) -> Optional[dict]:
    if not raw: return None
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(raw, start)
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError: pass
        start = raw.find("{", start + 1)
    return None

async def _groq_call_async(messages, model, temperature=0.7, max_tokens=600, json_mode=False, timeout=20) -> Optional[str]:
    if not GROQ_API_KEY: return None
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode: payload["response_format"] = {"type": "json_object"}
    try:
        resp = await _http_client_groq.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq error ({model}): {e}")
        return None

def _difficulty_fallback(text: str) -> int:
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75: return 2
    return 1

async def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    difficulty_prompt = (
        "You are an AI difficulty assessor. Rate the COGNITIVE difficulty from 1 to 5. "
        'Output ONLY JSON: {"k": <1-5>, "reason": "<one sentence>"}\n'
        f'Prompt: """{prompt}"""'
    )
    raw = await _groq_call_async(
        messages=[{"role": "user", "content": difficulty_prompt}],
        model="llama-3.1-8b-instant", temperature=0.1, max_tokens=80, json_mode=True, timeout=10
    )
    if raw is None: return _difficulty_fallback(prompt), "fallback", False
    data = _extract_json(raw)
    if data:
        try:
            k = int(data["k"])
            if 1 <= k <= 5: return k, data.get("reason", ""), True
        except: pass
    return _difficulty_fallback(prompt), "fallback", False

async def call_tested_model(prompt: str, custom_model: Optional[str] = None) -> Tuple[Optional[str], float]:
    model = custom_model or TESTED_MODEL
    start = time.time()
    response = await _groq_call_async(
        messages=[{"role": "user", "content": prompt}], model=model, temperature=0.7, max_tokens=600
    )
    return response, time.time() - start

async def evaluate_single_jury(model_response: str, k: int, jury_model: str, temperature: float) -> Dict:
    criteria = get_criteria_for_k(k)
    shuffled = criteria.copy()
    random.shuffle(shuffled)
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in shuffled])
    jury_prompt = (
        "You are an independent AI evaluator. For EACH criterion give score (0-10) and short reason.\n"
        f"CRITERIA:\n{criteria_descs}\n"
        'Output JSON: {"criterion_name": {"score": <0-10>, "reason": "..."}}\n\n'
        f"RESPONSE:\n\"\"\"{model_response}\"\"\""
    )
    raw = await _groq_call_async(
        messages=[{"role": "user", "content": jury_prompt}],
        model=jury_model, temperature=temperature, max_tokens=500, json_mode=True, timeout=15
    )
    scores = {}; reasons = {}
    data = _extract_json(raw) if raw else None
    if data:
        for c in criteria:
            entry = data.get(c["name"])
            if isinstance(entry, dict) and "score" in entry:
                try:
                    scores[c["name"]] = min(max(float(entry["score"]), 0.0), 10.0)
                    reasons[c["name"]] = str(entry.get("reason", ""))[:200]
                except: pass
            elif isinstance(entry, (int, float)):
                scores[c["name"]] = min(max(float(entry), 0.0), 10.0)
                reasons[c["name"]] = ""
    for c in criteria:
        scores.setdefault(c["name"], 5.0)
        reasons.setdefault(c["name"], "N/A")
    return {"scores": scores, "reasons": reasons}

async def multi_jury_evaluate(model_response: str, k: int) -> Tuple[Dict, Dict]:
    tasks = [evaluate_single_jury(model_response, k, jury["model"], jury["temperature"]) for jury in JURY_MODELS]
    results = await asyncio.gather(*tasks)
    all_scores: Dict[str, list] = {}
    merged_reasons: Dict[str, List[Dict[str, str]]] = {}
    for idx, res in enumerate(results):
        family = JURY_MODELS[idx]["family"]
        for name, score in res["scores"].items():
            all_scores.setdefault(name, []).append(score)
            merged_reasons.setdefault(name, []).append({"family": family, "reason": res["reasons"].get(name, "")})
    final_scores = {}
    for name, scores_list in all_scores.items():
        weighted_sum = sum(scores_list[i] * JURY_MODELS[i]["weight"] for i in range(len(scores_list)))
        total_w = sum(j["weight"] for j in JURY_MODELS)
        final_scores[name] = round(weighted_sum / total_w, 2) if total_w else 0.0
    return final_scores, merged_reasons

async def run_ai_unit(prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    
    # 1. فحص الكاش أولاً
    cached = get_cached_result(prompt_hash)
    if cached:
        print(f"✅ كاش: تم الرد على السؤال من قاعدة البيانات (توفير حصة Groq).")
        cached["from_cache"] = True
        return cached

    # 2. التقييم الجديد
    k, k_reason, k_is_real = await assess_difficulty(prompt)
    
    model_response, t_actual = await call_tested_model(prompt, model_override)
    if model_response is None:
        return {"success": False, "error": "failed to call tested model"}
    
    scores, reasons = await multi_jury_evaluate(model_response, k)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    
    w_k = round(math.e ** k, 4)
    ai_unit_score = round(avg_score * w_k, 4)
    
    result = {
        "success": True,
        "model_tested": model_override or TESTED_MODEL,
        "k": k,
        "k_reason": k_reason,
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "w_k": w_k,
        "ai_unit_score": ai_unit_score,
        "t_actual": round(t_actual, 3),
        "model_response": model_response,
        "prompt_hash": prompt_hash,
        "from_cache": False,
        "criterion_details": [
            {"name": n, "score": s, "reasons_by_family": reasons.get(n, [])}
            for n, s in scores.items()
        ]
    }
    
    save_cached_result(prompt_hash, result)
    return result

# ================= دوال التيليجرام والأوامر =================
_background_tasks: set = set()

async def _send_tg(chat_id: int, text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token: return
    for part in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        try:
            await _http_client_tg.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": part, "parse_mode": "Markdown"},
                timeout=10
            )
        except: pass

async def process_and_reply(chat_id: int, user_text: str):
    try:
        # ---- أمر المقارنة ----
        if user_text.startswith("/compare"):
            parts = user_text.split(maxsplit=3)
            if len(parts) < 4:
                await _send_tg(chat_id, "⚠️ الصيغة: `/compare [نموذج1] [نموذج2] [السؤال]`\nمثال: `/compare llama-3.3-70b-versatile openai/gpt-oss-20b ما هي العاصمة؟`")
                return
            _, model1, model2, question = parts
            await _send_tg(chat_id, f"⚖️ جارٍ مقارنة `{model1}` vs `{model2}`... (قد يستغرق 15 ثانية)")
            
            res1 = await run_ai_unit(question, model_override=model1)
            res2 = await run_ai_unit(question, model_override=model2)
            
            if not res1["success"] or not res2["success"]:
                await _send_tg(chat_id, "❌ فشل في تقييم أحد النماذج. تأكد من الأسماء.")
                return
            
            reply = (
                f"📊 *نتيجة المقارنة*\n"
                f"——————————————\n"
                f"*السؤال:* {question[:100]}...\n\n"
                f"🔹 *{model1}*: AIU = {res1['ai_unit_score']} (الوقت: {res1['t_actual']}ث)\n"
                f"🔸 *{model2}*: AIU = {res2['ai_unit_score']} (الوقت: {res2['t_actual']}ث)\n\n"
                f"📌 *التفاصيل:*\n"
                f"{model1}: {res1['scores']}\n"
                f"{model2}: {res2['scores']}\n\n"
                f"💡 *التوصية:* النموذج {model1 if res1['ai_unit_score'] > res2['ai_unit_score'] else model2} يتفوق في هذا السؤال."
            )
            await _send_tg(chat_id, reply)
            return

        # ---- الأمر العادي ----
        await _send_tg(chat_id, "⏳ جارٍ التقييم السريع (محلفان مستقلان)...")
        result = await run_ai_unit(user_text)
        if not result["success"]:
            await _send_tg(chat_id, f"❌ خطأ: {result['error']}")
            return
        
        if result.get("from_cache"):
            cache_tag = "⚡ (من الكاش - رد فوري)"
        else:
            cache_tag = f"⏱️ {result['t_actual']} ثانية"

        scores_lines = "\n".join([f"  • {k}: {v}" for k, v in result["scores"].items()])
        reply = (
            f"🏆 *AI-Unit V11*\n"
            f"——————————————\n"
            f"📊 الصعوبة: k={result['k']} | AIU={result['ai_unit_score']}\n"
            f"⚙️ المحلفين: {len(JURY_MODELS)} (مستقلين)\n"
            f"📝 التقييم:\n{scores_lines}\n"
            f"⏳ {cache_tag}\n"
            f"🤖 النموذج المختبر: {result['model_tested']}"
        )
        await _send_tg(chat_id, reply)

    except Exception as e:
        await _send_tg(chat_id, f"❌ خطأ داخلي: {str(e)[:200]}")

# ================= نقاط نهاية FastAPI =================
@app.on_event("startup")
async def startup():
    global _http_client_groq, _http_client_tg
    _http_client_groq = httpx.AsyncClient(timeout=30.0)
    _http_client_tg = httpx.AsyncClient(timeout=15.0)
    if not GROQ_API_KEY: print("❌ GROQ_API_KEY مفقود")
    print("🚀 AI-Unit V11 جاهز (مجاني، سريع، مع كاش Supabase)")

@app.on_event("shutdown")
async def shutdown():
    if _http_client_groq: await _http_client_groq.aclose()
    if _http_client_tg: await _http_client_tg.aclose()

@app.post("/tg-webhook")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: Optional[str] = Header(None)):
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=401)
    data = await request.json()
    if "message" not in data or "text" not in data["message"]: return {"status": "ok"}
    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    task = asyncio.create_task(process_and_reply(chat_id, user_text))
    _background_tasks.add(task); task.add_done_callback(_background_tasks.discard)
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {
        "version": "11.0",
        "status": "operational",
        "supabase": "connected" if _is_supabase_ready() else "memory_fallback",
        "cache_enabled": True,
        "jury_count": len(JURY_MODELS),
        "note": "جاهز للاستخدام التجاري مع ميزانية صفرية."
    }

# ================= الواجهة البصرية (Dashboard) =================
# التأكد من وجود مجلد static
os.makedirs("static", exist_ok=True)

# نقطة النهاية الرئيسية (الصفحة الرئيسية)
@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")

# ربط مجلد الملفات الثابتة
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= تشغيل الخادم =================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
