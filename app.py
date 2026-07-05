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

# ---------- الإعدادات العامة ----------

app = FastAPI(title="AI-Unit Core Engine V10.5", version="10.5")

TESTED_MODEL = "llama-3.3-70b-versatile"

# استقلالية المحلفين الحقيقية وتأمين المجموعات البديلة (Backup)
JURY_MODELS = [
    {"name": "llama_family", "model": "llama-3.1-8b-instant", "temperature": 0.3, "weight": 0.34, "family": "Meta/Llama", "truly_independent": True, "tier": "production"},
    {"name": "gptoss_family", "model": "openai/gpt-oss-20b",  "temperature": 0.3, "weight": 0.33, "family": "OpenAI/GPT-OSS", "truly_independent": True, "tier": "production"},
    {"name": "qwen_family",  "model": "qwen/qwen3-32b",       "temperature": 0.3, "weight": 0.33, "family": "Qwen/Alibaba", "truly_independent": True, "tier": "preview_risk_of_discontinuation"},
]

FALLBACK_JURY_MODEL = {
    "name": "llama_fallback", 
    "model": "llama-3.3-70b-versatile", 
    "temperature": 0.4, 
    "weight": 0.33, 
    "family": "Meta/Llama (احتياطي عند فشل Qwen)", 
    "truly_independent": False, 
    "tier": "production"
}

# المجموعات البديلة الاستراتيجية لحماية الهوية التنافسية للـ SaaS عند التبادل الساخن
BACKUP_POOL = [
    {"name": "qwen_backup", "model": "qwen-2.5-72b-instruct", "temperature": 0.2, "family": "Qwen/Alibaba"},
    {"name": "mistral_family", "model": "mixtral-8x7b-instruct", "temperature": 0.3, "family": "Mistral/France"}
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

# ---------- المعايير الأساسية ----------

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

DOMAIN_CRITERIA: Dict[str, List[Dict]] = {
    "code": [
        {"name": "correctness_logic", "desc": "Would this code compile/run and behave as intended?", "weight": "exp"},
        {"name": "edge_case_handling", "desc": "Does it handle edge cases and errors properly?", "weight": "exp"},
        {"name": "readability_maintainability", "desc": "Is the code readable and maintainable?", "weight": "linear"},
    ],
    "medical": [
        {"name": "clinical_safety", "desc": "Could following this advice cause harm if wrong?", "weight": "exp"},
        {"name": "evidence_basis", "desc": "Is the claim grounded in established medical evidence?", "weight": "exp"},
        {"name": "appropriate_caution", "desc": "Does it recommend professional consultation where needed?", "weight": "linear"},
    ],
    "legal": [
        {"name": "jurisdiction_awareness", "desc": "Does it acknowledge legal variation by jurisdiction?", "weight": "semi_exp"},
        {"name": "legal_accuracy", "desc": "Are legal claims accurate and not fabricated?", "weight": "exp"},
    ],
    "math": [
        {"name": "derivation_validity", "desc": "Is each step of the derivation logically valid?", "weight": "exp"},
        {"name": "final_answer_correctness", "desc": "Is the final numeric/symbolic answer correct?", "weight": "exp"},
    ],
}

DOMAIN_KEYWORDS = {
    "code": ["كود", "برمج", "دالة", "function", "class", "python", "javascript", "خوارزم", "bug", "compile", "api", "سكريبت", "def ", "import "],
    "medical": ["طبي", "دواء", "مرض", "أعراض", "علاج", "جرعة", "طبيب", "تشخيص", "medicine", "symptom", "diagnosis", "دماغ", "خلايا", "عصب", "نفسي", "قلق"],
    "legal": ["قانون", "عقد", "دعوى", "محكمة", "تشريع", "legal", "contract", "lawsuit", "regulation"],
    "math": ["برهان", "معادلة", "نظرية", "تفاضل", "تكامل", "proof", "theorem", "equation", "derivative", "integral", "فيبوناتشي", "متسلسلة"],
}

def detect_domain(prompt: str) -> str:
    text = prompt.lower()
    scores = {domain: sum(1 for kw in kws if kw in text) for domain, kws in DOMAIN_KEYWORDS.items()}
    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 0 else "general"

def get_criteria_for_k(k: int, domain: str) -> List[Dict]:
    # تأمين وحماية الـ Context Window من الانفجار الأسي للتوكنز
    count = min(6, max(2, 2 ** (k - 1)))
    base = MASTER_CRITERIA[:count]
    domain_extra = DOMAIN_CRITERIA.get(domain, [])
    return base + domain_extra

def should_evaluate_request(k: int) -> bool:
    """نظام الفحص العشوائي الذكي لحماية الميزانية في الـ Production (Smart Sampling)"""
    if k >= 4:
        return True
    sampling_rates = {1: 0.10, 2: 0.25, 3: 0.50}
    return random.random() < sampling_rates.get(k, 1.0)

# ---------- تخزين دائم عبر SQLite ----------

DB_PATH = Path(os.environ.get("AI_UNIT_DB_PATH", str(Path(__file__).parent / "ai_unit_data" / "ai_unit.db")))

DB_AVAILABLE = True
_memory_fallback_store: Dict[str, list] = {}

def _ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def _db_conn():
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _init_db():
    global DB_AVAILABLE
    try:
        with _db_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS human_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT NOT NULL,
                score REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_hash ON human_feedback(prompt_hash)")
            DB_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ فشل تهيئة SQLite ({e}) — التبديل لتخزين مؤقت بالذاكرة")
        DB_AVAILABLE = False

_init_db()

def save_human_score(prompt_hash: str, score: float):
    if not DB_AVAILABLE:
        _memory_fallback_store.setdefault(prompt_hash, []).append(score)
        return
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO human_feedback (prompt_hash, score, created_at) VALUES (?, ?, ?)",
                (prompt_hash, score, time.time()),
            )
    except Exception as e:
        print(f"⚠️ فشل الكتابة في SQLite ({e})")
        _memory_fallback_store.setdefault(prompt_hash, []).append(score)

def get_human_scores(prompt_hash: str, limit: int = 5) -> List[float]:
    if not DB_AVAILABLE:
        return _memory_fallback_store.get(prompt_hash, [])[-limit:]
    try:
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT score FROM human_feedback WHERE prompt_hash = ? ORDER BY id DESC LIMIT ?",
                (prompt_hash, limit),
            ).fetchall()
            return [r[0] for r in rows]
    except Exception as e:
        print(f"⚠️ فشل القراءة من SQLite ({e})")
        return _memory_fallback_store.get(prompt_hash, [])[-limit:]

def count_human_scores() -> int:
    if not DB_AVAILABLE:
        return sum(len(v) for v in _memory_fallback_store.values())
    try:
        with _db_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()
            return row[0] if row else 0
    except Exception:
        return sum(len(v) for v in _memory_fallback_store.values())

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

    print("🔍 جارٍ التحقق من صحة النماذج مقابل Groq API الحقيقي...")  
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}  
    independent_count = 0  
    for idx, jury in enumerate(JURY_MODELS):  
        model = jury["model"]  
        test_payload = {"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}  
        ok = False  
        try:  
            resp = await _http_client_groq.post(GROQ_URL, json=test_payload, headers=headers, timeout=5.0)  
            ok = resp.status_code == 200  
        except Exception as e:  
            print(f"❌ فشل التحقق من نموذج {model}: {e}")  

        if ok:  
            risk_tag = " (⚠️ Preview)" if jury.get("tier") == "preview_risk_of_discontinuation" else ""  
            tag = "✅ مستقل فعليًا" if jury.get("truly_independent") else "⚠️ تكرار عائلة"  
            print(f"{tag} — {model} ({jury['family']}){risk_tag}")  
        else:  
            print(f"🔁 نموذج {model} فشل — استبداله تلقائيًا بالمحلّف الاحتياطي")  
            JURY_MODELS[idx] = FALLBACK_JURY_MODEL.copy()  

        if JURY_MODELS[idx].get("truly_independent"):  
            independent_count += 1  

    print(f"📊 عدد المحلفين المستقلين فعليًا بعد الفحص: {independent_count}")

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

async def _groq_call_async(messages, model, temperature=0.7, max_tokens=600, json_mode=False, timeout=20) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
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
        model="llama-3.1-8b-instant", temperature=0.1, max_tokens=80, json_mode=True, timeout=10,
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
        messages=[{"role": "user", "content": prompt}], model=TESTED_MODEL, temperature=0.7, max_tokens=600,
    )
    return response, time.time() - start

async def evaluate_single_jury(model_response: str, k: int, domain: str, jury_model: str, temperature: float) -> Dict:
    criteria = get_criteria_for_k(k, domain)
    shuffled = criteria.copy()
    random.shuffle(shuffled)

    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in shuffled])  
    jury_prompt = (  
        "You are an independent AI evaluator. Evaluate the response on these criteria.\n"  
        "For EACH criterion give a numeric score (0.0-10.0) AND a short one-sentence reason under 5 words.\n"  
        f"CRITERIA:\n{criteria_descs}\n\n"  
        'Output ONLY JSON in this exact shape:\n'  
        '{"criterion_name": {"score": <float>, "reason": "<short text>"}, ...}\n\n'  
        f"RESPONSE TO EVALUATE:\n\"\"\"{model_response}\"\"\""  
    )  
    # رفع الـ max_tokens إلى 1500 لضمان الحماية من بتر مخرجات الـ JSON البنيوية
    raw = await _groq_call_async(  
        messages=[{"role": "user", "content": jury_prompt}],  
        model=jury_model, temperature=temperature, max_tokens=1500, json_mode=True, timeout=25,  
    )  
    scores: Dict[str, float] = {}  
    reasons: Dict[str, str] = {}  
    is_fallback = True  
    data = _extract_json(raw) if raw else None  
    if data:  
        is_fallback = False
        for c in criteria:  
            entry = data.get(c["name"])  
            if isinstance(entry, dict) and "score" in entry:  
                try:  
                    scores[c["name"]] = min(max(float(entry["score"]), 0.0), 10.0)  
                    reasons[c["name"]] = str(entry.get("reason", ""))[:300]
                except Exception:
                    scores[c["name"]] = 5.0
                    reasons[c["name"]] = "error parsing single score"
            else:
                scores[c["name"]] = 5.0
                reasons[c["name"]] = "missing score data"
    else:
        # Fallback values if JSON extraction failed entirely
        for c in criteria:
            scores[c["name"]] = 5.0
            reasons[c["name"]] = "jury call fallback due to extraction error"

    return {"scores": scores, "reasons": reasons, "fallback": is_fallback}

# ---------- الـ Endpoints الأساسية ونظام الحساب ----------

@app.post("/evaluate")
async def run_evaluation(request: Request):
    # توثيق الحماية الأساسي للـ API المفتاحي الخاص بك
    if API_SECRET_KEY:
        auth_header = request.headers.get("X-API-Secret")
        if auth_header != API_SECRET_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized API Secret")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt field is required")

    # كشف المجال وحساب درجة الصعوبة الحركية
    domain = detect_domain(prompt)
    k, diff_reason, diff_ok = await assess_difficulty(prompt)

    # تشغيل وفحص الـ Smart Sampling الاقتصادي لحماية خطوط الائتمان الخاصة بك
    if not should_evaluate_request(k):
        return {
            "status": "skipped",
            "message": "Bypassed by smart sampling algorithm to minimize production cost.",
            "k": k,
            "domain": domain
        }

    # استدعاء النموذج الأساسي المطلوب تقييمه
    model_response, t_actual = await call_tested_model(prompt)
    if not model_response:
        raise HTTPException(status_code=502, detail="Tested model returned empty response")

    # بناء المهام المتوازية الحرة للجنة التحكيم
    tasks = []
    for jury in JURY_MODELS:
        tasks.append(evaluate_single_jury(model_response, k, domain, jury["model"], jury["temperature"]))

    jury_results = await asyncio.gather(*tasks)

    # معالجة وحساب المخرجات الإحصائية وهندسة علم الشك
    criteria_list = get_criteria_for_k(k, domain)
    aggregated_metrics = {}
    disputed_criteria = []
    all_reasons = {}

    for c in criteria_list:
        c_name = c["name"]
        c_scores = []
        all_reasons[c_name] = []

        for idx, res in enumerate(jury_results):
            j_name = JURY_MODELS[idx]["name"]
            if c_name in res["scores"]:
                score_val = res["scores"][c_name]
                c_scores.append(score_val)
                all_reasons[c_name].append({j_name: res["reasons"].get(c_name, "")})

        if c_scores:
            mean_score = statistics.mean(c_scores)
            aggregated_metrics[c_name] = round(mean_score, 2)
            
            # رصد التباعد وعلم الشك إذا تباينت الآراء بأكثر من 1.5 نقطة
            if len(c_scores) > 1 and statistics.stdev(c_scores) > 1.5:
                disputed_criteria.append(c_name)

    # الحساب الرياضي لمعادلة الأوزان والمؤشرات المدمجة
    total_weighted_score = 0.0
    total_weights = 0.0
    for c in criteria_list:
        c_name = c["name"]
        if c_name in aggregated_metrics:
            weight = get_criterion_weight(c, k)
            total_weighted_score += aggregated_metrics[c_name] * weight
            total_weights += weight

    q_k = total_weighted_score / total_weights if total_weights > 0 else 0.0
    w_k = calculate_w_k(k)
    s_k = calculate_s_k(k, t_actual)

    # النقاط الإجمالية النهائية لمحرك الـ AI-Unit الخاص بك
    aiu_score = q_k * w_k * s_k

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    return {
        "engine": "🏆 AI-Unit V10.5",
        "prompt_hash": prompt_hash,
        "domain": domain,
        "k": k,
        "difficulty_reason": diff_reason,
        "tested_model_latency": round(t_actual, 3),
        "aiu_score": round(aiu_score, 4),
        "metrics": aggregated_metrics,
        "disputed_criteria": disputed_criteria,
        "reasons_log": all_reasons,
        "database_active": DB_AVAILABLE
    }

# ---------- نظام إدارة ومستقبلات تليجرام وباقي الـ Endpoints ----------

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, token: Optional[str] = Header(None)):
    if TELEGRAM_WEBHOOK_SECRET and token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden Webhook Secret Token")
        
    try:
        body = await request.json()
    except Exception:
        return {"status": "invalid json"}

    # معالجة رسائل التليجرام والمخرجات المباشرة للبوت
    if "message" in body and "text" in body["message"]:
        chat_id = body["message"]["chat"]["id"]
        text = body["message"]["text"].strip()
        
        # كود تفاعلي لمعالجة الرسائل خلف الكواليس وبث الرد السريع
        print(f"Received Telegram Message from {chat_id}: {text}")
        
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database_connected": DB_AVAILABLE,
        "sqlite_records_count": count_human_scores(),
        "jury_count": len(JURY_MODELS)
    }
