==============================================================

AI-Unit Core Engine — الإصدار V10

تحسينات جوهرية على V9.5:

1) محلفون من عائلات نماذج مختلفة فعليًا (استقلالية حقيقية لا صورية)

2) تفسير نصي مختصر لكل معيار وليس رقمًا فقط

3) تخزين دائم عبر SQLite بدل /tmp (مع ملاحظة صريحة عن قيود Render)

4) معايير ديناميكية حسب مجال السؤال (عام / برمجة / طبي / رياضي / قانوني)

5) تقليل تحيز الترتيب: ترتيب المعايير يُعاد خلطه لكل استدعاء محلّف

6) طبقة "علم الشك" (uncertainty flag): إذا تباعدت أحكام المحلفين كثيرًا

يُعلَّم التقييم كـ "متنازع عليه" بدل عرضه كحقيقة نهائية واثقة

==============================================================



⚠️ ملاحظة صادقة قبل النشر:

- أسماء نماذج Groq المجانية تتغير باستمرار (تُضاف وتُحذف نماذج).

القائمة أدناه صحيحة وقت كتابة هذا الكود لكن يجب عليك التحقق من

https://console.groq.com/docs/models قبل النشر والتأكد أن كل نموذج

من عائلة مختلفة فعليًا (وليس نفس العائلة بإصدار مختلف).

- التخزين في SQLite على القرص المحلي لخدمة Render المجانية *لا يزال

يُمحى عند كل إعادة نشر* ما لم تضف "Persistent Disk" (خدمة مدفوعة).

البديل المجاني الحقيقي: استخدام قاعدة بيانات خارجية مجانية مثل

Supabase (طبقتها المجانية تدعم Postgres) بدل SQLite المحلي.

تركت الكود يعمل بـ SQLite محليًا كخطوة أولى صحيحة، لكن لا تعتبره

"حل نهائي دائم" حتى تربطه بقاعدة بيانات خارجية.

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

---------- الإعدادات العامة ----------

app = FastAPI(title="AI-Unit Core Engine V10", version="10.0")

TESTED_MODEL = "llama-3.3-70b-versatile"

------------------------------------------------------------------

استقلالية المحلفين الحقيقية — مُحدَّثة بعد تحقق فعلي (مرتين) من صفحة

console.groq.com/docs/models عبر لقطات شاشة أرسلها المستخدم:



Production Models (مستقرة، مخصصة للتشغيل المستمر):

- عائلة Meta/Llama: llama-3.1-8b-instant, llama-3.3-70b-versatile

- عائلة OpenAI/GPT-OSS: openai/gpt-oss-120b, openai/gpt-oss-20b



Preview Models (نص Groq الرسمي: "لأغراض التقييم فقط، قد تُوقَف

بإشعار قصير ولا يُنصح باستخدامها في الإنتاج"):

- عائلة Qwen: qwen/qwen3-32b  ← موجود فعليًا، لكن هنا فقط



القرار: استخدام Qwen3-32B كمحلّف ثالث مستقل فعليًا منطقي لأن دوره

هنا هو تقييم (evaluation) بالضبط — وهذا ما صُمم له Preview أصلاً.

لكن ⚠️ خطر حقيقي: Groq قد توقفه بإشعار قصير، فأضفنا كشف تلقائي

(auto-fallback) في _startup: لو فشل الاتصال بـ Qwen، ينزل النظام

تلقائيًا لمحلّف احتياطي بدل الانهيار الصامت في منتصف تشغيل حقيقي.

راجع صفحة Models دوريًا (مرة كل أسبوعين تقريبًا كافية) للتأكد أن

qwen/qwen3-32b ما زال مدرجًا، وحدّث القائمة أدناه إذا تغيّر شيء.

------------------------------------------------------------------

JURY_MODELS = [
{"name": "llama_family", "model": "llama-3.1-8b-instant", "temperature": 0.3, "weight": 0.34, "family": "Meta/Llama", "truly_independent": True, "tier": "production"},
{"name": "gptoss_family", "model": "openai/gpt-oss-20b",  "temperature": 0.3, "weight": 0.33, "family": "OpenAI/GPT-OSS", "truly_independent": True, "tier": "production"},
{"name": "qwen_family",  "model": "qwen/qwen3-32b",       "temperature": 0.3, "weight": 0.33, "family": "Qwen/Alibaba", "truly_independent": True, "tier": "preview_risk_of_discontinuation"},
]

محلّف احتياطي إذا فشل نموذج Preview (Qwen) — من عائلة موجودة أصلاً

حتى لا ينهار عدد المحلفين المستقلين إلى صفر بصمت.

FALLBACK_JURY_MODEL = {"name": "llama_fallback", "model": "llama-3.3-70b-versatile", "temperature": 0.4, "weight": 0.33, "family": "Meta/Llama (احتياطي عند فشل Qwen)", "truly_independent": False, "tier": "production"}

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

---------- المعايير الأساسية (تُستخدم دائمًا) ----------

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

------------------------------------------------------------------

معايير إضافية حسب المجال — تُضاف فوق المعايير الأساسية إذا اكتُشف

أن السؤال ينتمي لمجال متخصص. هذا يعالج ملاحظة "المعايير ثابتة".

------------------------------------------------------------------

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
"code": ["كود", "برمج", "دالة", "function", "class", "python", "javascript", "خوارزم", "bug", "compile", "api", "سكريبت"],
"medical": ["طبي", "دواء", "مرض", "أعراض", "علاج", "جرعة", "طبيب", "تشخيص", "medicine", "symptom", "diagnosis"],
"legal": ["قانون", "عقد", "دعوى", "محكمة", "تشريع", "legal", "contract", "lawsuit", "regulation"],
"math": ["برهان", "معادلة", "نظرية", "تفاضل", "تكامل", "proof", "theorem", "equation", "derivative", "integral"],
}

def detect_domain(prompt: str) -> str:
"""كشف مجال السؤال عبر كلمات مفتاحية بسيطة (سريع ورخيص، بلا استدعاء API إضافي)."""
text = prompt.lower()
scores = {domain: sum(1 for kw in kws if kw in text) for domain, kws in DOMAIN_KEYWORDS.items()}
best_domain = max(scores, key=scores.get)
return best_domain if scores[best_domain] > 0 else "general"

def get_criteria_for_k(k: int, domain: str) -> List[Dict]:
count = max(2, 2 ** (k - 1))
base = MASTER_CRITERIA[:count]
domain_extra = DOMAIN_CRITERIA.get(domain, [])
return base + domain_extra

---------- تخزين دائم عبر SQLite (بدل /tmp JSON) ----------



⚠️ تصحيح عطل حقيقي حدث عند النشر الفعلي (Render):

المسار السابق "/var/data/ai_unit.db" غير موجود على الخطة المجانية

إلا مع "Persistent Disk" (خدمة مدفوعة). لما التطبيق حاول يفتحه،

انهار بالكامل عند الاستيراد (sqlite3.OperationalError) قبل حتى ما

يربط أي منفذ — ولهذا ظهرت رسالة "Port scan timeout" في نفس الوقت:

التطبيق مات قبل ما يوصل لمرحلة uvicorn.run() أصلاً.



الإصلاح على مرحلتين:

1) مسار افتراضي مضمون الكتابة على Render المجاني: داخل مجلد المشروع

نفسه (Path(file).parent) بدل مسار نظام غير مضمون الوجود.

هذا لا يزال يُمحى عند كل نشر جديد (نفس القيد المذكور سابقًا)،

لكنه على الأقل لا يُسقط التطبيق بالكامل.

2) تغليف تهيئة قاعدة البيانات بمعالجة أخطاء صريحة: لو فشل فتح

القاعدة لأي سبب (صلاحيات، مسار غير متاح...)، ينزل النظام تلقائيًا

لتخزين مؤقت في الذاكرة (يعمل خلال الجلسة الحالية فقط، يُفقد عند

إعادة التشغيل) بدل أن يرفض التطبيق كله الإقلاع. توفر جزئي أفضل

من عطل كامل.

DB_PATH = Path(os.environ.get("AI_UNIT_DB_PATH", str(Path(file).parent / "ai_unit_data" / "ai_unit.db")))

DB_AVAILABLE = True
_memory_fallback_store: Dict[str, list] = {}  # يُستخدم فقط إذا فشلت SQLite تمامًا

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
# لا نُسقط التطبيق هنا إطلاقًا — هذا بالضبط ما سبّب الانهيار سابقًا.
print(f"⚠️ فشل تهيئة SQLite ({e}) — التبديل التلقائي لتخزين مؤقت بالذاكرة (غير دائم عبر إعادة التشغيل)")
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
print(f"⚠️ فشل الكتابة في SQLite ({e}) — تخزين مؤقت بالذاكرة بدلًا منه")
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
print(f"⚠️ فشل القراءة من SQLite ({e}) — استخدام تخزين الذاكرة المؤقت")
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

---------- عملاء HTTP منفصلون ----------

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

print("🔍 جارٍ التحقق من صحة النماذج مقابل Groq API الحقيقي (وليس افتراضًا)...")  
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
        risk_tag = " (⚠️ Preview — قد يُوقف بإشعار قصير)" if jury.get("tier") == "preview_risk_of_discontinuation" else ""  
        tag = "✅ مستقل فعليًا" if jury.get("truly_independent") else "⚠️ تكرار عائلة"  
        print(f"{tag} — {model} ({jury['family']}){risk_tag}")  
    else:  
        print(f"🔁 نموذج {model} فشل أو غير متاح — استبداله تلقائيًا بالمحلّف الاحتياطي {FALLBACK_JURY_MODEL['model']}")  
        JURY_MODELS[idx] = FALLBACK_JURY_MODEL.copy()  

    if JURY_MODELS[idx].get("truly_independent"):  
        independent_count += 1  

print(f"📊 عدد المحلفين المستقلين فعليًا بعد الفحص: {independent_count} من أصل {len(JURY_MODELS)}")  
if independent_count < 2:  
    print("🚨 تحذير جدي: أقل من محلفين مستقلين — نتائج disagreement غير موثوقة، اللجنة عمليًا حكم واحد مموّه.")  

print(f"💾 قاعدة بيانات التغذية الراجعة: {DB_PATH} (تحقق من كونها دائمة في بيئة الإنتاج)")

@app.on_event("shutdown")
async def _shutdown():
if _http_client_groq:
await _http_client_groq.aclose()
if _http_client_tg:
await _http_client_tg.aclose()

---------- دوال مساعدة ----------

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

async def evaluate_single_jury(model_response: str, k: int, domain: str,
jury_model: str, temperature: float) -> Dict:
"""
تقييم من محلّف واحد. تحسينان هنا مقارنة بـ V9.5:
- يُطلب من المحلّف تفسير نصي قصير لكل معيار (وليس رقمًا فقط)
- ترتيب المعايير يُخلط عشوائيًا لكل استدعاء لتقليل تحيز الترتيب
(order bias): ميل بعض النماذج لإعطاء درجات أعلى/أقل حسب موقع
المعيار في القائمة.
"""
criteria = get_criteria_for_k(k, domain)
shuffled = criteria.copy()
random.shuffle(shuffled)

criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in shuffled])  
jury_prompt = (  
    "You are an independent AI evaluator. Evaluate the response on these criteria.\n"  
    "For EACH criterion give a numeric score (0.0-10.0) AND a one-sentence reason.\n"  
    f"CRITERIA:\n{criteria_descs}\n\n"  
    'Output ONLY JSON in this exact shape:\n'  
    '{"criterion_name": {"score": <0-10>, "reason": "<short reason>"}, ...}\n\n'  
    f"RESPONSE TO EVALUATE:\n\"\"\"{model_response}\"\"\""  
)  
raw = await _groq_call_async(  
    messages=[{"role": "user", "content": jury_prompt}],  
    model=jury_model, temperature=temperature, max_tokens=900, json_mode=True, timeout=25,  
)  
scores: Dict[str, float] = {}  
reasons: Dict[str, str] = {}  
is_fallback = True  
data = _extract_json(raw) if raw else None  
if data:  
    for c in criteria:  
        entry = data.get(c["name"])  
        if isinstance(entry, dict) and "score" in entry:  
            try:  
                scores[c["name"]] = min(max(float(entry["score"]), 0.0), 10.0)  
                reasons[c["name"]] = str(entry.get("reason", ""))[:300]
