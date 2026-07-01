# ==============================================================
# AI-Unit Core Engine — بوت تيليجرام (V8.5 - Hybrid Edition)
# الملف: app.py
# ==============================================================
# المعمارية المتقدمة المدمجة:
#   • تقدير الصعوبة (k) ذكياً بواسطة الذكاء الاصطناعي (1-5).
#   • نظام المحلّف الهجين (Hybrid Jury): Groq للمستويات السهلة والمتوسطة (k <= 3)،
#     ويترقى تلقائياً إلى OpenRouter (Claude-3.5) للمهمات المعقدة (k >= 4).
#   • معالجة حذرة ومقاومة للانقطاع: max_tokens=1000 للمحلّف، وtimeout=40 للنموذج المختبر.
#   • تنسيق رقمي فخم للنتيجة النهائية لسهولة القراءة في تلغرام.
# ==============================================================

from fastapi import FastAPI, Request
import statistics
import time
import json
import math
import os
import re
import requests
from typing import Dict, Optional, List, Any, Tuple

# ---------- الإعدادات العامة ----------
app = FastAPI(title="AI-Unit Core Engine", version="8.5")

TESTED_MODEL     = "llama-3.3-70b-versatile"   # النموذج المختبر (عبر Groq)
JURY_MODEL       = "llama-3.1-8b-instant"      # المحلّف القياسي للمهمات العادية (عبر Groq)
SUPER_JURY_MODEL = "anthropic/claude-3.5-sonnet" # المحلّف الخارق للمهمات المعقدة (عبر OpenRouter)

GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"

# أوقات السوق القياسية (للحصول على عامل السرعة S_k)
MARKET_LEADER_RUNTIMES: Dict[int, list] = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}

# ---------- قائمة المعايير الأم (Master List) – 16 معياراً ----------
MASTER_CRITERIA = [
    # k=1 (معيار واحد)
    {"name": "accuracy", "desc": "هل الجواب صحيح تماماً وخالٍ من الأخطاء الواقعية؟", "weight": "exp"},
    # k=2 (معياران)
    {"name": "clarity", "desc": "هل الجواب واضح ومباشر دون غموض؟", "weight": "linear"},
    # k=3 (أربعة معايير)
    {"name": "completeness", "desc": "هل غطى جميع جوانب السؤال دون نقص؟", "weight": "linear"},
    {"name": "coherence", "desc": "هل الأفكار مترابطة منطقياً ومتسلسلة؟", "weight": "linear"},
    # k=4 (ثمانية معايير)
    {"name": "depth", "desc": "هل تجاوز السطح إلى الأسباب الجذرية والتحليل العميق؟", "weight": "exp"},
    {"name": "uniqueness", "desc": "هل يقدم وجهة نظر نادرة أو غير مكررة؟", "weight": "semi_exp"},
    {"name": "creativity", "desc": "هل يقدم حلولاً مبتكرة أو زوايا جديدة؟", "weight": "semi_exp"},
    {"name": "safety", "desc": "هل يتجنب التحيز، الكراهية، أو الضرر؟", "weight": "linear"},
    # k=5 (ستة عشر معياراً – الانفجار الكامل)
    {"name": "strategy", "desc": "هل يقدم خطة عمل قابلة للتنفيذ استراتيجياً؟", "weight": "exp"},
    {"name": "predictive_power", "desc": "هل يتنبأ بالنتائج أو التحديات المستقبلية بدقة؟", "weight": "semi_exp"},
    {"name": "critical_analysis", "desc": "هل حلل الفرضيات ونقدها بناءً على أدلة؟", "weight": "semi_exp"},
    {"name": "originality", "desc": "هل الإجابة جديدة كلياً وغير موجودة في النماذج الأخرى？", "weight": "semi_exp"},
    {"name": "fallacy_detection", "desc": "هل اكتشف المغالطات المنطقية في السؤال نفسه؟", "weight": "exp"},
    {"name": "rhetorical_beauty", "desc": "هل الصياغة لغوياً بليغة ومؤثرة؟", "weight": "linear"},
    {"name": "adaptability", "desc": "هل يتكيف الجواب مع سياقات أو جماهير مختلفة؟", "weight": "linear"},
    {"name": "generative_power", "desc": "هل يولد معرفة جديدة أم يعيد تدوير المعرفة القديمة؟", "weight": "semi_exp"},
]


# ---------- 1. دوال استدعاء واجهات البرمجة (APIs) ----------
def _groq_call(
    messages: list,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
    timeout: int = 20
) -> Optional[str]:
    """إرسال طلب إلى Groq API وإرجاع النص."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY غير موجود")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict = {
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
        print(f"❌ خطأ استدعاء Groq ({model}): {e}")
        return None


def _openrouter_call(
    messages: list,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
    timeout: int = 30
) -> Optional[str]:
    """إرسال طلب إلى OpenRouter API وإرجاع النص."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY غير موجود")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-unit.engine",
        "X-Title": "AI-Unit Core Engine"
    }
    payload: Dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ خطأ استدعاء OpenRouter ({model}): {e}")
        return None


# ---------- 2. تقدير الصعوبة (k) بواسطة الذكاء الاصطناعي ----------
def _difficulty_fallback(text: str) -> int:
    """احتياطي بسيط في حال فشل الذكاء الاصطناعي."""
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75:  return 2
    return 1

def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    """يُقرر الذكاء الاصطناعي قيمة k من 1 إلى 5 بناءً على الصعوبة المعرفية الفعلية."""
    difficulty_prompt = (
        "You are an AI difficulty assessor for an LLM benchmarking system.\n"
        "Rate the COGNITIVE difficulty of the following prompt from 1 to 5.\n\n"
        "Scale:\n"
        "  1 = Greeting or trivial one-word answer\n"
        "  2 = Simple factual question with a direct answer\n"
        "  3 = Requires explanation, context, or moderate reasoning\n"
        "  4 = Requires multi-step analysis or domain knowledge\n"
        "  5 = Deep expertise, complex synthesis, or original thinking\n\n"
        "CRITICAL: Judge by actual cognitive demand — NOT by text length or keywords.\n"
        "A short deep question CAN be k=5. A long simple question CAN be k=1.\n\n"
        'Output ONLY this JSON (no text outside it):\n'
        '{"k": <integer 1-5>, "reason": "<one concise sentence>"}\n\n'
        f'Prompt to assess:\n"""\n{prompt}\n"""'
    )

    raw = _groq_call(
        messages=[{"role": "user", "content": difficulty_prompt}],
        model=JURY_MODEL,
        temperature=0.1,
        max_tokens=80,
        json_mode=True,
        timeout=10
    )

    if raw is None:
        k = _difficulty_fallback(prompt)
        return k, "تقدير احتياطي — فشل استدعاء الذكاء الاصطناعي", False

    for candidate in re.findall(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            data = json.loads(candidate)
            k = int(data["k"])
            reason = str(data.get("reason", "")).strip()
            if k not in range(1, 6):
                raise ValueError(f"k={k} خارج النطاق")
            return k, reason, True
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    k = _difficulty_fallback(prompt)
    return k, "تقدير احتياطي — JSON غير صالح", False


# ---------- 3. الحسابات الرياضية الأساسية ----------
def calculate_w_k(k: int) -> float:
    """W_k = e^k (وزن أسّي)"""
    return round(math.e ** k, 4)

def calculate_s_k(k: int, t_actual: float) -> float:
    """S_k = T_target / (T_actual + T_target) — محدودة بين 0 و1."""
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)


# ---------- 4. استدعاء النموذج المختبَر ----------
def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    """يستدعي النموذج المختبَر ويعيد (الرد, الزمن الفعلي) مع مهلة 40 ثانية للمهام الطويلة."""
    start = time.time()
    response = _groq_call(
        messages=[{"role": "user", "content": prompt}],
        model=TESTED_MODEL,
        temperature=0.7,
        max_tokens=600,
        timeout=40
    )
    return response, time.time() - start


# ---------- 5. هيئة المحلفين الهجينة (Dynamic Routing) ----------
def get_criteria_for_k(k: int) -> List[Dict]:
    """عدد المعايير = 2^(k-1)"""
    k = max(1, min(k, 5))
    count = 2 ** (k - 1)
    return MASTER_CRITERIA[:count]

def get_criterion_weight(criterion: Dict, k: int) -> float:
    """ترجيح المعيار حسب نوعه هندسياً."""
    w_type = criterion["weight"]
    if w_type == "exp":
        return math.e ** k
    elif w_type == "semi_exp":
        return math.e ** (k / 2)
    else:
        return float(k)

def call_jury_exponential(model_response: str, k: int) -> Dict:
    """
    المحلِّف الذكي: يوجه الطلب لـ Groq سريجاً إذا كان k <= 3،
    ويستدعي محلفاً خارقاً (Claude) عبر OpenRouter إذا كان k >= 4 مع رفع الـ max_tokens لـ 1000.
    """
    criteria = get_criteria_for_k(k)
    criteria_names = [c["name"] for c in criteria]
    criteria_descs = "\n".join([f"  - {c['name']}: {c['desc']}" for c in criteria])

    jury_prompt = (
        f"You are the ULTIMATE AI Jury for a top-tier benchmarking system.\n"
        f"You must evaluate the response based on EXACTLY {len(criteria)} criteria.\n"
        f"Output a RAW JSON object with these keys: {criteria_names}\n"
        f"Values must be floats from 0.0 to 10.0. Be harsh. 10 is perfection, 0 is useless.\n\n"
        f"CRITERIA:\n{criteria_descs}\n\n"
        f"RESPONSE TO EVALUATE:\n\"\"\"\n{model_response}\n\"\"\""
    )

    # التوجيه الذكي للمحلّف (Dynamic Routing)
    if k >= 4:
        active_jury = SUPER_JURY_MODEL
        raw = _openrouter_call(
            messages=[{"role": "user", "content": jury_prompt}],
            model=SUPER_JURY_MODEL,
            temperature=0.1,
            max_tokens=1000,
            json_mode=True,
            timeout=30
        )
    else:
        active_jury = JURY_MODEL
        raw = _groq_call(
            messages=[{"role": "user", "content": jury_prompt}],
            model=JURY_MODEL,
            temperature=0.1,
            max_tokens=1000,
            json_mode=True,
            timeout=20
        )

    if raw is None:
        return {"scores": {c["name"]: 1.0 for c in criteria},
                "is_real": False,
                "error": "فشل استدعاء المحلِّف",
                "jury_used": active_jury}

    try:
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("لم يُعثر على JSON")
        json_str = raw[json_start:json_end]
        scores = json.loads(json_str)
        
        final_scores = {}
        for c in criteria:
            name = c["name"]
            val = scores.get(name)
            if val is None: val = 1.0
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 1.0
            final_scores[name] = min(max(val, 0.0), 10.0)
        return {"scores": final_scores, "is_real": True, "error": None, "jury_used": active_jury}
    except Exception as e:
        print(f"⚠️ فشل تحليل مخرجات المحلّف: {e}")
        return {"scores": {c["name"]: 1.0 for c in criteria},
                "is_real": False,
                "error": f"JSON غير صالح: {str(e)}",
                "jury_used": active_jury}


# ---------- 6. خط الإنتاج الرئيسي (Master Pipeline) ----------
def run_ai_unit(prompt: str) -> Dict[str, Any]:
    """تنسيق دورة معالجة البيانات وتقييم المعادلة بالكامل."""
    # أ. تقدير الصعوبة k
    k, k_reason, k_is_real = assess_difficulty(prompt)
    w_k = calculate_w_k(k)

    # ب. استدعاء النموذج المختبَر
    model_response, t_actual = call_tested_model(prompt)
    if model_response is None:
        return {"success": False, "error": "فشل استدعاء النموذج المختبَر — تحقق من مفاتيح الربط"}

    # ج. استدعاء المحلّف الموجه ديناميكياً
    jury = call_jury_exponential(model_response, k)
    scores = jury["scores"]
    active_jury_model = jury["jury_used"]
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    # د. حساب النتيجة الكلية بالتكاثر الأسّي للمؤشرات
    total_weighted_score = 0.0
    criterion_details = []
    for name, score in scores.items():
        criterion_data = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        weight = get_criterion_weight(criterion_data, k) if criterion_data else 1.0
        
        # (درجة)^k × الوزن
        contribution = (score ** k) * weight
        total_weighted_score += contribution
        criterion_details.append({
            "name": name,
            "score": round(score, 2),
            "weight": round(weight, 4),
            "contribution": round(contribution, 4)
        })

    # هـ. تطبيق عامل السرعة وعامل الجودة النسبي
    s_k = calculate_s_k(k, t_actual)
    a_k = avg_score / 10.0
    ai_unit_score = total_weighted_score * s_k

    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "jury_model": active_jury_model,
        "k": k,
        "k_reason": k_reason,
        "k_assessed_by_ai": k_is_real,
        "criteria_count": len(scores),
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "a_k": round(a_k, 4),
        "w_k": round(w_k, 4),
        "s_k": round(s_k, 4),
        "total_weighted_sum": round(total_weighted_score, 4),
        "ai_unit_score": round(ai_unit_score, 4),
        "criterion_details": criterion_details,
        "t_actual": round(t_actual, 3),
        "model_response": model_response,
        "jury_is_real": jury["is_real"],
        "jury_error": jury["error"],
    }


# ---------- 7. Telegram Webhook Endpoint ----------
def _send_tg(token: str, chat_id: int, text: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"❌ فشل إرسال Telegram: {e}")

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}

    chat_id   = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    token     = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    _send_tg(token, chat_id, "⏳ جاري التقييم الشامل وضخ البيانات عبر خط المعالجة الهجين...")

    result = run_ai_unit(user_text)

    if not result["success"]:
        _send_tg(token, chat_id, f"❌ {result['error']}")
        return {"status": "ok"}

    s = result["scores"]
    jury_status = "✅" if result["jury_is_real"] else "⚠️ احتياطي (فشل الهيكل الرئيسي)"
    k_status    = "🧠 ذكاء اصطناعي" if result["k_assessed_by_ai"] else "⚠️ احتياطي السلسلة"

    criteria_lines = "\n".join([f"  • {name}: {score:.1f}/10" for name, score in s.items()])

    # بناء الرد مع تطبيق التنسيق الاحترافي للآلاف والمستند الرقمي المطور
    reply = (
        f"🏆 *AI-Unit Core Engine V8.5*\n"
        f"——————————————————\n"
        f"🤖 النموذج المختبَر: `{result['model_tested']}`\n\n"
        f"🎯 *مستوى الصعوبة المتوقع:* k={result['k']} ({k_status})\n"
        f"💬 مبرر التوجيه: _{result['k_reason']}_\n\n"
        f"📊 *المعايير المفعلة أسّياً:* {result['criteria_count']} معيار تقييم\n"
        f"📋 *بيان مخرجات التحكيم:*\n{criteria_lines}\n\n"
        f"📐 *الهندسة الرياضية للحساب:* `W_k × S_k × Σ(Score_i^k × Weight_i)`\n"
        f"  • W_k (الوزن الأسّي) = {result['w_k']}\n"
        f"  • S_k (عامل الاستجابة السريعة) = {result['s_k']}\n"
        f"  • المجموع الموزون الخام = {result['total_weighted_sum']:,.2f}\n\n"
        f"🏅 *النتيجة النهائية الخالصة:* `{result['ai_unit_score']:,.2f} AIU`\n"
        f"⏱️ زمن الاستجابة الفعلي: {result['t_actual']} ثانية\n"
        f"——————————————————\n"
        f"⚖️ المحلِّف النشط: `{result['jury_model']}` {jury_status}"
    )
    
    _send_tg(token, chat_id, reply)
    return {"status": "ok"}


# ---------- 8. الواجهات البرمجية العامة (API Endpoints) ----------
@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"error": "حقل 'prompt' مطلوب لتوليد التقييم"}
    return run_ai_unit(prompt)

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "8.5",
        "tested_model": TESTED_MODEL,
        "standard_jury": JURY_MODEL,
        "super_jury": SUPER_JURY_MODEL,
        "groq_key": "✅ متصل" if os.environ.get("GROQ_API_KEY") else "❌ مفقود",
        "openrouter_key": "✅ متصل" if os.environ.get("OPENROUTER_API_KEY") else "❌ مفقود",
        "telegram_token": "✅ متصل" if os.environ.get("TELEGRAM_BOT_TOKEN") else "❌ مفقود"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
