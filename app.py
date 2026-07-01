# ==============================================================
# AI-Unit Core Engine — بوت تيليجرام (V8.0 - Optimized)
# الملف: app.py
# ==============================================================
# الميزات الجديدة المدمجة:
#   • زيادة الـ max_tokens للمحلّف إلى 1000 لمنع انقطاع الـ JSON في k=5
#   • رفع الـ timeout للنموذج المختبر إلى 40 ثانية للمهام البرمجية المعقدة
#   • تنسيق رقم النتيجة النهائية بجمالية احترافية لسهولة القراءة
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
app = FastAPI(title="AI-Unit Core Engine", version="8.0")

TESTED_MODEL = "llama-3.3-70b-versatile"   # النموذج المختبر
JURY_MODEL   = "llama-3.1-8b-instant"      # النموذج المحلِّف (لتقدير k والتقييم)
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# أوقات السوق القيادية (للحصول على S_k)
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
    {"name": "creativity", "desc": "هل يقدم حلولاً مبتكرة أو زوايا جديدة？", "weight": "semi_exp"},
    {"name": "safety", "desc": "هل يتجنب التحيز، الكراهية، أو الضرر؟", "weight": "linear"},
    # k=5 (ستة عشر معياراً – الانفجار الكامل)
    {"name": "strategy", "desc": "هل يقدم خطة عمل قابلة للتنفيذ استراتيجياً؟", "weight": "exp"},
    {"name": "predictive_power", "desc": "هل يتنبأ بالنتائج أو التحديات المستقبلية بدقة؟", "weight": "semi_exp"},
    {"name": "critical_analysis", "desc": "هل حلل الفرضيات ونقدها بناءً على أدلة؟", "weight": "semi_exp"},
    {"name": "originality", "desc": "هل الإجابة جديدة كلياً وغير موجودة في النماذج الأخرى؟", "weight": "semi_exp"},
    {"name": "fallacy_detection", "desc": "هل اكتشف المغالطات المنطقية في السؤال نفسه؟", "weight": "exp"},
    {"name": "rhetorical_beauty", "desc": "هل الصياغة لغوياً بليغة ومؤثرة؟", "weight": "linear"},
    {"name": "adaptability", "desc": "هل يتكيف الجواب مع سياقات أو جماهير مختلفة؟", "weight": "linear"},
    {"name": "generative_power", "desc": "هل يولد معرفة جديدة أم يعيد تدوير المعرفة القديمة؟", "weight": "semi_exp"},
]


# ---------- 1. استدعاء Groq (مشترك) ----------
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
    except requests.exceptions.Timeout:
        print(f"❌ Groq timeout {timeout}s (model={model})")
        return None
    except requests.exceptions.HTTPError:
        print(f"❌ Groq HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ تعذّر تحليل رد Groq: {e}")
        return None
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        return None


# ---------- 2. تقدير الصعوبة (k) بواسطة الذكاء الاصطناعي ----------
def _difficulty_fallback(text: str) -> int:
    """احتياطي بسيط (يُستخدم فقط عند فشل الذكاء الاصطناعي)."""
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75:  return 2
    return 1

def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    """
    يُقرر الذكاء الاصطناعي k من 1 إلى 5 بناءً على الصعوبة المعرفية الفعلية.
    يعيد: (k, السبب, هل الحكم من الذكاء الاصطناعي؟)
    """
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
                raise ValueError(f"k={k} خارج 1-5")
            return k, reason, True
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    # لم يُعثر على JSON صالح
    k = _difficulty_fallback(prompt)
    print(f"⚠️ فشل تحليل JSON من تقدير الصعوبة: {raw[:100]}")
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
    """يستدعي النموذج المختبَر ويعيد (الرد, الزمن الفعلي)."""
    start = time.time()
    # التعديل الثاني: رفع مهلة الانتظار إلى 40 ثانية لضمان استقرار الاستجابات الطويلة
    response = _groq_call(
        messages=[{"role": "user", "content": prompt}],
        model=TESTED_MODEL,
        temperature=0.7,
        max_tokens=600,
        timeout=40
    )
    return response, time.time() - start


# ---------- 5. هيئة المحلفين – التكاثر الأسّي للمعايير ----------
def get_criteria_for_k(k: int) -> List[Dict]:
    """عدد المعايير = 2^(k-1)، نأخذ أول N من MASTER_CRITERIA."""
    k = max(1, min(k, 5))
    count = 2 ** (k - 1)
    return MASTER_CRITERIA[:count]

def get_criterion_weight(criterion: Dict, k: int) -> float:
    """ترجيح أسّي أو خطي حسب نوع المعيار."""
    w_type = criterion["weight"]
    if w_type == "exp":
        return math.e ** k
    elif w_type == "semi_exp":
        return math.e ** (k / 2)
    else:  # linear
        return float(k)

def call_jury_exponential(model_response: str, k: int) -> Dict:
    """
    المحلِّف يُقيّم الرد بناءً على المعايير المتضاعفة أسّياً.
    يُرسل جميع المعايير دفعة واحدة في برومت واحد.
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

    # التعديل الأول: رفع الـ max_tokens إلى 1000 لمنع انقطاع مخرجات الـ JSON الطويلة في المستويات العالية
    raw = _groq_call(
        messages=[{"role": "user", "content": jury_prompt}],
        model=JURY_MODEL,
        temperature=0.1,
        max_tokens=1000,
        json_mode=True,
        timeout=20
    )

    if raw is None:
        # احتياطي قاسٍ: جميع الدرجات = 1.0 (عقاب شديد)
        return {"scores": {c["name"]: 1.0 for c in criteria},
                "is_real": False,
                "error": "فشل استدعاء المحلِّف"}

    try:
        # استخراج JSON من الرد (قد يحتوي على فاصلة أو علامات)
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("لم يُعثر على JSON")
        json_str = raw[json_start:json_end]
        scores = json.loads(json_str)
        # التأكد من وجود جميع المفاتيح بالقيم المناسبة
        final_scores = {}
        for c in criteria:
            name = c["name"]
            val = scores.get(name)
            if val is None:
                val = 1.0
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 1.0
            final_scores[name] = min(max(val, 0.0), 10.0)
        return {"scores": final_scores, "is_real": True, "error": None}
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"⚠️ فشل تحليل JSON من المحلِّف: {e} | raw: {raw[:200]}")
        return {"scores": {c["name"]: 1.0 for c in criteria},
                "is_real": False,
                "error": f"JSON غير صالح: {str(e)}"}


# ---------- 6. خط الإنتاج الرئيسي (Master Pipeline) ----------
def run_ai_unit(prompt: str) -> Dict[str, Any]:
    """
    التسلسل الكامل:
      أ. تقدير k بواسطة الذكاء الاصطناعي
      ب. استدعاء النموذج المختبَر (قياس الزمن)
      ج. المحلِّف يُقيّم باستخدام المعايير المتضاعفة أسّياً
      د. حساب النتيجة النهائية: Σ (Score_i^k × Weight_i) × S_k
    """
    # أ. تقدير الصعوبة
    k, k_reason, k_is_real = assess_difficulty(prompt)
    w_k = calculate_w_k(k)

    # ب. استدعاء النموذج المختبَر
    model_response, t_actual = call_tested_model(prompt)
    if model_response is None:
        return {"success": False,
                "error": "فشل استدعاء النموذج المختبَر — تحقق من GROQ_API_KEY"}

    # ج. المحلِّف (معايير متضاعفة)
    jury = call_jury_exponential(model_response, k)
    scores = jury["scores"]
    # حساب متوسط الدرجات (لأغراض العرض)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    # د. تجميع النتيجة باستخدام التكاثر الأسّي
    total_weighted_score = 0.0
    criterion_details = []
    for name, score in scores.items():
        # العثور على تعريف المعيار من القائمة الأم
        criterion_data = next((c for c in MASTER_CRITERIA if c["name"] == name), None)
        if criterion_data:
            weight = get_criterion_weight(criterion_data, k)
        else:
            weight = 1.0
        # (درجة)^k × الوزن
        contribution = (score ** k) * weight
        total_weighted_score += contribution
        criterion_details.append({
            "name": name,
            "score": round(score, 2),
            "weight": round(weight, 4),
            "contribution": round(contribution, 4)
        })

    # هـ. تطبيق عامل السرعة
    s_k = calculate_s_k(k, t_actual)
    a_k = avg_score / 10.0  # نستخدم متوسط الدرجات المعياري (0-1) لتقريب A_k
    ai_unit_score = total_weighted_score * s_k

    return {
        "success": True,
        "model_tested": TESTED_MODEL,
        "jury_model": JURY_MODEL,
        # تقدير الصعوبة
        "k": k,
        "k_reason": k_reason,
        "k_assessed_by_ai": k_is_real,
        # المعايير والقيم
        "criteria_count": len(scores),
        "criteria_names": list(scores.keys()),
        "scores": scores,
        "avg_score": round(avg_score, 2),
        "a_k": round(a_k, 4),
        # المعادلة
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


# ---------- 7. Telegram Webhook ----------
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

    _send_tg(token, chat_id, "⏳ جارٍ التقييم باستخدام AI‑Unit V8.0 ...")

    result = run_ai_unit(user_text)

    if not result["success"]:
        _send_tg(token, chat_id, f"❌ {result['error']}")
        return {"status": "ok"}

    s = result["scores"]
    jury_real = "✅" if result["jury_is_real"] else "⚠️ احتياطي"
    k_real    = "🧠 ذكاء اصطناعي" if result["k_assessed_by_ai"] else "⚠️ احتياطي"

    # بناء الرد المُفصّل
    criteria_lines = "\n".join([
        f"  • {name}: {score:.1f}/10" for name, score in s.items()
    ])

    # التعديل الثالث: إضافة فواصل الآلاف وتثبيت خانتين عشريتين لتنسيق النتيجة النهائية الفلكية بجمالية فائقة
    reply = (
        f"🏆 *AI-Unit Core Engine V8.0*\n"
        f"——————————————————\n"
        f"🤖 النموذج: `{result['model_tested']}`\n\n"
        f"🎯 *مستوى الصعوبة:* k={result['k']} ({k_real})\n"
        f"💬 السبب: _{result['k_reason']}_\n\n"
        f"📊 *عدد المعايير المتضاعفة:* {result['criteria_count']} (2^({result['k']-1}))\n"
        f"📋 *تفاصيل التقييم:*\n{criteria_lines}\n\n"
        f"📐 *المعادلة:* `W_k × S_k × Σ(Score_i^k × Weight_i)`\n"
        f"  • W_k = e^{result['k']} = {result['w_k']}\n"
        f"  • S_k = {result['s_k']}\n"
        f"  • المجموع الموزون = {result['total_weighted_sum']}\n\n"
        f"🏅 *النتيجة النهائية:* `{result['ai_unit_score']:,.2f} AIU`\n"
        f"⏱️ زمن الاستجابة: {result['t_actual']} ث\n"
        f"——————————————————\n"
        f"⚖️ المحلِّف: {jury_real}"
    )
    _send_tg(token, chat_id, reply)
    return {"status": "ok"}


# ---------- 8. نقاط النهاية الإضافية ----------
@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"error": "حقل 'prompt' مطلوب"}
    return run_ai_unit(prompt)

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "8.0",
        "tested_model": TESTED_MODEL,
        "jury_model": JURY_MODEL,
        "max_criteria": len(MASTER_CRITERIA),
        "proliferation_formula": "count = 2^(k-1)",
        "weight_formula": "W_k = e^k",
        "groq_key": "✅" if os.environ.get("GROQ_API_KEY") else "❌ مفقود",
        "tg_token": "✅" if os.environ.get("TELEGRAM_BOT_TOKEN") else "❌ مفقود",
    }

# ==============================================================
# تشغيل السيرفر محلياً أو عبر البيئة السحابية
# ==============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
