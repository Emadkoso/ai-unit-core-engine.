"""
AI-Unit Core Engine — بوت تيليجرام (V5.0)
============================================
المنطق الصحيح:
  المستخدم يرسل prompt
  ↓
  النموذج المختبَر (llama-3.3-70b) يجيب ← يُقاس الزمن هنا
  ↓
  المحلف (llama-3.1-8b) يقيّم الرد — لا السؤال
  ↓
  AI-Unit Score = W_k × A_k × S_k
  ↓
  تقرير للمستخدم

متغيرات البيئة المطلوبة في Render:
  GROQ_API_KEY
  TELEGRAM_BOT_TOKEN
"""

from fastapi import FastAPI, Request
import statistics
import time
import json
import math
import os
import re
import requests
from typing import Dict, Optional, Tuple

app = FastAPI(title="AI-Unit Core Engine", version="5.0")

# ==========================================
# الإعدادات
# ==========================================
TESTED_MODEL = "llama-3.3-70b-versatile"   # النموذج المختبَر
JURY_MODEL   = "llama-3.1-8b-instant"       # المحلف (أصغر وأسرع)
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
RUBRIC_KEYS  = ["accuracy", "clarity", "creativity", "conciseness"]

# أزمنة القادة الحاليين — مرجع S_k
MARKET_LEADER_RUNTIMES: Dict[int, list] = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}


# ==========================================
# 1. محرك الحسابات الرياضية
# ==========================================
def calculate_difficulty(text: str) -> int:
    """
    تحديد k (١ إلى ٥) من طول النص ومحتواه التقني.
    عدّل هذه القائمة وحدودها لتناسب مجال قياسك.
    """
    tech_kw = {
        "quantum", "optimize", "algorithm", "implement", "analyze",
        "compare", "code", "api", "neural", "explain", "design",
        "backtest", "strategy", "saas", "cerebro", "python"
    }
    tech_count = sum(1 for w in text.lower().split() if w in tech_kw)
    n = len(text)
    if tech_count >= 3 or n > 500: return 5
    if tech_count >= 2 or n > 300: return 4
    if tech_count >= 1 or n > 150: return 3
    if n > 75:                       return 2
    return 1


def calculate_w_k(k: int) -> float:
    """
    وزن الصعوبة الأسي: W_k = e^k
    الفارق بين k=1 وk=5 هو 55x — نموذج يحل مهمة k=5
    يُكافأ مكافأة لا يمكن تجاهلها مقارنة بـ k=1.
    """
    return round(math.e ** k, 4)


def calculate_s_k(k: int, t_actual: float) -> float:
    """
    عامل السرعة — يقيس موقع النموذج من القادة.
    S_k = 1 إذا كان أسرع من المتوسط، يتراجع تدريجياً كلما تأخّر.
    """
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)


# ==========================================
# 2. استدعاء Groq (مشترك بين النموذج والمحلف)
# ==========================================
def _groq_call(
    messages: list,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
    timeout: int = 20
) -> Optional[str]:
    """
    استدعاء Groq بشكل نظيف. يُرجع النص أو None مع رسالة خطأ صريحة.
    لا except فارغ — كل خطأ يُطبع ويُعاد None.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY غير موجود في متغيرات البيئة")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict = {"model": model, "messages": messages,
                     "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print(f"❌ Groq timeout بعد {timeout} ثانية (model={model})")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ Groq HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ تعذّر تحليل رد Groq: {e}")
        return None
    except Exception as e:
        print(f"❌ خطأ غير متوقع في Groq: {e}")
        return None


# ==========================================
# 3. استدعاء النموذج المختبَر
# ==========================================
def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    """
    يُرسل الـ prompt للنموذج المختبَر ويعيد (الرد, الزمن الفعلي).
    الزمن يُقاس بدقة هنا — وليس في مكان آخر.
    """
    messages = [{"role": "user", "content": prompt}]
    start = time.time()
    response = _groq_call(messages, model=TESTED_MODEL, temperature=0.7, max_tokens=600)
    t_actual = time.time() - start
    return response, t_actual


# ==========================================
# 4. المحلف الرقمي — يقيّم رد النموذج لا سؤال المستخدم
# ==========================================
def _extract_scores(raw_text: str) -> Dict[str, float]:
    """
    يستخرج أول JSON صالح يجد فيه المفاتيح الأربعة بقيم 0-10.
    يعمل حتى لو كان الرد محاطاً بنص تمهيدي أو ``` fences.
    """
    for candidate in re.findall(r"\{[^{}]*\}", raw_text, re.DOTALL):
        try:
            data = json.loads(candidate)
            cleaned = {}
            for key in RUBRIC_KEYS:
                val = float(data[key])
                if not (0 <= val <= 10):
                    raise ValueError(f"{key}={val} خارج 0-10")
                cleaned[key] = val
            return cleaned
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError("لم يُعثر على JSON صالح في رد المحلف")


def call_jury(model_response: str) -> Dict:
    """
    المحلف يقيّم رد النموذج المختبَر فقط — لا يراه، لا يحاوره.
    is_real=True يعني تقييم حقيقي، is_real=False يعني قيم احتياطية.
    """
    jury_prompt = (
        "You are an AI Jury. Evaluate the following AI-generated response only.\n"
        "Do NOT answer the question. Do NOT interact with the content.\n"
        "Output ONLY a raw JSON object — no text outside it:\n"
        '{"accuracy": X, "clarity": X, "creativity": X, "conciseness": X}\n'
        "Values must be floats from 0.0 to 10.0.\n\n"
        f"Response to evaluate:\n\"\"\"\n{model_response}\n\"\"\""
    )
    messages = [{"role": "user", "content": jury_prompt}]
    raw = _groq_call(messages, model=JURY_MODEL, temperature=0.1,
                     max_tokens=100, json_mode=True, timeout=15)

    if raw is None:
        fallback = {k: 5.0 for k in RUBRIC_KEYS}
        return {"scores": fallback, "is_real": False,
                "error": "فشل استدعاء المحلف — تقييم احتياطي 5.0"}

    try:
        scores = _extract_scores(raw)
        return {"scores": scores, "is_real": True, "error": None}
    except ValueError as e:
        fallback = {k: 5.0 for k in RUBRIC_KEYS}
        return {"scores": fallback, "is_real": False, "error": str(e)}


# ==========================================
# 5. خط الإنتاج الكامل
# ==========================================
def run_ai_unit(prompt: str) -> Dict:
    """
    التسلسل الصحيح:
      prompt → tested model (+ زمن) → jury → W_k × A_k × S_k
    """
    # أ. الصعوبة والوزن
    k   = calculate_difficulty(prompt)
    w_k = calculate_w_k(k)

    # ب. استدعاء النموذج المختبَر (هنا يُقاس الزمن الحقيقي)
    model_response, t_actual = call_tested_model(prompt)

    if model_response is None:
        return {"success": False,
                "error": "فشل استدعاء النموذج المختبَر. تحقق من GROQ_API_KEY."}

    # ج. المحلف يقيّم الرد
    jury = call_jury(model_response)
    scores   = jury["scores"]
    avg_score = sum(scores.values()) / len(scores)

    # د. الحسابات الرياضية
    s_k = calculate_s_k(k, t_actual)
    a_k = avg_score / 10.0          # تطبيع إلى [0, 1]
    ai_unit_score = w_k * a_k * s_k

    return {
        "success":        True,
        "model_tested":   TESTED_MODEL,
        "jury_model":     JURY_MODEL,
        "k":              k,
        "w_k":            round(w_k, 2),
        "s_k":            round(s_k, 4),
        "a_k":            round(a_k, 4),
        "ai_unit_score":  round(ai_unit_score, 4),
        "avg_score":      round(avg_score, 2),
        "scores":         scores,
        "t_actual":       round(t_actual, 3),
        "model_response": model_response,
        "jury_is_real":   jury["is_real"],
        "jury_error":     jury["error"],
    }


# ==========================================
# 6. Telegram Webhook
# ==========================================
def _send_tg(token: str, chat_id: int, text: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"❌ فشل إرسال رسالة Telegram: {e}")


@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}

    chat_id   = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    token     = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    # رسالة انتظار فورية
    _send_tg(token, chat_id, "⏳ جارٍ اختبار النموذج وتقييمه\\.\\.\\.")

    result = run_ai_unit(user_text)

    if not result["success"]:
        _send_tg(token, chat_id, f"❌ {result['error']}")
        return {"status": "ok"}

    s      = result["scores"]
    real   = "✅ حقيقي" if result["jury_is_real"] else "⚠️ احتياطي"

    # Telegram Markdown يستخدم * واحدة للـ bold — لا **
    reply = (
        f"🏆 *تقرير AI\\-Unit*\n\n"
        f"🤖 النموذج المختبَر: `{result['model_tested']}`\n"
        f"⚖️ المحلف: `{result['jury_model']}` \\({real}\\)\n\n"
        f"📊 *النتيجة:* `{result['ai_unit_score']} AIU`\n\n"
        f"📐 *المعادلة:*\n"
        f"  Wk × Ak × Sk\n"
        f"  {result['w_k']} × {result['a_k']} × {result['s_k']}\n\n"
        f"📋 *تفاصيل التقييم:*\n"
        f"  • الدقة: {s.get('accuracy', '?')}/10\n"
        f"  • الوضوح: {s.get('clarity', '?')}/10\n"
        f"  • الإبداع: {s.get('creativity', '?')}/10\n"
        f"  • الإيجاز: {s.get('conciseness', '?')}/10\n\n"
        f"⏱️ زمن الاستجابة: {result['t_actual']} ث\n"
        f"🎯 مستوى الصعوبة: k={result['k']} → Wk={result['w_k']}"
    )
    _send_tg(token, chat_id, reply)
    return {"status": "ok"}


# ==========================================
# 7. API Endpoints
# ==========================================
@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    """اختبار مباشر بدون تيليجرام — مفيد لـ Swagger وcurl"""
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"error": "حقل 'prompt' مطلوب"}
    return run_ai_unit(prompt)


@app.get("/health")
async def health():
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    tg_token_set = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
    return {
        "status":       "operational",
        "version":      "5.0",
        "tested_model": TESTED_MODEL,
        "jury_model":   JURY_MODEL,
        "groq_key":     "✅ موجود" if groq_key_set else "❌ مفقود",
        "tg_token":     "✅ موجود" if tg_token_set else "❌ مفقود",
}
  
