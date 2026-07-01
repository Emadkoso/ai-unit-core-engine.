"""
AI-Unit Core Engine — بوت تيليجرام (V6.0)
============================================
المسار الكامل:
  prompt المستخدم
    ↓
  [ذكاء اصطناعي] يقرر k والسبب ← جديد في V6
    ↓
  [النموذج المختبَر] يجيب ← يُقاس الزمن هنا
    ↓
  [المحلف] يقيّم الرد — لا السؤال
    ↓
  W_k(=e^k) × A_k × S_k = AI-Unit Score

متغيرات البيئة في Render:
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

app = FastAPI(title="AI-Unit Core Engine", version="6.0")

# ==========================================
# الإعدادات
# ==========================================
TESTED_MODEL = "llama-3.3-70b-versatile"
JURY_MODEL   = "llama-3.1-8b-instant"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
RUBRIC_KEYS  = ["accuracy", "clarity", "creativity", "conciseness"]

MARKET_LEADER_RUNTIMES: Dict[int, list] = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}


# ==========================================
# 1. استدعاء Groq — مشترك بين كل الوحدات
# ==========================================
def _groq_call(
    messages: list,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
    timeout: int = 20
) -> Optional[str]:
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


# ==========================================
# 2. تقدير الصعوبة — ذكاء اصطناعي لا قواعد يدوية
# ==========================================
def _difficulty_fallback(text: str) -> int:
    """يُستخدم فقط إذا فشل استدعاء الذكاء الاصطناعي."""
    n = len(text)
    if n > 500: return 5
    if n > 300: return 4
    if n > 150: return 3
    if n > 75:  return 2
    return 1


def assess_difficulty(prompt: str) -> Tuple[int, str, bool]:
    """
    الذكاء الاصطناعي يقرر k من ١ إلى ٥ بناءً على الصعوبة الفعلية،
    لا على طول النص أو كلمات مفتاحية مبرمجة مسبقاً.

    سؤال قصير عميق = k عالي.
    سؤال طويل بسيط = k منخفض.

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


# ==========================================
# 3. الحسابات الرياضية
# ==========================================
def calculate_w_k(k: int) -> float:
    """W_k = e^k — معادلة أسية حقيقية."""
    return round(math.e ** k, 4)


def calculate_s_k(k: int, t_actual: float) -> float:
    """S_k = T_target / (T_actual + T_target) — محدودة بين 0 و1."""
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [float(k * 1.5)]))
    return min(t_target / (t_actual + t_target), 1.0)


# ==========================================
# 4. النموذج المختبَر
# ==========================================
def call_tested_model(prompt: str) -> Tuple[Optional[str], float]:
    """يستدعي النموذج المختبَر ويعيد (الرد, الزمن الفعلي بالثواني)."""
    start = time.time()
    response = _groq_call(
        messages=[{"role": "user", "content": prompt}],
        model=TESTED_MODEL,
        temperature=0.7,
        max_tokens=600
    )
    return response, time.time() - start


# ==========================================
# 5. المحلف — يقيّم رد النموذج لا سؤال المستخدم
# ==========================================
def _extract_rubric_scores(raw_text: str) -> Dict[str, float]:
    """يستخرج أول JSON صالح يحوي المفاتيح الأربعة بقيم 0-10."""
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
    """المحلف يقيّم جودة رد النموذج المختبَر بأربعة معايير."""
    jury_prompt = (
        "You are an AI Jury evaluating an AI-generated response.\n"
        "DO NOT answer the original question. Evaluate ONLY the response quality.\n"
        "Output ONLY a raw JSON object:\n"
        '{"accuracy": X, "clarity": X, "creativity": X, "conciseness": X}\n'
        "Values: floats from 0.0 to 10.0.\n\n"
        f'Response to evaluate:\n"""\n{model_response}\n"""'
    )

    raw = _groq_call(
        messages=[{"role": "user", "content": jury_prompt}],
        model=JURY_MODEL,
        temperature=0.1,
        max_tokens=100,
        json_mode=True,
        timeout=15
    )

    if raw is None:
        return {"scores": {k: 5.0 for k in RUBRIC_KEYS},
                "is_real": False,
                "error": "فشل استدعاء المحلف"}

    try:
        scores = _extract_rubric_scores(raw)
        return {"scores": scores, "is_real": True, "error": None}
    except ValueError as e:
        return {"scores": {k: 5.0 for k in RUBRIC_KEYS},
                "is_real": False,
                "error": str(e)}


# ==========================================
# 6. خط الإنتاج الكامل
# ==========================================
def run_ai_unit(prompt: str) -> Dict:
    """
    التسلسل الكامل:
      أ. الذكاء الاصطناعي يقدّر k والسبب
      ب. النموذج المختبَر يجيب (+ قياس الزمن)
      ج. المحلف يقيّم الرد
      د. W_k × A_k × S_k
    """
    # أ. تقدير الصعوبة بالذكاء الاصطناعي
    k, k_reason, k_is_real = assess_difficulty(prompt)
    w_k = calculate_w_k(k)

    # ب. استدعاء النموذج المختبَر
    model_response, t_actual = call_tested_model(prompt)
    if model_response is None:
        return {"success": False,
                "error": "فشل استدعاء النموذج المختبَر — تحقق من GROQ_API_KEY"}

    # ج. المحلف يقيّم الرد
    jury = call_jury(model_response)
    scores    = jury["scores"]
    avg_score = sum(scores.values()) / len(scores)

    # د. الحسابات النهائية
    s_k           = calculate_s_k(k, t_actual)
    a_k           = avg_score / 10.0
    ai_unit_score = w_k * a_k * s_k

    return {
        "success":        True,
        "model_tested":   TESTED_MODEL,
        "jury_model":     JURY_MODEL,
        # تقدير الصعوبة
        "k":              k,
        "k_reason":       k_reason,
        "k_assessed_by_ai": k_is_real,
        # المعادلة
        "w_k":            round(w_k, 4),
        "s_k":            round(s_k, 4),
        "a_k":            round(a_k, 4),
        "ai_unit_score":  round(ai_unit_score, 4),
        # تفاصيل التقييم
        "avg_score":      round(avg_score, 2),
        "scores":         scores,
        "t_actual":       round(t_actual, 3),
        "model_response": model_response,
        "jury_is_real":   jury["is_real"],
        "jury_error":     jury["error"],
    }


# ==========================================
# 7. Telegram Webhook
# ==========================================
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

    _send_tg(token, chat_id, "⏳ جارٍ التقييم...")

    result = run_ai_unit(user_text)

    if not result["success"]:
        _send_tg(token, chat_id, f"❌ {result['error']}")
        return {"status": "ok"}

    s         = result["scores"]
    jury_real = "✅" if result["jury_is_real"] else "⚠️ احتياطي"
    k_real    = "🧠 ذكاء اصطناعي" if result["k_assessed_by_ai"] else "⚠️ احتياطي"

    reply = (
        f"🏆 *تقرير AI-Unit*\n\n"
        f"🤖 النموذج: `{result['model_tested']}`\n\n"
        f"🎯 *مستوى الصعوبة:* k={result['k']} \\({k_real}\\)\n"
        f"💬 السبب: _{result['k_reason']}_\n\n"
        f"📊 *النتيجة النهائية:* `{result['ai_unit_score']} AIU`\n\n"
        f"📐 *المعادلة:*\n"
        f"  e^k × Ak × Sk\n"
        f"  {result['w_k']} × {result['a_k']} × {result['s_k']}\n\n"
        f"📋 *تقييم المحلف* {jury_real}:\n"
        f"  • الدقة: {s.get('accuracy', '?')}/10\n"
        f"  • الوضوح: {s.get('clarity', '?')}/10\n"
        f"  • الإبداع: {s.get('creativity', '?')}/10\n"
        f"  • الإيجاز: {s.get('conciseness', '?')}/10\n\n"
        f"⏱️ زمن الاستجابة: {result['t_actual']} ث"
    )
    _send_tg(token, chat_id, reply)
    return {"status": "ok"}


# ==========================================
# 8. API Endpoints
# ==========================================
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
        "status":       "operational",
        "version":      "6.0",
        "tested_model": TESTED_MODEL,
        "jury_model":   JURY_MODEL,
        "groq_key":     "✅" if os.environ.get("GROQ_API_KEY") else "❌ مفقود",
        "tg_token":     "✅" if os.environ.get("TELEGRAM_BOT_TOKEN") else "❌ مفقود",
        "difficulty_assessment": "AI-based (llama-3.1-8b)",
        "weight_formula": "W_k = e^k (exponential)",
    }
    
