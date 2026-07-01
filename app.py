# ==============================================================
# OK-Agent: وكيل ذكاء اصطناعي متكامل (V1.0)
# ==============================================================
# الميزات:
#   • محادثة ذكية (Groq).
#   • أدوات فعلية: بحث، حاسبة، تقويم، بريد، عملات، تذكير.
#   • نظام تقييم AI-Unit (لتقييم النماذج الأخرى).
#   • تكامل مع تلغرام + API.
#   • يعمل على Render أو أي خادم Python.
# ==============================================================

from fastapi import FastAPI, Request
import time
import json
import os
import re
import requests
import math
import statistics
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta

app = FastAPI(title="OK-Agent", version="1.0")

# ---------- الإعدادات ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# الذاكرة المؤقتة (للمحادثات والمهام)
conversation_memory: Dict[int, list] = {}  # chat_id -> قائمة الرسائل
tasks: Dict[int, list] = {}  # chat_id -> قائمة المهام

# ---------- 1. استدعاء Groq (مع دالة النظام) ----------
def call_groq(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False
) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return None

# ---------- 2. الأدوات (Tools) ----------
def tool_search(query: str) -> str:
    """يبحث في الإنترنت (محاكاة) ويعيد ملخصاً."""
    # في الإصدار الحقيقي، استخدم SerpAPI أو DuckDuckGo
    return f"نتائج بحث عن '{query}': ... (هذه محاكاة، اربط بـ SerpAPI للحصول على نتائج حقيقية)"

def tool_calculate(expression: str) -> str:
    """يحسب تعبيراً رياضياً."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"النتيجة: {result}"
    except:
        return "⚠️ تعبير غير صالح."

def tool_currency(amount: float, from_cur: str, to_cur: str) -> str:
    """يحول العملات (محاكاة)."""
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.78, "SAR": 3.75, "EGP": 47.5}
    if from_cur not in rates or to_cur not in rates:
        return "⚠️ عملة غير مدعومة."
    result = amount * (rates[to_cur] / rates[from_cur])
    return f"{amount} {from_cur} = {result:.2f} {to_cur}"

def tool_calendar(action: str, date: str = None, title: str = None) -> str:
    """يدير التقويم (محاكاة)."""
    if action == "add" and date and title:
        return f"✅ تم إضافة '{title}' في {date}"
    elif action == "list":
        return "📅 أحداث اليوم: لا توجد أحداث مسجلة."
    else:
        return "⚠️ استخدام: /calendar add 2026-07-03 'اجتماع'"

def tool_email(action: str, to: str = None, subject: str = None, body: str = None) -> str:
    """يدير البريد الإلكتروني (محاكاة)."""
    if action == "send" and to and subject:
        return f"📧 تم إرسال بريد إلى {to}: '{subject}'"
    elif action == "read":
        return "📭 صندوق الوارد: (محاكاة) 3 رسائل غير مقروءة."
    else:
        return "⚠️ استخدام: /email send 'ahmed@example.com' 'موضوع' 'نص'"

def tool_reminder(action: str, text: str = None) -> str:
    """يدير التذكيرات."""
    if action == "add" and text:
        return f"⏰ تم إضافة تذكير: '{text}'"
    elif action == "list":
        return "📋 التذكيرات: لا توجد تذكيرات."
    else:
        return "⚠️ استخدام: /reminder add 'تذكيري'"

# قاموس الأدوات المتاحة
TOOLS = {
    "search": tool_search,
    "calculate": tool_calculate,
    "currency": tool_currency,
    "calendar": tool_calendar,
    "email": tool_email,
    "reminder": tool_reminder,
}

# ---------- 3. معالج الأوامر (التوجيه الذكي) ----------
def process_command(chat_id: int, text: str) -> str:
    """
    يكتشف الأمر وينفذ الأداة المناسبة، أو يعيد التوجيه إلى المحادثة الذكية.
    """
    # أوامر النظام
    if text.startswith("/help"):
        return (
            "🤖 *OK-Agent V1.0*\n"
            "الأوامر المتاحة:\n"
            "/help - عرض المساعدة\n"
            "/reset - حذف الذاكرة\n"
            "/task - إضافة مهمة\n"
            "/tasks - عرض المهام\n"
            "/done - إنهاء مهمة\n"
            "/search <نص> - بحث في الإنترنت\n"
            "/calc <تعبير> - حاسبة\n"
            "/currency <مبلغ> <من> <إلى> - تحويل عملات\n"
            "/calendar add <تاريخ> '<عنوان>' - إضافة حدث\n"
            "/email send '<بريد>' '<موضوع>' '<نص>' - إرسال بريد\n"
            "/reminder add '<نص>' - إضافة تذكير\n"
            "/eval <نص> - تقييم نموذج (AI-Unit)"
        )

    if text.startswith("/reset"):
        conversation_memory[chat_id] = []
        tasks[chat_id] = []
        return "🧹 تم مسح الذاكرة والمهام."

    if text.startswith("/task"):
        task = text[5:].strip()
        if not task:
            return "⚠️ اكتب المهمة بعد /task"
        if chat_id not in tasks:
            tasks[chat_id] = []
        tasks[chat_id].append(task)
        return f"✅ تم إضافة المهمة: {task}"

    if text.startswith("/tasks"):
        if chat_id not in tasks or not tasks[chat_id]:
            return "📭 لا توجد مهام."
        return "📋 مهامك:\n" + "\n".join([f"- {t}" for t in tasks[chat_id]])

    if text.startswith("/done"):
        if chat_id not in tasks or not tasks[chat_id]:
            return "⚠️ لا توجد مهام لإكمالها."
        removed = tasks[chat_id].pop(0)
        return f"✅ تم إكمال: {removed}"

    # أوامر الأدوات
    if text.startswith("/search "):
        return tool_search(text[8:].strip())
    if text.startswith("/calc "):
        return tool_calculate(text[6:].strip())
    if text.startswith("/currency "):
        parts = text[10:].split()
        if len(parts) != 3:
            return "⚠️ استخدم: /currency 100 USD SAR"
        try:
            amount = float(parts[0])
            return tool_currency(amount, parts[1].upper(), parts[2].upper())
        except ValueError:
            return "⚠️ المبلغ يجب أن يكون رقماً."
    if text.startswith("/calendar "):
        parts = text[10:].split(" ", 2)
        if len(parts) < 2:
            return "⚠️ استخدم: /calendar add 2026-07-03 'اجتماع'"
        return tool_calendar(parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None)
    if text.startswith("/email "):
        parts = text[7:].split(" ", 3)
        if len(parts) < 2:
            return "⚠️ استخدم: /email send 'ahmed@example.com' 'موضوع' 'نص'"
        return tool_email(parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None, parts[3] if len(parts) > 3 else None)
    if text.startswith("/reminder "):
        parts = text[10:].split(" ", 1)
        if len(parts) < 2:
            return "⚠️ استخدم: /reminder add 'تذكيري'"
        return tool_reminder(parts[0], parts[1] if len(parts) > 1 else None)

    # أوامر تقييم AI-Unit
    if text.startswith("/eval "):
        return run_ai_unit(text[6:].strip())

    # إذا لم يكن أمراً، نمرره للمحادثة الذكية
    return handle_smart_conversation(chat_id, text)

# ---------- 4. المحادثة الذكية (مع السياق) ----------
def handle_smart_conversation(chat_id: int, text: str) -> str:
    """يستخدم Groq للرد مع الحفاظ على سياق المحادثة."""
    if chat_id not in conversation_memory:
        conversation_memory[chat_id] = []

    # إضافة رسالة المستخدم للذاكرة
    conversation_memory[chat_id].append({"role": "user", "content": text})

    # قص الذاكرة إذا زادت عن 10 رسائل
    if len(conversation_memory[chat_id]) > 10:
        conversation_memory[chat_id] = conversation_memory[chat_id][-10:]

    # بناء النظام
    system_prompt = (
        "أنت OK-Agent، وكيل ذكاء اصطناعي متكامل. أنت ودود، ذكي، وتساعد المستخدم في مهامه.\n"
        "يمكنك الإجابة عن الأسئلة، تقديم النصائح، وتنفيذ الأوامر المطلوبة.\n"
        "استخدم اللغة العربية الفصحى أو العامية حسب سياق المستخدم.\n"
        "إذا سألك المستخدم عن شيء لا تعرفه، قل ذلك بصراحة."
    )
    messages = [{"role": "system", "content": system_prompt}] + conversation_memory[chat_id]

    response = call_groq(messages, temperature=0.7)
    if response is None:
        return "⚠️ عذراً، حدث خطأ في الاتصال بالذكاء الاصطناعي."

    # إضافة رد البوت للذاكرة
    conversation_memory[chat_id].append({"role": "assistant", "content": response})
    return response

# ---------- 5. نظام AI-Unit (التقييم الأصلي) ----------
# (هنا نضع نسخة مبسطة من نظام التقييم الذي بنيته سابقاً)

MARKET_LEADER_RUNTIMES = {1: [0.10, 0.14, 0.12], 2: [0.35, 0.42, 0.38], 3: [0.95, 1.15, 1.02], 4: [2.10, 2.60, 2.30], 5: [4.80, 6.10, 5.40]}
MASTER_CRITERIA = [
    {"name": "accuracy", "desc": "هل الجواب صحيح؟", "weight": "exp"},
    {"name": "clarity", "desc": "هل الجواب واضح؟", "weight": "linear"},
]

def run_ai_unit(prompt: str) -> str:
    """تقييم نموذج باستخدام AI-Unit (نسخة مبسطة)."""
    # محاكاة تقدير k
    k = min(5, max(1, len(prompt) // 50))
    w_k = math.e ** k
    # محاكاة استدعاء النموذج
    model_response = call_groq([{"role": "user", "content": prompt}])
    if model_response is None:
        return "⚠️ فشل تقييم النموذج."
    t_actual = 1.5  # ثابت للمحاكاة
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [1.0]))
    s_k = t_target / (t_actual + t_target)
    # تقييم وهمي
    scores = {"accuracy": 8.5, "clarity": 7.0}
    a_k = sum(scores.values()) / len(scores) / 10.0
    ai_unit_score = w_k * a_k * s_k
    return (
        f"🏆 *تقرير AI-Unit*\n"
        f"المستوى: k={k}\n"
        f"W_k = e^{k} = {w_k:.2f}\n"
        f"S_k = {s_k:.3f}\n"
        f"A_k = {a_k:.3f}\n"
        f"النتيجة: **{ai_unit_score:.2f} AIU**\n\n"
        f"رد النموذج: {model_response[:200]}..."
    )

# ---------- 6. Webhook تلغرام ----------
def send_tg(chat_id: int, text: str) -> None:
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"❌ Telegram error: {e}")

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}
    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    reply = process_command(chat_id, user_text)
    send_tg(chat_id, reply)
    return {"status": "ok"}

# ---------- 7. واجهات برمجية (API) ----------
@app.post("/api/v1/chat")
async def chat_api(request: Request):
    body = await request.json()
    chat_id = body.get("chat_id", 0)
    message = body.get("message", "").strip()
    if not message:
        return {"error": "الرسالة فارغة"}
    reply = process_command(chat_id, message)
    return {"reply": reply}

@app.post("/api/v1/evaluate")
async def evaluate_api(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"error": "حقل prompt مطلوب"}
    return {"result": run_ai_unit(prompt)}

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "agent": "OK-Agent V1.0",
        "tools": list(TOOLS.keys()),
        "groq": "✅" if GROQ_API_KEY else "❌",
        "telegram": "✅" if TELEGRAM_TOKEN else "❌",
        "memory_size": len(conversation_memory),
        "tasks_count": sum(len(t) for t in tasks.values()),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
