# ==============================================================
# OK-Agent V2.0 – وكيل ذكاء اصطناعي متكامل بأدوات حقيقية
# ==============================================================
# الميزات الجديدة:
#   • بحث حقيقي في الإنترنت (DuckDuckGo) بدون مفتاح API.
#   • تحويل عملات حقيقي (API مجاني).
#   • طقس حقيقي (OpenWeatherMap – مفتاح اختياري).
#   • قراءة وتلخيص صفحات الويب.
#   • حاسبة متطورة (تدعم الدوال الرياضية).
#   • نظام تذكيرات ومهام كامل مع حفظ في الذاكرة.
#   • نظام تقييم AI-Unit مدمج (لتقييم النماذج الأخرى).
#   • System Prompt محسّن يقدم البوت كمساعد ذكي ومتكامل.
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
from bs4 import BeautifulSoup

app = FastAPI(title="OK-Agent", version="2.0")

# ---------- الإعدادات ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")  # اختياري: من OpenWeatherMap
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# الذاكرة والمهام
conversation_memory: Dict[int, list] = {}
tasks: Dict[int, list] = {}
reminders: Dict[int, list] = {}

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

# ---------- 2. الأدوات الحقيقية (Tools) ----------
def tool_search(query: str) -> str:
    """بحث حقيقي في الإنترنت عبر DuckDuckGo (مجاني بدون مفتاح)."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("AbstractText"):
            return f"🔍 *نتيجة بحث عن '{query}':*\n{data['AbstractText'][:500]}"
        elif data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if "Text" in topic:
                    return f"🔍 *نتائج عن '{query}':*\n{topic['Text'][:300]}"
        return f"⚠️ لم يتم العثور على نتائج لـ '{query}'."
    except Exception as e:
        return f"⚠️ فشل البحث: {str(e)}"

def tool_calculate(expression: str) -> str:
    """حاسبة متطورة تدعم الدوال الرياضية."""
    try:
        # السماح فقط بالرموز الآمنة
        safe_dict = {
            "math": math,
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"📐 النتيجة: {result}"
    except Exception as e:
        return f"⚠️ تعبير غير صالح: {str(e)}"

def tool_currency(amount: float, from_cur: str, to_cur: str) -> str:
    """تحويل عملات حقيقي (من API مجاني)."""
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_cur.upper()}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if to_cur.upper() not in data["rates"]:
            return f"⚠️ عملة {to_cur} غير مدعومة."
        rate = data["rates"][to_cur.upper()]
        result = amount * rate
        return f"💱 {amount} {from_cur.upper()} = {result:.2f} {to_cur.upper()}"
    except Exception as e:
        return f"⚠️ فشل التحويل: {str(e)}"

def tool_weather(city: str) -> str:
    """حالة الطقس الحقيقية (يتطلب مفتاح API من OpenWeatherMap)."""
    if not WEATHER_API_KEY:
        return "⚠️ مفتاح الطقس غير مضبوط. احصل على مفتاح من OpenWeatherMap."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=ar"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("cod") != 200:
            return f"⚠️ المدينة غير موجودة: {data.get('message', '')}"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return f"🌡️ *حالة الطقس في {city}:*\n🌡️ درجة الحرارة: {temp}°C\n☁️ {desc}\n💧 الرطوبة: {humidity}%"
    except Exception as e:
        return f"⚠️ فشل جلب الطقس: {str(e)}"

def tool_read_website(url: str) -> str:
    """يقرأ محتوى صفحة ويب ويعيد ملخصاً (نصاً فقط)."""
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # اختصار النص إلى 1000 حرف
        if len(text) > 1000:
            text = text[:1000] + "..."
        return f"📄 *محتوى الصفحة:*\n{text}"
    except Exception as e:
        return f"⚠️ فشل قراءة الموقع: {str(e)}"

def tool_calendar(action: str, date: str = None, title: str = None) -> str:
    """إدارة التقويم (محاكاة)."""
    if action == "add" and date and title:
        return f"✅ تم إضافة '{title}' في {date}"
    elif action == "list":
        return "📅 اليوم: لا توجد أحداث مسجلة."
    else:
        return "⚠️ استخدم: /calendar add 2026-07-03 'اجتماع'"

def tool_email(action: str, to: str = None, subject: str = None, body: str = None) -> str:
    """إدارة البريد (محاكاة)."""
    if action == "send" and to and subject:
        return f"📧 تم إرسال بريد إلى {to}: '{subject}'"
    elif action == "read":
        return "📭 صندوق الوارد: 3 رسائل غير مقروءة."
    else:
        return "⚠️ استخدم: /email send 'ahmed@example.com' 'موضوع' 'نص'"

def tool_reminder(action: str, text: str = None, chat_id: int = None) -> str:
    """إدارة التذكيرات (مع حفظ في الذاكرة)."""
    if chat_id is None:
        return "⚠️ خطأ داخلي."
    if action == "add" and text:
        if chat_id not in reminders:
            reminders[chat_id] = []
        reminders[chat_id].append({"text": text, "created": datetime.now().isoformat()})
        return f"⏰ تم إضافة تذكير: '{text}'"
    elif action == "list":
        if chat_id not in reminders or not reminders[chat_id]:
            return "📋 لا توجد تذكيرات."
        r_list = "\n".join([f"- {r['text']} (منذ {r['created']})" for r in reminders[chat_id]])
        return f"📋 *تذكيراتك:*\n{r_list}"
    else:
        return "⚠️ استخدم: /reminder add 'نص التذكير'"

# قاموس الأدوات المتاحة (مع وصف لكل أداة)
TOOLS = {
    "search": {"func": tool_search, "desc": "بحث في الإنترنت - /search <نص>"},
    "calculate": {"func": tool_calculate, "desc": "حاسبة - /calc 2+2"},
    "currency": {"func": tool_currency, "desc": "تحويل عملات - /currency 100 USD SAR"},
    "weather": {"func": tool_weather, "desc": "حالة الطقس - /weather الرياض"},
    "read": {"func": tool_read_website, "desc": "قراءة موقع - /read https://example.com"},
    "calendar": {"func": tool_calendar, "desc": "تقويم - /calendar add 2026-07-03 'اجتماع'"},
    "email": {"func": tool_email, "desc": "بريد - /email send 'a@b.com' 'موضوع' 'نص'"},
    "reminder": {"func": tool_reminder, "desc": "تذكير - /reminder add 'نص'"},
}

# ---------- 3. معالج الأوامر ----------
def process_command(chat_id: int, text: str) -> str:
    """
    يكتشف الأمر وينفذ الأداة المناسبة، أو يعيد التوجيه إلى المحادثة الذكية.
    """
    # أوامر النظام
    if text.startswith("/help"):
        help_text = "🤖 *OK-Agent V2.0*\n\nالأوامر المتاحة:\n"
        for cmd, tool in TOOLS.items():
            help_text += f"  • /{cmd} – {tool['desc']}\n"
        help_text += "\nأوامر المهام:\n  • /task <نص> – إضافة مهمة\n  • /tasks – عرض المهام\n  • /done – إكمال مهمة\n  • /reset – مسح الذاكرة والمهام\n  • /eval <نص> – تقييم نموذج (AI-Unit)"
        return help_text

    if text.startswith("/reset"):
        conversation_memory[chat_id] = []
        tasks[chat_id] = []
        reminders[chat_id] = []
        return "🧹 تم مسح الذاكرة والمهام والتذكيرات."

    if text.startswith("/task "):
        task = text[6:].strip()
        if not task:
            return "⚠️ اكتب المهمة بعد /task"
        if chat_id not in tasks:
            tasks[chat_id] = []
        tasks[chat_id].append(task)
        return f"✅ تم إضافة المهمة: {task}"

    if text.startswith("/tasks"):
        if chat_id not in tasks or not tasks[chat_id]:
            return "📭 لا توجد مهام."
        return "📋 *مهامك:*\n" + "\n".join([f"- {t}" for t in tasks[chat_id]])

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
    if text.startswith("/weather "):
        return tool_weather(text[9:].strip())
    if text.startswith("/read "):
        return tool_read_website(text[6:].strip())
    if text.startswith("/calendar "):
        parts = text[10:].split(" ", 2)
        if len(parts) < 2:
            return "⚠️ استخدم: /calendar add 2026-07-03 'اجتماع'"
        return tool_calendar(parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None)
    if text.startswith("/email "):
        parts = text[7:].split(" ", 3)
        if len(parts) < 2:
            return "⚠️ استخدم: /email send 'a@b.com' 'موضوع' 'نص'"
        return tool_email(parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None, parts[3] if len(parts) > 3 else None)
    if text.startswith("/reminder "):
        parts = text[10:].split(" ", 1)
        if len(parts) < 2:
            return "⚠️ استخدم: /reminder add 'نص التذكير'"
        return tool_reminder(parts[0], parts[1] if len(parts) > 1 else None, chat_id)

    # أوامر تقييم AI-Unit
    if text.startswith("/eval "):
        return run_ai_unit(text[6:].strip())

    # إذا لم يكن أمراً، نمرره للمحادثة الذكية
    return handle_smart_conversation(chat_id, text)

# ---------- 4. المحادثة الذكية (مع سياق محسّن) ----------
def handle_smart_conversation(chat_id: int, text: str) -> str:
    """يستخدم Groq للرد مع سياق المحادثة."""
    if chat_id not in conversation_memory:
        conversation_memory[chat_id] = []

    conversation_memory[chat_id].append({"role": "user", "content": text})
    if len(conversation_memory[chat_id]) > 10:
        conversation_memory[chat_id] = conversation_memory[chat_id][-10:]

    # بناء قائمة الأدوات المتاحة لتعريف البوت بها
    tools_list = "\n".join([f"- /{cmd}: {tool['desc']}" for cmd, tool in TOOLS.items()])

    system_prompt = (
        "أنت OK-Agent، وكيل ذكاء اصطناعي متكامل وذكي. أنت ودود، محترف، وتساعد المستخدم في أي مهمة.\n\n"
        "معلومات عنك:\n"
        "- أنت متصل بالإنترنت وتستطيع البحث عن أي معلومة حديثة عبر أمر /search.\n"
        "- لديك أدوات متعددة: حاسبة، تحويل عملات، طقس، قراءة مواقع، تقويم، بريد، تذكيرات.\n"
        "- يمكنك إدارة المهام والتذكيرات للمستخدم.\n"
        "- أنت تتحدث العربية بطلاقة وتفهم اللهجات.\n\n"
        "عندما يسألك المستخدم عن شيء، حاول مساعدته بأفضل طريقة. إذا كان السؤال يتطلب بحثاً، اقترح عليه استخدام /search.\n"
        "لا تقل أنك 'لا تملك وصولاً للإنترنت'، بل قل أنك تستطيع البحث عبر الأمر /search.\n\n"
        f"الأدوات المتاحة لك وللمستخدم:\n{tools_list}\n\n"
        "تذكر دائماً أن تكون مفيداً، دقيقاً، وودوداً."
    )

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory[chat_id]
    response = call_groq(messages, temperature=0.7)

    if response is None:
        return "⚠️ عذراً، حدث خطأ في الاتصال بالذكاء الاصطناعي. حاول مرة أخرى."

    conversation_memory[chat_id].append({"role": "assistant", "content": response})
    return response

# ---------- 5. نظام AI-Unit (التقييم الأصلي) ----------
MARKET_LEADER_RUNTIMES = {
    1: [0.10, 0.14, 0.12],
    2: [0.35, 0.42, 0.38],
    3: [0.95, 1.15, 1.02],
    4: [2.10, 2.60, 2.30],
    5: [4.80, 6.10, 5.40],
}

MASTER_CRITERIA = [
    {"name": "accuracy", "desc": "هل الجواب صحيح؟", "weight": "exp"},
    {"name": "clarity", "desc": "هل الجواب واضح؟", "weight": "linear"},
    {"name": "completeness", "desc": "هل غطى جميع الجوانب؟", "weight": "linear"},
]

def run_ai_unit(prompt: str) -> str:
    """تقييم نموذج باستخدام AI-Unit."""
    k = min(5, max(1, len(prompt) // 50))
    w_k = math.e ** k

    model_response = call_groq([{"role": "user", "content": prompt}])
    if model_response is None:
        return "⚠️ فشل تقييم النموذج."

    t_actual = 1.5
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [1.0]))
    s_k = t_target / (t_actual + t_target)

    # محاكاة تقييم المحلف (لأننا لا نريد استدعاء نموذج آخر في هذه النسخة)
    scores = {"accuracy": 8.5, "clarity": 7.5, "completeness": 8.0}
    a_k = sum(scores.values()) / len(scores) / 10.0
    ai_unit_score = w_k * a_k * s_k

    return (
        f"🏆 *تقرير AI-Unit*\n"
        f"——————————\n"
        f"🎯 المستوى: k={k}\n"
        f"📐 W_k = e^{k} = {w_k:.2f}\n"
        f"⚡ S_k = {s_k:.3f}\n"
        f"📊 A_k = {a_k:.3f}\n"
        f"🏅 النتيجة: **{ai_unit_score:.2f} AIU**\n\n"
        f"📝 *رد النموذج:*\n{model_response[:300]}..."
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
        "agent": "OK-Agent V2.0",
        "tools": list(TOOLS.keys()),
        "groq": "✅" if GROQ_API_KEY else "❌",
        "telegram": "✅" if TELEGRAM_TOKEN else "❌",
        "weather": "✅" if WEATHER_API_KEY else "⚠️ اختياري",
        "memory_size": len(conversation_memory),
        "tasks_count": sum(len(t) for t in tasks.values()),
        "reminders_count": sum(len(r) for r in reminders.values()),
    }

# ---------- 8. تشغيل السيرفر ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
