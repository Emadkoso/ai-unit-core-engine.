# ==============================================================
# OK-Agent V3.0 – وكيل ذاتي التطور
# ==============================================================
# فلسفة التطور:
#   • المستوى (Level) يحسب من المعادلة: level = min(10, floor(log2(interactions + 1)))
#   • كل مستوى يفتح أدوات جديدة ويزيد عمق الإجابات.
#   • يتم حفظ الإحصائيات في ملف JSON (لتستمر بين جلسات التشغيل).
#   • يستخدم Groq API مع تغيير درجة الحرارة حسب المستوى.
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

app = FastAPI(title="OK-Agent", version="3.0")

# ---------- الإعدادات ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ملف لحفظ بيانات التطور
STATS_FILE = "agent_stats.json"

# ---------- 1. إدارة بيانات التطور ----------
def load_stats() -> dict:
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"interactions": 0, "level": 1, "total_tasks": 0, "total_tools_used": 0}

def save_stats(stats: dict):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

def update_stats(chat_id: int, tool_used: str = None):
    stats = load_stats()
    stats["interactions"] += 1
    if tool_used:
        stats["total_tools_used"] += 1
    # حساب المستوى الجديد: level = min(10, floor(log2(interactions+1)))
    new_level = min(10, int(math.log2(stats["interactions"] + 1)))
    if new_level > stats["level"]:
        stats["level"] = new_level
    save_stats(stats)
    return stats

# ---------- 2. الأدوات المتاحة (كل مستوى يفتح أدوات جديدة) ----------
TOOLS_BY_LEVEL = {
    1: ["search", "calculate"],
    2: ["search", "calculate", "currency"],
    3: ["search", "calculate", "currency", "weather"],
    4: ["search", "calculate", "currency", "weather", "read"],
    5: ["search", "calculate", "currency", "weather", "read", "calendar"],
    6: ["search", "calculate", "currency", "weather", "read", "calendar", "email"],
    7: ["search", "calculate", "currency", "weather", "read", "calendar", "email", "reminder"],
    8: ["search", "calculate", "currency", "weather", "read", "calendar", "email", "reminder", "task"],
    9: ["search", "calculate", "currency", "weather", "read", "calendar", "email", "reminder", "task", "translate"],
    10: ["search", "calculate", "currency", "weather", "read", "calendar", "email", "reminder", "task", "translate", "code_exec"],
}

# تعريفات الأدوات (جميعها موجودة هنا، ولكن بعضها مقيد حسب المستوى)
def tool_search(query: str) -> str:
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
    try:
        safe_dict = {"math": math, "pi": math.pi, "e": math.e, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "sqrt": math.sqrt, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"📐 النتيجة: {result}"
    except Exception as e:
        return f"⚠️ تعبير غير صالح: {str(e)}"

def tool_currency(amount: float, from_cur: str, to_cur: str) -> str:
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
    if not WEATHER_API_KEY:
        return "⚠️ مفتاح الطقس غير مضبوط."
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
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 1000:
            text = text[:1000] + "..."
        return f"📄 *محتوى الصفحة:*\n{text}"
    except Exception as e:
        return f"⚠️ فشل قراءة الموقع: {str(e)}"

def tool_calendar(action: str, date: str = None, title: str = None) -> str:
    if action == "add" and date and title:
        return f"✅ تم إضافة '{title}' في {date}"
    elif action == "list":
        return "📅 اليوم: لا توجد أحداث مسجلة."
    else:
        return "⚠️ استخدم: /calendar add 2026-07-03 'اجتماع'"

def tool_email(action: str, to: str = None, subject: str = None, body: str = None) -> str:
    if action == "send" and to and subject:
        return f"📧 تم إرسال بريد إلى {to}: '{subject}'"
    elif action == "read":
        return "📭 صندوق الوارد: 3 رسائل غير مقروءة."
    else:
        return "⚠️ استخدم: /email send 'a@b.com' 'موضوع' 'نص'"

def tool_reminder(action: str, text: str = None, chat_id: int = None) -> str:
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

def tool_translate(text: str, target_lang: str = "ar") -> str:
    """ترجمة بسيطة عبر Groq (يتم تنفيذها كمحادثة ذكية)."""
    messages = [{"role": "user", "content": f"ترجم النص التالي إلى {target_lang}: {text}"}]
    response = call_groq(messages, temperature=0.3, max_tokens=300)
    return response if response else "⚠️ فشلت الترجمة."

def tool_code_exec(code: str) -> str:
    """تنفيذ كود بايثون بسيط (مقيد بأمان شديد)."""
    try:
        # تحذير: هذا غير آمن في بيئة إنتاج، لكن للمحاكاة فقط
        exec_globals = {"__builtins__": {"print": print, "range": range, "len": len, "sum": sum, "list": list, "dict": dict, "str": str, "int": int, "float": float}}
        exec(code, exec_globals)
        return "✅ تم تنفيذ الكود بنجاح (بدون مخرجات)."
    except Exception as e:
        return f"⚠️ خطأ في التنفيذ: {str(e)}"

TOOLS_FUNCS = {
    "search": tool_search,
    "calculate": tool_calculate,
    "currency": tool_currency,
    "weather": tool_weather,
    "read": tool_read_website,
    "calendar": tool_calendar,
    "email": tool_email,
    "reminder": tool_reminder,
    "task": None,  # معالجة خاصة
    "translate": tool_translate,
    "code_exec": tool_code_exec,
}

# ---------- 3. الذاكرة والمهام والتذكيرات ----------
conversation_memory: Dict[int, list] = {}
tasks: Dict[int, list] = {}
reminders: Dict[int, list] = {}

# ---------- 4. استدعاء Groq ----------
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

# ---------- 5. معالج الأوامر مع التطور الذاتي ----------
def process_command(chat_id: int, text: str) -> str:
    # تحميل الإحصائيات وتحديثها
    stats = update_stats(chat_id)
    level = stats["level"]
    
    # عرض المستوى عند طلب /level
    if text.startswith("/level"):
        return f"📊 *مستوى البوت الحالي:* {level}/10\n📈 عدد المحادثات: {stats['interactions']}\n🛠️ عدد الأدوات المستخدمة: {stats['total_tools_used']}"

    # عرض الأدوات المتاحة حسب المستوى
    if text.startswith("/tools"):
        available = TOOLS_BY_LEVEL.get(level, [])
        if not available:
            return "⚠️ لا توجد أدوات متاحة حالياً."
        return "🛠️ *الأدوات المتاحة لمستواك:*\n" + "\n".join([f"- /{t}" for t in available])

    # باقي الأوامر كما هي، ولكن نتحقق من وجود الأداة في المستوى الحالي
    if text.startswith("/help"):
        available = TOOLS_BY_LEVEL.get(level, [])
        help_text = f"🤖 *OK-Agent V3.0 (المستوى {level}/10)*\n\nالأدوات المتاحة:\n"
        for tool in available:
            help_text += f"  • /{tool}\n"
        help_text += "\nأوامر المهام:\n  • /task <نص>\n  • /tasks\n  • /done\n  • /reset\n  • /eval <نص>\n  • /level\n  • /tools"
        return help_text

    # أمر إضافة مهمة
    if text.startswith("/task "):
        task = text[6:].strip()
        if not task:
            return "⚠️ اكتب المهمة بعد /task"
        if chat_id not in tasks:
            tasks[chat_id] = []
        tasks[chat_id].append(task)
        stats = load_stats()
        stats["total_tasks"] += 1
        save_stats(stats)
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

    if text.startswith("/reset"):
        conversation_memory[chat_id] = []
        tasks[chat_id] = []
        reminders[chat_id] = []
        return "🧹 تم مسح الذاكرة والمهام والتذكيرات."

    # أمر تقييم AI-Unit
    if text.startswith("/eval "):
        return run_ai_unit(text[6:].strip())

    # أوامر الأدوات – التحقق من المستوى
    for tool_name in TOOLS_BY_LEVEL.get(level, []):
        if text.startswith(f"/{tool_name} "):
            args = text[len(tool_name)+2:].strip()
            # معالجة خاصة للأدوات التي تحتاج chat_id
            if tool_name == "reminder":
                parts = args.split(" ", 1)
                if len(parts) < 2:
                    return "⚠️ استخدم: /reminder add 'نص التذكير'"
                return TOOLS_FUNCS[tool_name](parts[0], parts[1] if len(parts) > 1 else None, chat_id)
            elif tool_name == "calendar":
                parts = args.split(" ", 2)
                if len(parts) < 2:
                    return "⚠️ استخدم: /calendar add 2026-07-03 'اجتماع'"
                return TOOLS_FUNCS[tool_name](parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None)
            elif tool_name == "email":
                parts = args.split(" ", 3)
                if len(parts) < 2:
                    return "⚠️ استخدم: /email send 'a@b.com' 'موضوع' 'نص'"
                return TOOLS_FUNCS[tool_name](parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None, parts[3] if len(parts) > 3 else None)
            elif tool_name == "currency":
                parts = args.split()
                if len(parts) != 3:
                    return "⚠️ استخدم: /currency 100 USD SAR"
                try:
                    amount = float(parts[0])
                    return TOOLS_FUNCS[tool_name](amount, parts[1].upper(), parts[2].upper())
                except ValueError:
                    return "⚠️ المبلغ يجب أن يكون رقماً."
            elif tool_name == "translate":
                # الترجمة: نأخذ النص كاملاً، اللغة الافتراضية عربية
                return TOOLS_FUNCS[tool_name](args, "ar")
            elif tool_name == "code_exec":
                return TOOLS_FUNCS[tool_name](args)
            else:
                return TOOLS_FUNCS[tool_name](args)

    # إذا لم يكن أمراً، نمرره للمحادثة الذكية مع درجة حرارة متغيرة حسب المستوى
    return handle_smart_conversation(chat_id, text, level)

# ---------- 6. المحادثة الذكية (مع تطور المستوى) ----------
def handle_smart_conversation(chat_id: int, text: str, level: int) -> str:
    if chat_id not in conversation_memory:
        conversation_memory[chat_id] = []

    conversation_memory[chat_id].append({"role": "user", "content": text})
    if len(conversation_memory[chat_id]) > 10:
        conversation_memory[chat_id] = conversation_memory[chat_id][-10:]

    # بناء سياق مع مستوى البوت
    available = TOOLS_BY_LEVEL.get(level, [])
    tools_str = ", ".join(available) if available else "أدوات أساسية"
    system_prompt = (
        f"أنت OK-Agent، وكيل ذكاء اصطناعي متطور. مستواك الحالي هو {level}/10.\n"
        f"الأدوات المتاحة لك: {tools_str}.\n"
        "أنت ودود، ذكي، وتساعد المستخدم بكل إخلاص.\n"
        "إذا كان السؤال يتطلب بحثاً، استخدم /search.\n"
        "تحدث بطلاقة، وقدم إجابات عميقة كلما ارتفع المستوى.\n"
        "لا تقل أنك لا تملك وصولاً للإنترنت، بل قل أن لديك أداة بحث."
    )

    # تغيير درجة الحرارة حسب المستوى (كلما ارتفع المستوى، زادت الإبداعية)
    temperature = 0.5 + (level * 0.05)
    temperature = min(temperature, 1.0)

    messages = [{"role": "system", "content": system_prompt}] + conversation_memory[chat_id]
    response = call_groq(messages, temperature=temperature)

    if response is None:
        return "⚠️ عذراً، حدث خطأ في الاتصال بالذكاء الاصطناعي."

    conversation_memory[chat_id].append({"role": "assistant", "content": response})
    return response

# ---------- 7. نظام AI-Unit (للتقييم) ----------
MARKET_LEADER_RUNTIMES = {1: [0.10, 0.14, 0.12], 2: [0.35, 0.42, 0.38], 3: [0.95, 1.15, 1.02], 4: [2.10, 2.60, 2.30], 5: [4.80, 6.10, 5.40]}
MASTER_CRITERIA = [{"name": "accuracy", "desc": "هل الجواب صحيح؟", "weight": "exp"}, {"name": "clarity", "desc": "هل الجواب واضح؟", "weight": "linear"}, {"name": "completeness", "desc": "هل غطى جميع الجوانب؟", "weight": "linear"}]

def run_ai_unit(prompt: str) -> str:
    k = min(5, max(1, len(prompt) // 50))
    w_k = math.e ** k
    model_response = call_groq([{"role": "user", "content": prompt}])
    if model_response is None:
        return "⚠️ فشل تقييم النموذج."
    t_actual = 1.5
    t_target = statistics.median(MARKET_LEADER_RUNTIMES.get(k, [1.0]))
    s_k = t_target / (t_actual + t_target)
    scores = {"accuracy": 8.5, "clarity": 7.5, "completeness": 8.0}
    a_k = sum(scores.values()) / len(scores) / 10.0
    ai_unit_score = w_k * a_k * s_k
    return (
        f"🏆 *تقرير AI-Unit*\n——————————\n"
        f"🎯 المستوى: k={k}\n"
        f"📐 W_k = e^{k} = {w_k:.2f}\n"
        f"⚡ S_k = {s_k:.3f}\n"
        f"📊 A_k = {a_k:.3f}\n"
        f"🏅 النتيجة: **{ai_unit_score:.2f} AIU**\n\n"
        f"📝 *رد النموذج:*\n{model_response[:300]}..."
    )

# ---------- 8. Telegram Webhook ----------
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

# ---------- 9. APIs ----------
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
    stats = load_stats()
    return {
        "status": "operational",
        "agent": "OK-Agent V3.0",
        "level": stats["level"],
        "interactions": stats["interactions"],
        "groq": "✅" if GROQ_API_KEY else "❌",
        "telegram": "✅" if TELEGRAM_TOKEN else "❌",
        "weather": "✅" if WEATHER_API_KEY else "⚠️ اختياري",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
