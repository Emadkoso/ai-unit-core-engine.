# ==============================================================
# آلة الدروبشيبينغ الذاتية – Autonomous Dropshipping Engine
# ==============================================================
# الميزات:
#   • بحث آلي عن المنتجات الرائجة.
#   • تحليل السوق والتسعير الديناميكي.
#   • تسويق تلقائي وإنشاء إعلانات.
#   • استقبال الطلبات عبر واجهات متعددة.
#   • شراء تلقائي من الموردين.
#   • شحن آلي ومتابعة الطلبات.
#   • تقييم وتحسين الاستراتيجية ذاتياً.
#   • توسع أسّي (S = ∑ 2^k) في المنتجات والعملاء.
# ==============================================================

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
import requests
import time
import math
import random
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
from enum import Enum

app = FastAPI(title="Autonomous Dropshipping Engine", version="4.0")

# ---------- الإعدادات ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
SHIPPING_API_KEY = os.environ.get("SHIPPING_API_KEY", "")

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# الملفات
STATE_FILE = "autonomous_state.json"
PRODUCTS_FILE = "products_auto.json"
ORDERS_FILE = "orders_auto.json"
CUSTOMERS_FILE = "customers_auto.json"
ANALYTICS_FILE = "analytics_auto.json"

# ---------- 1. نظام الحالة الذاتية (Self-State) ----------
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "level": 0,
        "products_count": 0,
        "customers_count": 0,
        "orders_count": 0,
        "total_revenue": 0.0,
        "total_profit": 0.0,
        "last_update": datetime.now().isoformat(),
        "active": True
    }

def save_state(state: dict):
    state["last_update"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def update_level(state: dict) -> dict:
    """يرفع مستوى النظام حسب المعادلة الأسية."""
    new_level = min(10, int(math.log2(state["orders_count"] + 1)))
    if new_level > state["level"]:
        state["level"] = new_level
        state["products_count"] = 2 ** new_level
        state["customers_count"] = 2 ** new_level
    return state

# ---------- 2. محرك البحث عن المنتجات (Product Sourcing Engine) ----------
def search_products(keyword: str = None) -> List[Dict]:
    """يبحث عن منتجات رائجة من الموردين (محاكاة)."""
    # في الواقع، سيكون تكامل مع AliExpress API، Amazon API، إلخ.
    # هنا محاكاة ذكية.
    categories = ["إلكترونيات", "ملابس", "منزل", "جمال", "رياضة"]
    products = []
    for i in range(5):
        products.append({
            "id": f"P{int(time.time())}{i}",
            "name": f"{random.choice(categories)} {random.choice(['فاخر', 'ذكي', 'مريح', 'عملي', 'جديد'])}",
            "price": round(random.uniform(20, 500), 2),
            "supplier": random.choice(["AliExpress", "Amazon", "Noon"]),
            "link": f"https://supplier.com/product/{i}",
            "category": random.choice(categories),
            "stock": random.randint(10, 100),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews_count": random.randint(10, 500)
        })
    return products

# ---------- 3. محرك تحليل السوق (Market Analysis Engine) ----------
def analyze_product(product: Dict) -> Dict:
    """يحلل المنتج ويقرر مدى ربحية بيعه."""
    profit_margin = random.uniform(0.1, 0.4)  # هامش ربح 10%-40%
    demand_score = random.uniform(0.3, 1.0)  # درجة الطلب
    competition_score = random.uniform(0.1, 0.9)  # درجة المنافسة
    
    # حساب السعر الأمثل
    optimal_price = product["price"] * (1 + profit_margin)
    
    return {
        "product_id": product["id"],
        "profit_margin": profit_margin,
        "demand_score": demand_score,
        "competition_score": competition_score,
        "optimal_price": optimal_price,
        "score": (demand_score * 0.6) + ((1 - competition_score) * 0.4),  # درجة الفرصة
        "recommended": demand_score > 0.5 and competition_score < 0.7
    }

# ---------- 4. محرك التسعير الديناميكي (Dynamic Pricing Engine) ----------
def dynamic_pricing(product: Dict, analysis: Dict) -> float:
    """يحدد السعر الأمثل بناءً على تحليل السوق."""
    base_price = product["price"]
    demand_factor = 1 + (analysis["demand_score"] * 0.2)  # زيادة 0-20%
    competition_factor = 1 - (analysis["competition_score"] * 0.1)  # تخفيض 0-10%
    price = base_price * demand_factor * competition_factor * (1 + analysis["profit_margin"])
    return round(price, 2)

# ---------- 5. محرك التسويق التلقائي (Marketing Engine) ----------
def generate_ad_content(product: Dict) -> str:
    """يكتب محتوى إعلاني تلقائياً."""
    prompt = f"اكتب إعلاناً جذاباً لمنتج: {product['name']} بسعر {product['price']} ريال."
    messages = [{"role": "user", "content": prompt}]
    response = call_groq(messages, temperature=0.8)
    return response if response else f"🔥 اكتشف {product['name']} الآن! بسعر مميز."

# ---------- 6. محرك استقبال الطلبات (Order Reception Engine) ----------
def receive_order(customer_name: str, customer_phone: str, customer_address: str, product_id: str) -> Dict:
    """يستقبل الطلب ويبدأ عملية الشراء."""
    products = load_json(PRODUCTS_FILE)
    if product_id not in products:
        return {"success": False, "message": "المنتج غير موجود"}
    
    product = products[product_id]
    # إنشاء الطلب
    orders = load_json(ORDERS_FILE)
    oid = str(int(time.time()))
    orders[oid] = {
        "id": oid,
        "product_id": product_id,
        "customer_name": customer_name,
        "customer_phone": customer_phone,
        "customer_address": customer_address,
        "amount": product["price"],
        "status": "pending",
        "created": datetime.now().isoformat()
    }
    save_json(ORDERS_FILE, orders)
    
    # تحديث الحالة
    state = load_state()
    state["orders_count"] += 1
    state["total_revenue"] += product["price"]
    state = update_level(state)
    save_state(state)
    
    return {"success": True, "order_id": oid, "message": f"✅ تم استلام الطلب #{oid}"}

# ---------- 7. محرك الشراء من المورد (Supplier Purchase Engine) ----------
def purchase_from_supplier(order_id: str) -> Dict:
    """يشترى المنتج من المورد تلقائياً."""
    orders = load_json(ORDERS_FILE)
    if order_id not in orders:
        return {"success": False, "message": "الطلب غير موجود"}
    
    order = orders[order_id]
    # محاكاة الشراء من المورد
    time.sleep(1)  # محاكاة عملية الشراء
    order["status"] = "purchased"
    order["purchase_date"] = datetime.now().isoformat()
    save_json(ORDERS_FILE, orders)
    
    return {"success": True, "message": f"✅ تم شراء المنتج من المورد للطلب #{order_id}"}

# ---------- 8. محرك الشحن (Shipping Engine) ----------
def ship_order(order_id: str) -> Dict:
    """يشحن الطلب تلقائياً عبر شركة الشحن."""
    orders = load_json(ORDERS_FILE)
    if order_id not in orders:
        return {"success": False, "message": "الطلب غير موجود"}
    
    order = orders[order_id]
    # محاكاة الشحن
    tracking_number = f"TRK{int(time.time())}{random.randint(100, 999)}"
    order["status"] = "shipped"
    order["tracking_number"] = tracking_number
    order["shipped_date"] = datetime.now().isoformat()
    save_json(ORDERS_FILE, orders)
    
    return {"success": True, "tracking_number": tracking_number, "message": f"✅ تم شحن الطلب #{order_id}"}

# ---------- 9. محرك التقييم والتحسين (Evaluation Engine) ----------
def evaluate_performance() -> Dict:
    """يحلل الأداء ويقدم توصيات للتحسين."""
    orders = load_json(ORDERS_FILE)
    customers = load_json(CUSTOMERS_FILE)
    state = load_state()
    
    total_orders = len(orders)
    total_revenue = sum(o.get("amount", 0) for o in orders.values())
    total_customers = len(customers)
    avg_order = total_revenue / total_orders if total_orders > 0 else 0
    
    # تقييم الأداء
    performance = {
        "orders_per_day": total_orders / 30 if total_orders > 0 else 0,
        "revenue_per_day": total_revenue / 30 if total_revenue > 0 else 0,
        "customer_retention": random.uniform(0.1, 0.5),  # محاكاة
        "profit_margin": random.uniform(0.1, 0.3),  # محاكاة
    }
    
    # توصيات
    recommendations = []
    if performance["orders_per_day"] < 5:
        recommendations.append("📈 قم بزيادة التسويق لجذب المزيد من العملاء.")
    if performance["customer_retention"] < 0.3:
        recommendations.append("🎯 قدم عروضاً خاصة للعملاء الحاليين.")
    if performance["profit_margin"] < 0.15:
        recommendations.append("💰 ارفع الأسعار قليلاً أو ابحث عن موردين أرخص.")
    
    # تحسين تلقائي
    state["total_profit"] = total_revenue * performance["profit_margin"]
    save_state(state)
    
    return {
        "performance": performance,
        "recommendations": recommendations,
        "score": total_revenue / (state["level"] + 1) if state["level"] > 0 else total_revenue
    }

# ---------- 10. محرك التوسع الذاتي (Self-Expansion Engine) ----------
def expand_business() -> Dict:
    """يوسع الأعمال تلقائياً حسب المعادلة الأسية."""
    state = load_state()
    level = state["level"]
    
    # عدد المنتجات الجديدة المطلوبة
    target_products = 2 ** level
    current_products = len(load_json(PRODUCTS_FILE))
    needed = target_products - current_products
    
    results = []
    for i in range(max(0, needed)):
        # بحث عن منتج جديد
        products = search_products()
        if products:
            product = random.choice(products)
            # تحليل السوق
            analysis = analyze_product(product)
            if analysis["recommended"]:
                # تسعير ديناميكي
                product["price"] = dynamic_pricing(product, analysis)
                # حفظ المنتج
                products_db = load_json(PRODUCTS_FILE)
                products_db[product["id"]] = product
                save_json(PRODUCTS_FILE, products_db)
                results.append(f"✅ تم إضافة المنتج: {product['name']} بسعر {product['price']} ريال")
    
    # تحديث الحالة
    state["products_count"] = len(load_json(PRODUCTS_FILE))
    save_state(state)
    
    return {
        "level": level,
        "products_added": len(results),
        "results": results,
        "message": f"🚀 تم التوسع إلى المستوى {level} مع {state['products_count']} منتج"
    }

# ---------- 11. الدورة الكاملة للنظام (Full Cycle) ----------
def run_autonomous_cycle() -> Dict:
    """دورة تشغيل كاملة للنظام."""
    # 1. التوسع الذاتي
    expansion = expand_business()
    
    # 2. تحسين التسعير (كل 10 منتجات)
    products = load_json(PRODUCTS_FILE)
    for pid, product in list(products.items())[:5]:
        analysis = analyze_product(product)
        new_price = dynamic_pricing(product, analysis)
        product["price"] = new_price
        products[pid] = product
    save_json(PRODUCTS_FILE, products)
    
    # 3. تقييم الأداء
    evaluation = evaluate_performance()
    
    return {
        "expansion": expansion,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat()
    }

# ---------- 12. استدعاء Groq ----------
def call_groq(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return None

# ---------- 13. إدارة الملفات ----------
def load_json(filename: str) -> dict:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_json(filename: str, data: dict):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# ---------- 14. Webhook تلغرام ----------
def send_tg(chat_id: int, text: str) -> None:
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram error: {e}")

@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" not in data or "text" not in data["message"]:
        return {"status": "ok"}
    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip()
    
    # أوامر المستخدم
    if user_text.startswith("/start"):
        reply = "🚀 *مرحباً بكم في آلة الدروبشيبينغ الذاتية!*\nأنا نظام متكامل يدير كل شيء بنفسه. فقط شغلني وانسى."
    elif user_text.startswith("/status"):
        state = load_state()
        reply = f"📊 *حالة النظام*\nالمستوى: {state['level']}\nالمنتجات: {state['products_count']}\nالعملاء: {state['customers_count']}\nالطلبات: {state['orders_count']}\nالإيرادات: {state['total_revenue']:.2f} ريال"
    elif user_text.startswith("/run"):
        result = run_autonomous_cycle()
        reply = f"🔄 *تم تشغيل الدورة الكاملة*\n{json.dumps(result, indent=2, ensure_ascii=False)}"
    elif user_text.startswith("/expand"):
        result = expand_business()
        reply = f"🚀 *تم التوسع*\n{result['message']}"
    elif user_text.startswith("/products"):
        products = load_json(PRODUCTS_FILE)
        if not products:
            reply = "📭 لا توجد منتجات."
        else:
            reply = "📦 *المنتجات:*\n"
            for pid, p in list(products.items())[:10]:
                reply += f"• {p['name']} – {p['price']} ريال\n"
    else:
        # محادثة ذكية
        messages = [{"role": "system", "content": "أنت آلة دروبشيبينغ ذاتية التشغيل."}, {"role": "user", "content": user_text}]
        response = call_groq(messages)
        reply = response if response else "⚠️ عذراً، حدث خطأ."
    
    send_tg(chat_id, reply)
    return {"status": "ok"}

# ---------- 15. واجهات برمجية ----------
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
        <head><title>آلة الدروبشيبينغ الذاتية</title></head>
        <body>
            <h1>🚀 Autonomous Dropshipping Engine</h1>
            <p>النظام يعمل بشكل كامل وآلي.</p>
            <p>الحالة: 🟢 قيد التشغيل</p>
            <a href="/status">عرض الحالة</a>
        </body>
    </html>
    """)

@app.get("/status")
async def status():
    state = load_state()
    return JSONResponse(state)

@app.get("/run")
async def run_cycle():
    return JSONResponse(run_autonomous_cycle())

@app.get("/health")
async def health():
    state = load_state()
    return {
        "status": "operational",
        "version": "4.0",
        "level": state["level"],
        "products": state["products_count"],
        "orders": state["orders_count"],
        "revenue": state["total_revenue"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
