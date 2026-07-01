from fastapi import FastAPI, Request
import os
import time
import statistics
import requests

app = FastAPI(title="AI-Unit MVP", version="1.0")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

TEST_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
GROQ_JUDGE_MODEL = "llama-3.1-8b-instant"


def send_tg(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text
            },
            timeout=30
        )
    except:
        pass


def call_openrouter(prompt: str, model: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1200
    }

    start = time.time()

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90
    )

    r.raise_for_status()

    duration = time.time() - start

    answer = r.json()["choices"][0]["message"]["content"]

    return answer, duration


def call_groq(prompt: str, model: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1200
    }

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90
    )

    r.raise_for_status()

    answer = r.json()["choices"][0]["message"]["content"]

    return answer


def estimate_k(prompt: str):
    score = 1
    text = prompt.lower()

    words = len(prompt.split())

    if words > 100:
        score += 2
    elif words > 40:
        score += 1

    technical = [
        "python",
        "api",
        "database",
        "algorithm",
        "program",
        "design",
        "architecture",
        "برمجة",
        "خوارزمية",
        "قاعدة بيانات",
        "تحليل",
        "صمم"
    ]

    reasoning = [
        "prove",
        "analyze",
        "compare",
        "optimize",
        "برهن",
        "حلل",
        "قارن",
        "حسن"
    ]

    for x in technical:
        if x in text:
            score += 2
            break

    for x in reasoning:
        if x in text:
            score += 2
            break

    return max(1, min(score, 10))


def judge_prompt(question, answer):
    return f"""
You are an AI judge.

Question:
{question}

Answer:
{answer}

Score the answer from 0.0 to 1.0 considering:
- correctness
- reasoning
- clarity

Return ONLY one number between 0 and 1.
"""


def parse_score(text):
    try:
        value = float(text.strip())
        return max(0.0, min(value, 1.0))
    except:
        return 0.5


def evaluate_quality(question, answer):
    scores = []

    try:
        result = call_groq(
            judge_prompt(question, answer),
            GROQ_JUDGE_MODEL
        )
        scores.append(parse_score(result))
    except:
        pass

    try:
        result, _ = call_openrouter(
            judge_prompt(question, answer),
            OPENROUTER_JUDGE_MODEL
        )
        scores.append(parse_score(result))
    except:
        pass

    if not scores:
        return 0.5

    return statistics.mean(scores)


def speed_factor(actual, target=3.0):
    return min(
        1.0,
        target / (actual + target)
    )


def compute_aiu(k, quality, speed, consistency=1.0):
    weight = 2 ** k

    aiu = (
        weight
        * quality
        * speed
        * consistency
    )

    return aiu, weight


@app.post("/tg-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" not in data:
        return {"status": "ignored"}

    if "text" not in data["message"]:
        return {"status": "ignored"}

    chat_id = data["message"]["chat"]["id"]
    prompt = data["message"]["text"]

    send_tg(chat_id, "⚡ Running AI-Unit...")

    try:
        answer, duration = call_openrouter(
            prompt,
            TEST_MODEL
        )

        k = estimate_k(prompt)

        quality = evaluate_quality(
            prompt,
            answer
        )

        speed = speed_factor(duration)

        consistency = 1.0

        aiu, weight = compute_aiu(
            k,
            quality,
            speed,
            consistency
        )

        report = f"""
🏆 AI-Unit Report

Difficulty k: {k}
Weight (2^k): {weight}

Quality: {quality:.3f}
Speed: {speed:.3f}
Consistency: {consistency:.3f}

Execution Time: {duration:.3f} sec

AIU Score: {aiu:.3f}

====================

Model Response:

{answer[:1200]}
"""

        send_tg(chat_id, report)

    except Exception as e:
        send_tg(chat_id, f"❌ Error\n\n{str(e)}")

    return {"status": "ok"}


@app.get("/")
def home():
    return {
        "status": "AI-Unit Running"
    }
