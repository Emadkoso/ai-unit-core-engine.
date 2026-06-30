

import statistics
import time
import json
import random
import re
import os
from typing import Dict, List, Optional, Callable, Tuple

import requests


# ==========================================
# 0. ملف الإعدادات الافتراضي
# ==========================================
# [إصلاح ١] اسم نموذج Gemini أصبح في الإعدادات لا مكتوباً داخل الكود.
#            "gemini-flash-latest" alias يتحدّث تلقائياً مع كل إصدار جديد من جوجل
#            بدل تثبيت اسم نموذج سيُهمَل بعد أشهر مثل ما حدث مع gemini-pro.
# [إصلاح ٨] أُضيف judges كقائمة قابلة للتوسعة — Gemini مفعّل افتراضياً،
#            Groq معطّل لكنه جاهز بسطر واحد. هذا يحوّل "الحَكَم الواحد"
#            إلى هيئة فعلية إن فعّلتها، بدل اسم بلا مسمى.
# [إصلاح ٩] أُضيف difficulty_rubric — وصف إلزامي لكل مستوى صعوبة بدل رقم
#            عشوائي قابل للتلاعب الكامل.
CONFIG_TEMPLATE = {
    "judges": {
        "gemini": {
            "enabled": True,
            "model": "gemini-flash-latest",
            "api_version": "v1beta",
            "api_key_env": "GEMINI_API_KEY"
        },
        "groq": {
            "enabled": False,
            "model": "llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY"
        }
    },
    "market_leader_runtimes": {
        "1": [0.10, 0.14, 0.12],
        "2": [0.35, 0.42, 0.38],
        "3": [0.95, 1.15, 1.02],
        "4": [2.10, 2.60, 2.30],
        "5": [4.80, 6.10, 5.40]
    },
    "weights": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
    "difficulty_rubric": {
        "1": "اكتب هنا: ما الذي يجعل مهمة من المستوى ١ بالتحديد؟",
        "2": "اكتب هنا تعريف المستوى ٢",
        "3": "اكتب هنا تعريف المستوى ٣",
        "4": "اكتب هنا تعريف المستوى ٤",
        "5": "اكتب هنا تعريف المستوى ٥"
    },
    "pass_threshold": 6.0
}


def load_or_create_config(path: str = "config.json") -> dict:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(CONFIG_TEMPLATE, f, indent=4, ensure_ascii=False)
        print(f"✅ تم إنشاء {path} — عدّل النماذج وملف difficulty_rubric قبل الاستخدام الجاد")
        return json.loads(json.dumps(CONFIG_TEMPLATE))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# 1. محرك الحسابات الرياضية
# ==========================================
class AIUnitEngine:
    def __init__(self, config: dict):
        self.market_leader_runtimes = config.get("market_leader_runtimes", {})
        self.weights = config.get("weights", {})
        self.difficulty_rubric = config.get("difficulty_rubric", {})

    def calculate_difficulty_weight(self, k: int) -> float:
        base_weight = k ** 2
        multiplier = self.weights.get(str(k), 1.0)
        return base_weight * multiplier

    def calculate_speed_factor(self, k: int, t_actual: float) -> float:
        # [إصلاح ٢] الخطأ الأصلي: `if k in self.market_leader_runtimes` يقارن
        # رقماً صحيحاً (1) بمفاتيح JSON النصية ("1") — تساوٍ يفشل دائماً صامتاً،
        # فتُتجاهل بيانات "الشركات الرائدة" كلياً. التحويل str(k) هنا يصلح هذا
        # بنفس الطريقة التي كانت calculate_difficulty_weight تفعلها بالفعل بشكل صحيح.
        runtimes = self.market_leader_runtimes.get(str(k))
        t_target = statistics.median(runtimes) if runtimes else float(k * 1.5)
        return min(t_target / (t_actual + t_target), 1.0)

    def rubric_is_defined(self, k: int) -> bool:
        text = self.difficulty_rubric.get(str(k), "").strip()
        return bool(text) and "اكتب هنا" not in text


# ==========================================
# 2. الحُكّام — واجهة موحّدة، كل حَكَم كلاس مستقل
# ==========================================
RUBRIC_KEYS = ["accuracy", "clarity", "creativity", "conciseness"]

RUBRIC_PROMPT_AR = """قيّم النص التالي من 0 إلى 10 على المعايير الأربعة، وأعد JSON فقط بلا أي نص إضافي حوله:

النص: {content}

{{"accuracy": رقم, "clarity": رقم, "creativity": رقم, "conciseness": رقم}}"""


class JudgeResult:
    def __init__(self, judge_name, scores, avg_score, is_real, error=None):
        self.judge_name = judge_name
        self.scores = scores
        self.avg_score = avg_score
        self.is_real = is_real  # هل هذا تقييم حقيقي من النموذج أم احتياطي بدائي؟
        self.error = error


def _heuristic_fallback(text: str) -> Dict[str, float]:
    """
    تقييم احتياطي بدائي جداً — يقيس تنوّع المفردات وكثافة الترقيم فقط،
    لا صحة المعلومة. يُستخدم فقط حين يفشل كل الحُكّام الحقيقيين،
    ويُعلَّم بوضوح (is_real=False) في كل نتيجة بدل التظاهر بأنه تقييم حقيقي.
    """
    words = text.split()
    word_count = max(1, len(words))
    unique_ratio = len(set(words)) / word_count
    punctuation_density = sum(1 for c in text if c in ".,!?") / word_count
    score = min(10.0, unique_ratio * 5 + punctuation_density * 30)
    return {k: round(score, 2) for k in RUBRIC_KEYS}


def _extract_json_scores(raw_text: str) -> Dict[str, float]:
    """
    [إصلاح ٥] الأصل كان يقتطع كل شيء بين أول '{' وآخر '}' في الرد كاملاً —
    هش جداً، أي قوس إضافي في تعليق النموذج يكسره. هنا نبحث عن كل كائنات
    JSON المحتملة، ونقبل فقط أول واحد يحوي المفاتيح الأربعة بقيم رقمية
    ضمن 0-10. أي فشل يرفع استثناء واضح بدل تمرير بيانات فاسدة بصمت.
    """
    candidates = re.findall(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if not candidates:
        raise ValueError("لم يتم العثور على أي كائن JSON في رد النموذج")

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            cleaned = {}
            for key in RUBRIC_KEYS:
                if key not in data:
                    raise ValueError(f"المفتاح '{key}' غير موجود")
                value = float(data[key])
                if not (0 <= value <= 10):
                    raise ValueError(f"قيمة '{key}' خارج النطاق 0-10: {value}")
                cleaned[key] = value
            return cleaned
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_error = e
            continue
    raise ValueError(f"لا يوجد JSON صالح مطابق للمعايير الأربعة: {last_error}")


class GeminiJudge:
    def __init__(self, judge_config: dict):
        self.model = judge_config.get("model", "gemini-flash-latest")
        self.api_version = judge_config.get("api_version", "v1beta")
        self.api_key = os.getenv(judge_config.get("api_key_env", "GEMINI_API_KEY"))

    def evaluate(self, content: str) -> JudgeResult:
        if not self.api_key:
            scores = _heuristic_fallback(content)
            return JudgeResult("gemini", scores, sum(scores.values()) / len(scores),
                                is_real=False, error="لا يوجد GEMINI_API_KEY في البيئة")

        url = (f"https://generativelanguage.googleapis.com/{self.api_version}"
               f"/models/{self.model}:generateContent?key={self.api_key}")
        # [إصلاح ٧] أُزيلت safetySettings: BLOCK_NONE الإجبارية على الفئات الأربع.
        # تقييم جودة نص لا يحتاج تعطيل فلاتر الكراهية/المحتوى الجنسي/الخطر —
        # نترك إعدادات Google الافتراضية المعقولة.
        payload = {"contents": [{"parts": [{"text": RUBRIC_PROMPT_AR.format(content=content)}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            scores = _extract_json_scores(raw_text)
            return JudgeResult("gemini", scores, sum(scores.values()) / len(scores), is_real=True)
        except Exception as e:
            # [إصلاح ٤] الأصل: `except:` فارغ يبتلع الخطأ ويعيد 5.0 صامتاً —
            # تبدو كنتيجة حقيقية رغم أنها مُختلقة بالكامل. هنا: نتدهور بنفس
            # اللطف، لكن نُسجّل is_real=False والسبب الحقيقي صراحة.
            scores = _heuristic_fallback(content)
            return JudgeResult("gemini", scores, sum(scores.values()) / len(scores),
                                is_real=False, error=str(e))


class GroqJudge:
    """
    حَكَم ثانٍ مستقل فعلياً (مزوّد مختلف، نموذج مختلف) — يحوّل الكلاس من
    "حَكَم واحد باسم جماعة" إلى هيئة محلفين حقيقية إذا فعّلتها في config.json.
    Groq توفر طبقة مجانية سخية وواجهة متوافقة مع OpenAI.
    """
    def __init__(self, judge_config: dict):
        self.model = judge_config.get("model", "llama-3.3-70b-versatile")
        self.api_key = os.getenv(judge_config.get("api_key_env", "GROQ_API_KEY"))

    def evaluate(self, content: str) -> JudgeResult:
        if not self.api_key:
            scores = _heuristic_fallback(content)
            return JudgeResult("groq", scores, sum(scores.values()) / len(scores),
                                is_real=False, error="لا يوجد GROQ_API_KEY في البيئة")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": RUBRIC_PROMPT_AR.format(content=content)}],
            "temperature": 0.2
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
            scores = _extract_json_scores(raw_text)
            return JudgeResult("groq", scores, sum(scores.values()) / len(scores), is_real=True)
        except Exception as e:
            scores = _heuristic_fallback(content)
            return JudgeResult("groq", scores, sum(scores.values()) / len(scores),
                                is_real=False, error=str(e))


JUDGE_CLASSES = {"gemini": GeminiJudge, "groq": GroqJudge}


# ==========================================
# 3. هيئة المحلفين — الآن جمع فعلي، لا حَكَم واحد
# ==========================================
class DigitalJury:
    def __init__(self, config: dict):
        self.judges = []
        for name, jconf in config.get("judges", {}).items():
            if jconf.get("enabled") and name in JUDGE_CLASSES:
                self.judges.append(JUDGE_CLASSES[name](jconf))
        if not self.judges:
            self.judges.append(GeminiJudge(config.get("judges", {}).get("gemini", {})))

    def mask_model_identity(self, model_output: str) -> dict:
        return {"anonymous_id": f"Model_{random.randint(10000, 99999)}",
                "response_content": model_output}

    def evaluate(self, content: str) -> dict:
        results = [judge.evaluate(content) for judge in self.judges]
        avg_scores = [r.avg_score for r in results]
        consensus_avg = statistics.mean(avg_scores)
        # تشتت الآراء بين الحُكّام مؤشر ثقة فعلي: حُكّام متفقون = نتيجة أوثق،
        # حُكّام متباعدون = إشارة بأن هذه الحالة تحتاج مراجعة بشرية. هذا غير
        # موجود إطلاقاً في كود "حَكَم واحد" — لا يمكن قياس الاتفاق بدون أكثر من رأي.
        disagreement = statistics.stdev(avg_scores) if len(avg_scores) > 1 else 0.0
        return {
            "judges": [
                {"name": r.judge_name, "scores": r.scores, "avg": round(r.avg_score, 2),
                 "real_evaluation": r.is_real, "error": r.error}
                for r in results
            ],
            "consensus_avg_score": round(consensus_avg, 2),
            "judge_disagreement": round(disagreement, 3),
            "any_real_judge": any(r.is_real for r in results)
        }


# ==========================================
# 4. خط الإنتاج الرئيسي
# ==========================================
class AIUnitMasterPipeline:
    def __init__(self, config: dict):
        self.engine = AIUnitEngine(config)
        self.jury = DigitalJury(config)
        self.pass_threshold = config.get("pass_threshold", 6.0)
        self.results_history = []

    def _score_one(self, k: int, t_actual: float, raw_output: str) -> dict:
        if not self.engine.rubric_is_defined(k):
            print(f"⚠️  المستوى k={k} بلا تعريف صعوبة في difficulty_rubric — "
                  f"الوزن سيُحسب لكنه بلا معنى تشغيلي حتى تُعرّفه.")

        masked = self.jury.mask_model_identity(raw_output)
        jury_result = self.jury.evaluate(masked["response_content"])

        w_k = self.engine.calculate_difficulty_weight(k)
        s_k = self.engine.calculate_speed_factor(k, t_actual)

        # [إصلاح ٣] الأصل: vote ثنائي (٠/١) عند عتبة ٦.٠ مضروب أيضاً في
        # avg_score/10 المستمر → نتيجة ٥.٩٩ تأخذ صفراً مطلقاً، ونتيجة ٦.٠٠
        # تقفز فجأة لنحو ٦٠٪ من الوزن. هنا quality_factor مستمر من ٠ إلى ١
        # بلا قفزة، و"النجاح" يُسجَّل كحقل منفصل لا يُصفّر الدرجة.
        quality_factor = jury_result["consensus_avg_score"] / 10
        tier_score = w_k * s_k * quality_factor

        return {
            "Assigned_ID": masked["anonymous_id"],
            "Jury_Result": jury_result,
            "Difficulty_Weight_W_k": round(w_k, 2),
            "Speed_Factor_S_k": round(s_k, 4),
            "Quality_Factor": round(quality_factor, 4),
            "Passed_Quality_Bar": jury_result["consensus_avg_score"] >= self.pass_threshold,
            "Score_Earned": round(tier_score, 4)
        }

    def run_full_evaluation(
        self,
        model_name: str,
        test_suite: List[Dict],
        model_call_fn: Optional[Callable[[str], Tuple[str, float]]] = None,
        num_trials: int = 1
    ) -> Dict:
        """
        model_call_fn: دالة اختيارية (prompt) -> raw_output. إن وُفِّرت، يُستدعى
        نموذج حقيقي مباشرة ويُقاس زمنه فعلياً — [إصلاح ٦: لا تشغيل حقيقي للنموذج
        في الأصل، كل شيء كان نصوصاً وأزمنة ثابتة يدوياً]. بدونها، يعمل الكود
        بنفس بيانات test_suite التجريبية الثابتة كما في الأصل.

        num_trials: عدد التكرارات لكل مهمة — [إصلاح: استيراد statistics كان
        شبه معطّل]. مفيد فعلياً فقط مع model_call_fn حي (نص ثابت يتكرر بنفس
        القيمة دائماً)، فيحسب المتوسط والانحراف المعياري بدل عيّنة n=1 وحيدة.
        """
        total_aiu_score = 0.0
        breakdown = {}
        real_judge_calls = 0
        fallback_calls = 0

        for run in test_suite:
            k = run.get("k", 1)
            prompt = run.get("prompt", run.get("raw_output", ""))
            trial_scores = []
            last_result = None

            for _ in range(max(1, num_trials)):
                if model_call_fn:
                    start = time.time()
                    raw_output = model_call_fn(prompt)
                    t_actual = time.time() - start
                else:
                    raw_output = run.get("raw_output", "")
                    t_actual = run.get("t_actual", 1.0)

                last_result = self._score_one(k, t_actual, raw_output)
                trial_scores.append(last_result["Score_Earned"])
                for j in last_result["Jury_Result"]["judges"]:
                    real_judge_calls += 1 if j["real_evaluation"] else 0
                    fallback_calls += 0 if j["real_evaluation"] else 1

            mean_score = statistics.mean(trial_scores)
            std_score = statistics.stdev(trial_scores) if len(trial_scores) > 1 else 0.0
            total_aiu_score += mean_score

            breakdown[f"Level_{k}"] = {
                "Trials": num_trials,
                "Mean_Score": round(mean_score, 4),
                "Score_StdDev": round(std_score, 4),
                "Sample_Trial_Detail": last_result
            }

        result_summary = {
            "Model_Name": model_name,
            "Consolidated_AI_Unit_Score": round(total_aiu_score, 4),
            "Total_Tasks": len(test_suite),
            # [إصلاح ٦] أُزيلت "$0.0000" المزيّفة التي لم تكن تُحسب فعلياً أبداً.
            # في الطبقة المجانية، العدد المفيد فعلاً هو عدد الاستدعاءات الحقيقية
            # مقابل الاحتياطية — هذا ما يحدد هل نتائجك موثوقة أم وهمية.
            "Real_Judge_Calls": real_judge_calls,
            "Fallback_Heuristic_Calls": fallback_calls,
            "Evaluation_Breakdown": breakdown
        }
        self.results_history.append(result_summary)
        return result_summary


# ==========================================
# 5. تشغيل تجريبي
# ==========================================
if __name__ == "__main__":
    config = load_or_create_config("config.json")
    pipeline = AIUnitMasterPipeline(config)

    test_suite = [
        {"k": 1, "t_actual": 0.11, "raw_output": "Hello! How can I assist you today?"},
        {"k": 3, "t_actual": 1.20, "raw_output": "The correlation coefficient indicates a strong linear relationship between investment and economic growth, with r-squared 0.89."},
        {"k": 5, "t_actual": 5.90, "raw_output": "The optimal quantum-safe architecture requires continuous noise-injection filters and post-quantum cryptographic algorithms."},
        {"k": 2, "t_actual": 0.45, "raw_output": "AI-Unit engine is designed to evaluate AI models using an advanced mathematical framework and a blind jury system."},
    ]

    print("🚀 تشغيل AI-Unit Core Engine V3.0 (نسخة مُصلَّحة)...\n")
    final_result = pipeline.run_full_evaluation("Quantum-Model-V2", test_suite, num_trials=1)
    print(json.dumps(final_result, indent=4, ensure_ascii=False))

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print("\n✅ النتائج محفوظة في evaluation_results.json")
    
