from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import statistics
import time
import json
import random
import re
import os
from typing import Dict, List, Optional

app = FastAPI(title="AI-Unit Core SaaS API", version="3.0")

# --- الإعدادات الافتراضية للنظام ---
CONFIG = {
    "market_leader_runtimes": {
        "1": [0.10, 0.14, 0.12], "2": [0.35, 0.42, 0.38],
        "3": [0.95, 1.15, 1.02], "4": [2.10, 2.60, 2.30],
        "5": [4.80, 6.10, 5.40]
    },
    "pass_threshold": 6.0
}

# --- نماذج استقبال البيانات (Pydantic Models) ---
class TestCaseInput(BaseModel):
    k: int
    t_actual: float
    raw_output: str

class EvaluationRequest(BaseModel):
    model_name: str
    test_suite: List[TestCaseInput]

# --- محرك الحسابات الرياضية V3 ---
class AIUnitEngine:
    def __init__(self):
        self.market_leader_runtimes = CONFIG["market_leader_runtimes"]

    def calculate_difficulty_weight(self, k: int) -> float:
        return float(k ** 2)

    def calculate_speed_factor(self, k: int, t_actual: float) -> float:
        runtimes = self.market_leader_runtimes.get(str(k))
        t_target = statistics.median(runtimes) if runtimes else float(k * 1.5)
        return min(t_target / (t_actual + t_target), 1.0)

# --- نظام التقييم والتحكيم الاحتياطي الفعال ---
def _heuristic_fallback(text: str) -> Dict[str, float]:
    words = text.split()
    word_count = max(1, len(words))
    unique_ratio = len(set(words)) / word_count
    punctuation_density = sum(1 for c in text if c in ".,!?") / word_count
    score = min(10.0, unique_ratio * 5 + punctuation_density * 30)
    return {k: round(score, 2) for k in ["accuracy", "clarity", "creativity", "conciseness"]}

# --- نقاط الاتصال (Endpoints) للسيرفر ---
@app.get("/")
def home():
    return {"status": "Active", "framework": "AI-Unit Core V3.0"}

@app.post("/api/v1/evaluate")
def evaluate_api(request: EvaluationRequest):
    engine = AIUnitEngine()
    total_aiu_score = 0.0
    breakdown = {}
    
    try:
        for run in request.test_suite:
            k = run.k
            raw_output = run.raw_output
            t_actual = run.t_actual
            
            # تشغيل الفحص الاحتياطي الذكي والآمن حالياً
            scores = _heuristic_fallback(raw_output)
            consensus_avg = sum(scores.values()) / len(scores)
            
            w_k = engine.calculate_difficulty_weight(k)
            s_k = engine.calculate_speed_factor(k, t_actual)
            
            quality_factor = consensus_avg / 10
            tier_score = w_k * s_k * quality_factor
            total_aiu_score += tier_score
            
            breakdown[f"Level_{k}"] = {
                "Assigned_ID": f"Model_{random.randint(10000, 99999)}",
                "Consensus_Avg_Score": round(consensus_avg, 2),
                "Difficulty_Weight_W_k": w_k,
                "Speed_Factor_S_k": round(s_k, 4),
                "Score_Earned": round(tier_score, 4)
            }
            
        return {
            "Model_Name": request.model_name,
            "Consolidated_AI_Unit_Score": round(total_aiu_score, 4),
            "Total_Tasks": len(request.test_suite),
            "Evaluation_Breakdown": breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
      
