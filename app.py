def _call_jury_model(text: str, model_label: str) -> Optional[Dict[str, float]]:
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    system_instruction = (
        "You are an expert AI Jury. Your ONLY role is to evaluate the provided text. "
        "Output ONLY a raw JSON object with exactly these keys: 'accuracy', 'clarity', 'creativity', 'conciseness'. "
        "Values must be numbers from 0.0 to 10.0. No text outside JSON."
    )
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Evaluate: '{text}'"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # تنظيف النص من أي رموز ماركداون قد يضيفها النموذج
            clean_content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_content)
            
            # توحيد المفاتيح لتكون دائماً حروف صغيرة
            normalized_data = {k.lower(): float(v) for k, v in data.items()}
            
            # ضمان وجود كافة المفاتيح بقيمة افتراضية إذا نقصت
            defaults = {"accuracy": 5.0, "clarity": 5.0, "creativity": 5.0, "conciseness": 5.0}
            for k, v in defaults.items():
                if k not in normalized_data:
                    normalized_data[k] = v
            return normalized_data
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"accuracy": 5.0, "clarity": 5.0, "creativity": 5.0, "conciseness": 5.0}
    return None
