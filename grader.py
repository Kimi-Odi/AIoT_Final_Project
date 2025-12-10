# ================================================================
# grader.py — AI 虛擬面試官評分模組（含語音特徵調整 + 語音改善建議）
# ================================================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api.getenv("OPENAI_API_KEY"))

# ================================================================
# 🔹 語音特徵調整（B 功能）
# ================================================================
def speech_feature_adjustment(features):
    """
    將語音特徵映射為 0.6～1.0 評分係數
    回傳 float，影響 communication 與 structure 分數
    """
    if not features:
        return 1.0  # 無語音 → 不調整

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    # ----------------------------------------
    # WPM 語速（100~180 為正常）
    # ----------------------------------------
    if wpm < 80:
        wpm_score = 0.7
    elif 80 <= wpm <= 180:
        wpm_score = 1.0
    else:
        wpm_score = 0.8

    # ----------------------------------------
    # 停頓（越少越好）
    # ----------------------------------------
    if silence < 0.1:
        silence_score = 1.0
    elif silence < 0.25:
        silence_score = 0.85
    else:
        silence_score = 0.65

    # ----------------------------------------
    # 音量穩定度（0~1）
    # ----------------------------------------
    stability_score = min(max(stability, 0.0), 1.0)

    # ----------------------------------------
    # 填充詞（越少越好）
    # ----------------------------------------
    if filler < 0.02:
        filler_score = 1.0
    elif filler < 0.05:
        filler_score = 0.8
    else:
        filler_score = 0.65

    final = (wpm_score + silence_score + stability_score + filler_score) / 4
    return round(final, 3)

# ================================================================
# 🔹 語音改善建議（D 功能）
# ================================================================
def generate_speech_feedback(features):
    if not features:
        return "本次未提供語音回答，因此無法產生語音表達建議。"

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    fb = []

    # 語速
    if wpm < 100:
        fb.append(f"- 語速 {wpm} WPM：偏慢，可多練習口語反應。")
    elif wpm > 180:
        fb.append(f"- 語速 {wpm} WPM：偏快，建議放慢語速。")
    else:
        fb.append(f"- 語速 {wpm} WPM：良好。")

    # 停頓
    if silence > 0.25:
        fb.append(f"- 停頓比例 {silence}：停頓略多，建議先思考再回答。")
    else:
        fb.append(f"- 停頓比例 {silence}：自然。")

    # 音量
    if stability < 0.6:
        fb.append(f"- 音量穩定度 {stability}：音量起伏明顯，可加強穩定度。")
    else:
        fb.append(f"- 音量穩定度 {stability}：良好。")

    # 填充詞
    if filler > 0.05:
        fb.append(f"- 填充詞比例 {filler}：口頭禪偏多，建議練習更流暢的口語。")
    else:
        fb.append(f"- 填充詞比例 {filler}：正常。")

    fb.append("\n建議每日錄音練習 3~5 分鐘，會明顯改善語音表達。")

    return "\n".join(fb)


# ================================================================
# 🔹 逐題評分：技術 / 表達 / 結構 / 相關性 / 解題能力 / 潛力
# ================================================================
def grade_single_qa(question, answer, speech_features=None):
    prompt = f"""
你是一位專業面試官，請針對候選人的回答進行逐題評分。
請依 6 個面向評 1~5 分：

- technical：技術深度
- communication：表達清晰度
- structure：回答結構
- relevance：是否答在題目上（答非所問給 1~2 分）
- problem_solving：問題分析能力
- growth_potential：學習與成長潛力

題目：{question}
回答：{answer}

請以 JSON 回傳：
{{
  "technical": x,
  "communication": x,
  "structure": x,
  "relevance": x,
  "problem_solving": x,
  "growth_potential": x,
  "feedback": "一句話回饋"
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    data = json.loads(resp.choices[0].message.content)

    # ⭐ 將語音特徵納入 communication + structure
    if speech_features:
        factor = speech_feature_adjustment(speech_features)

        data["communication"] = round(data["communication"] * factor, 2)
        data["structure"] = round(data["structure"] * (0.7 + factor * 0.3), 2)

    return data


# ================================================================
# 🔹 整場面試總評
# ================================================================
def grade_interview(qa_list, job_role, resume_info=None, speech_features=None):

    per_question = []

    # --- (1) 逐題評分 ---
    for qa in qa_list:
        score = grade_single_qa(
            qa["question"],
            qa["answer"],
            speech_features=speech_features
        )
        per_question.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "score": score,
            "feedback": score["feedback"]
        })

    # --- (2) 計算整體六向度平均 ---
    n = len(per_question)
    overall = {
        "technical": 0,
        "communication": 0,
        "structure": 0,
        "relevance": 0,
        "problem_solving": 0,
        "growth_potential": 0,
    }

    for item in per_question:
        s = item["score"]
        for key in overall:
            overall[key] += s[key]

    for key in overall:
        overall[key] = round(overall[key] / n, 2)

    # --- (3) 整體總結（LLM 生成） ---
    summary_prompt = f"""
請根據以下面試分數（1~5）撰寫 3~5 句繁體中文整體評論：

職缺：{job_role}
分數：{overall}

不要列點，只要一段流暢評論。
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )

    overall["summary"] = resp.choices[0].message.content.strip()

    return {
        "overall": overall,
        "per_question": per_question
    }
