# ================================================================
# grader.py — AI 虛擬面試官評分模組（含語音特徵調整 + 語音改善建議）
# ================================================================

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================================================
# 🔹 語音特徵調整（你之前要的 B 功能）
# ================================================================
def speech_feature_adjustment(features):
    """
    將語音特徵映射為 0.6～1.0 評分係數
    回傳 float，影響 communication 與 structure 分數
    """
    if not features:
        return 1.0  # 沒有語音 → 不調整

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    # -------------------------
    # WPM 語速
    # 理想：100～180
    # -------------------------
    if wpm < 80:
        wpm_score = 0.7
    elif 80 <= wpm <= 180:
        wpm_score = 1.0
    else:
        wpm_score = 0.8

    # -------------------------
    # 停頓比例（越少越好）
    # -------------------------
    if silence < 0.10:
        silence_score = 1.0
    elif silence < 0.25:
        silence_score = 0.85
    else:
        silence_score = 0.65

    # -------------------------
    # 音量穩定度（0~1）
    # -------------------------
    stability_score = max(min(stability, 1.0), 0.0)

    # -------------------------
    # 填充詞（越少越好）
    # -------------------------
    if filler < 0.02:
        filler_score = 1.0
    elif filler < 0.05:
        filler_score = 0.8
    else:
        filler_score = 0.65

    final = (wpm_score + silence_score + stability_score + filler_score) / 4
    return round(final, 3)


# ================================================================
# 🔹 AI 給語音改善建議（你選的 D 功能）
# ================================================================
def generate_speech_feedback(features):
    if not features:
        return "本次未提供語音回答，因此無法產生語音表達建議。"

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    feedback = []

    # 語速
    if wpm < 100:
        feedback.append(f"- 語速 {wpm} WPM：偏慢，可多練習口語流暢度。")
    elif wpm > 180:
        feedback.append(f"- 語速 {wpm} WPM：偏快，建議放慢讓語句更清晰。")
    else:
        feedback.append(f"- 語速 {wpm} WPM：表現良好。")

    # 停頓
    if silence > 0.25:
        feedback.append(f"- 停頓比例 {silence}：停頓略多，建議先組織語句再回答。")
    else:
        feedback.append(f"- 停頓比例 {silence}：自然、表現正常。")

    # 音量穩定度
    if stability < 0.6:
        feedback.append(f"- 音量穩定度 {stability}：音量起伏較大，可練習更穩定的語調。")
    else:
        feedback.append(f"- 音量穩定度 {stability}：良好。")

    # 填充詞
    if filler > 0.05:
        feedback.append(f"- 填充詞比例 {filler}：'嗯'、'呃' 使用偏多，建議控制口頭禪。")
    else:
        feedback.append(f"- 填充詞比例 {filler}：使用正常。")

    feedback.append("\n建議每天錄音練習 5 分鐘，可以明顯改善語音表達。")

    return "\n".join(feedback)


# ================================================================
# 🔹 問題逐題評分（AI）
# ================================================================
def grade_single_qa(question, answer, speech_features=None):
    """
    使用 GPT 分析單題回答→ 回傳分數 + 回饋
    """
    prompt = f"""
你是一位專業面試官，請針對候選人的回答進行逐題評分。
請依「技術」、「表達」、「結構」、「相關性」、「解題能力」、「成長潛力」六項評分，每項 0~5 分。

題目：{question}
回答：{answer}

請回傳 JSON：
{{
  "technical": 分數0~5,
  "communication": 分數0~5,
  "structure": 分數0~5,
  "relevance": 分數0~5,
  "problem_solving": 分數0~5,
  "growth_potential": 分數0~5,
  "feedback": "一句話回饋"
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    data = json.loads(resp.choices[0].message.content)

    # ⭐ 語音特徵調整：communication & structure
    if speech_features:
        factor = speech_feature_adjustment(speech_features)
        data["communication"] = round(data["communication"] * factor, 2)
        data["structure"] = round(data["structure"] * (0.7 + 0.3 * factor), 2)

    return data


# ================================================================
# 🔹 整場面試評分（整合逐題）
# ================================================================
def grade_interview(qa_list, job_role, resume_info=None, speech_features=None):

    per_question_results = []

    # ----------- 逐題分析 -----------
    for qa in qa_list:
        score = grade_single_qa(
            qa["question"], qa["answer"], speech_features=speech_features
        )
        per_question_results.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "score": score,
            "feedback": score["feedback"]
        })

    # ----------- 整體平均 -----------
    n = len(per_question_results)
    overall = {
        "technical": 0,
        "communication": 0,
        "structure": 0,
        "relevance": 0,
        "problem_solving": 0,
        "growth_potential": 0,
    }

    for item in per_question_results:
        s = item["score"]
        for k in overall:
            overall[k] += s[k]

    for k in overall:
        overall[k] = round(overall[k] / n, 2)

    # ----------- 整體評論（AI）-----------
    overall_prompt = f"""
請根據以下面試分數，生成一段 100 字以內的整體評論（繁體中文）。

職缺：{job_role}
逐題平均分數如下：
{overall}

請給出總結，不要列點。
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": overall_prompt}]
    )
    overall_summary = resp.choices[0].message.content.strip()
    overall["summary"] = overall_summary

    return {
        "overall": overall,
        "per_question": per_question_results
    }
