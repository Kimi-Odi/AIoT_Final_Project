# grader.py
import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_grading_prompt(
    qas: List[Dict[str, str]],
    job_role: str,
    resume_info: Dict[str, Any] | None = None,
) -> str:
    """
    把整場面試的 QA + 履歷摘要整理成一段文字，給 LLM 當評分輸入。
    """
    lines = []

    lines.append(f"應徵職缺：{job_role}")
    lines.append("")

    if resume_info:
        skills = ", ".join(resume_info.get("skills", []))
        proj_lines = []
        for p in resume_info.get("projects", []):
            title = p.get("title", "")
            desc = p.get("description", "")
            proj_lines.append(f"- {title}: {desc}")

        summary = resume_info.get("summary", "")

        lines.append("[履歷摘要]")
        if skills:
            lines.append(f"技能：{skills}")
        if proj_lines:
            lines.append("專案：")
            lines.extend(proj_lines)
        if summary:
            lines.append(f"履歷總結：{summary}")
        lines.append("")

    lines.append("[面試問答紀錄]")
    for idx, qa in enumerate(qas, start=1):
        q = qa.get("question", "")
        a = qa.get("answer", "")
        lines.append(f"第 {idx} 題：")
        lines.append(f"面試官：{q}")
        lines.append(f"候選人：{a}")
        lines.append("")

    return "\n".join(lines)


def grade_interview(
    qas: List[Dict[str, str]],
    job_role: str,
    resume_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    針對整場面試進行評分，回傳 JSON 結果：

    {
      "overall": {
        "technical": 4,
        "communication": 3,
        "structure": 4,
        "relevance": 5,
        "summary": "..."
      },
      "per_question": [
        {
          "question": "...",
          "answer": "...",
          "score": {
            "technical": 3,
            "communication": 4,
            "structure": 3,
            "relevance": 4
          },
          "feedback": "..."
        },
        ...
      ]
    }
    """
    if not qas:
        return {}

    user_content = _build_grading_prompt(qas, job_role, resume_info)

    system_prompt = """
    你是一位專業的技術面試官與面試評分者，請你根據候選人的每一題回答進行嚴謹的評估。

    請依下列六個面向評分（1~5 分，5 分為最佳）：
    - technical: 技術正確性與深度
    - communication: 口語或文字表達清楚程度
    - structure: 答案結構與條理性（是否有步驟、是否有例子）
    - relevance: 回答與題目以及履歷內容的相關程度（是否答對題、是否「答非所問」）
    - problem_solving: 問題分析與解決能力
    - growth_potential: 學習潛力與發展性

    【答非所問判斷規則】
    若候選人的回答出現以下情況，relevance 給 1 或 2 分：
    - 明顯沒有回答題目的核心內容
    - 回答完全是無關的資訊
    - 提到的內容與題目毫無邏輯連結
    - 回答模糊、整段像自我介紹、但題目在問技術細節
    - 面試官問 A，候選人回答 B（經典「答非所問」）

    請在逐題回饋中明確指出這個問題，例如：
    -「此回答未能直接回應題目核心，屬於答非所問。」
    -「內容偏離題目方向，建議更聚焦。」

    【輸出格式很重要】
    請務必輸出以下 JSON 結構（不要加註解、不加多餘文字）：

    {
    "overall": {
        "technical": <1-5>,
        "communication": <1-5>,
        "structure": <1-5>,
        "relevance": <1-5>,
        "problem_solving": <1-5>,
        "growth_potential": <1-5>,
        "summary": "<整體總結>"
    },
    "per_question": [
        {
        "question": "<題目文字>",
        "answer": "<回答文字>",
        "score": {
            "technical": <1-5>,
            "communication": <1-5>,
            "structure": <1-5>,
            "relevance": <1-5>,
            "problem_solving": <1-5>,
            "growth_potential": <1-5>
        },
        "feedback": "<1~3 句具體回饋>"
        }
    ]
    }
    """.strip()



    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )

    result = completion.choices[0].message.content
    return json.loads(result)
