# resume_parser.py
import io
import json
from typing import Dict
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 讀 PDF 純文字
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


# ====== 使用 OpenAI 做語意解析（重點） ======
def llm_parse_resume_text(text: str) -> Dict:
    system_prompt = """
你是一位專業的履歷解析模型，請將輸入的履歷文本解析成結構化 JSON。
請用繁體中文輸出內容。

輸出的 JSON 結構如下：

{
  "skills": ["技能1", "技能2", ...],
  "projects": [
      {
          "title": "",
          "description": "",
          "tech_stack": []
      }
  ],
  "work_experience": [
      {
          "company": "",
          "position": "",
          "duration": "",
          "description": ""
      }
  ],
  "education": [
      {
          "school": "",
          "degree": "",
          "department": "",
          "duration": ""
      }
  ],
  "summary": "用 2-3 句話給使用者的履歷摘要"
}

請務必輸出有效 JSON，不要加註解、不需要額外說明。
    """

    user_prompt = f"以下為履歷文字內容，請依規則進行解析：\n\n{text}"

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # 解析履歷用這個最划算
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"}
    )

    result = completion.choices[0].message.content
    return json.loads(result)


# ====== 主函式：整合 PDF + LLM ======
def parse_resume(uploaded_file) -> Dict:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        raise ValueError("目前只支援 PDF 履歷")

    parsed = llm_parse_resume_text(text)

    # 存下原始文字以供 RAG 使用
    parsed["raw_text"] = text

    return parsed
