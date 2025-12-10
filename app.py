import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import io
import tempfile

# 自訂模組
from resume_parser import parse_resume
from grader import grade_interview, generate_suggestions
from pdf_export import export_pdf
from html_export import export_html
from voice_analysis import analyze_voice
from db import (
    init_db,
    save_candidate,
    save_interview,
    save_qa,
    save_scores,
    get_interviews,
    get_scores,
    get_qa
)

# ====== 初始化與設定 ======
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("請在 .env 設定 OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# 字型
# 注意：這個字型路徑在非 Windows 系統上會出錯，部署時需要更換
try:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception as e:
    print(f"無法設定 Matplotlib 字型: {e}")


# 初始化資料庫
init_db()

# ====== 語音功能 ======
def synthesize_speech(text: str) -> bytes:
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="opus",
        )
        return resp.read()
    except Exception as e:
        st.error(f"TTS 錯誤：{e}")
        return None

def speech_to_text(file) -> str:
    try:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
        )
        return resp.text
    except Exception as e:
        st.error(f"Whisper 錯誤：{e}")
        return ""

# ====== RAG ======
@st.cache_resource
def get_rag():
    class SimpleRAG:
        def __init__(self, folder="knowledge"):
            self.docs = []
            if not os.path.isdir(folder):
                return
            for fname in os.listdir(folder):
                if fname.endswith(".md"):
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        self.docs.append((fname, f.read()))

        def retrieve(self, job: str, query: str, top_k=3):
            if not self.docs:
                return []
            q = query.lower()
            scored = []
            for name, text in self.docs:
                score = sum(q.count(tok) for tok in q.split() if tok in text.lower())
                scored.append((score, name, text))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [x[2] for x in scored[:top_k] if x[0] > 0]
    return SimpleRAG("knowledge")

rag = get_rag()

# ====== Streamlit UI ======
st.set_page_config(page_title="AI 虛擬面試官", page_icon="🧑‍🏫")
st.title("🧑‍🏫 AI 虛擬面試官")
st.caption("履歷 + RAG + 語音 + 歷史紀錄 + 個人化建議")


# ====== Session State ======
def init_state(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

init_state("messages", [])
init_state("started", False)
init_state("resume_info", None)
init_state("candidate_id", "")
init_state("qa_list", [])
init_state("last_question", None)
init_state("grade_result", None)
init_state("selected_history_interview_id", None)
init_state("voice_mode", False)
init_state("voice_analysis_results", [])
init_state("suggestions", None)
init_state("last_processed_input", None)

# ====== Sidebar ======
with st.sidebar:
    st.header("面試設定")

    candidate_id = st.text_input(
        "受試者 ID（姓名 / 學號）",
        value=st.session_state.candidate_id,
        key="candidate_id_input"
    )
    st.session_state.candidate_id = candidate_id

    if candidate_id:
        save_candidate(candidate_id)

    job_role = st.selectbox(
        "應徵職缺",
        ["後端工程師", "AI 工程師", "資料工程師", "前端工程師"],
        key="job_role_input"
    )

    interview_style = st.selectbox(
        "面試風格", ["普通", "嚴格", "溫和"],
        key="interview_style_input"
    )

    st.markdown("---")
    st.subheader("上傳履歷（PDF）")
    uploaded_resume = st.file_uploader("PDF 履歷", type=["pdf"], key="resume_uploader")

    st.markdown("---")
    st.subheader("語音模式")
    st.session_state.voice_mode = st.checkbox("啟用語音（TTS + Whisper）", value=st.session_state.voice_mode, key="voice_mode_checkbox")

    st.markdown("---")
    st.subheader("歷史紀錄")
    if candidate_id:
        history_list = get_interviews(candidate_id)
        if history_list:
            labels = [f"{h['timestamp']}｜{h['job_role']}" for h in history_list]
            
            # Find the index of the selected interview
            default_idx = 0
            if st.session_state.selected_history_interview_id:
                for i, h in enumerate(history_list):
                    if h["interview_id"] == st.session_state.selected_history_interview_id:
                        default_idx = i
                        break
            
            picked_label = st.selectbox("選擇紀錄：", labels, index=default_idx, key="history_selectbox")
            # Find the id from the picked label
            for h in history_list:
                if f"{h['timestamp']}｜{h['job_role']}" == picked_label:
                    st.session_state.selected_history_interview_id = h["interview_id"]
                    break
        else:
            st.caption("此受試者目前沒有歷史資料")
    else:
        st.caption("請輸入受試者 ID 才能查詢歷史紀錄")

    st.markdown("---")
    reset_btn = st.button("🔁 重置本次面試")

# --------------------------------------------------------
# 重置狀態
# --------------------------------------------------------
if reset_btn:
    st.session_state.messages = []
    st.session_state.started = False
    st.session_state.resume_info = None
    st.session_state.qa_list = []
    st.session_state.last_question = None
    st.session_state.grade_result = None
    st.session_state.selected_history_interview_id = None
    st.session_state.voice_analysis_results = []
    st.session_state.suggestions = None
    st.session_state.last_processed_input = None
    st.rerun()

# --------------------------------------------------------
# 履歷解析（PDF → JSON）
# --------------------------------------------------------
if uploaded_resume and st.session_state.resume_info is None:
    with st.spinner("AI 正在解析履歷…"):
        st.session_state.resume_info = parse_resume(uploaded_resume)
    st.success("履歷解析完成！")

# 展開履歷摘要
with st.expander("📄 履歷解析結果"):
    ri = st.session_state.resume_info
    if ri:
        st.markdown("### 🧩 技能")
        st.write(", ".join(ri.get("skills", [])) or "（無）")
        st.markdown("### 📚 專案")
        for p in ri.get("projects", []):
            st.markdown(f"**{p['title']}** — {p['description']}")
            st.write("技術：", ", ".join(p.get("tech_stack", [])))
        st.markdown("### 💼 工作經驗")
        for w in ri.get("work_experience", []):
            st.markdown(f"**{w['company']} / {w['position']} ({w['duration']})**")
            st.write(w["description"])
        st.markdown("### 🎓 學歷")
        for e in ri.get("education", []):
            st.markdown(f"{e['school']} — {e['degree']} ({e['duration']})")
        st.markdown("### 📝 自我摘要")
        st.write(ri.get("summary", "（無）"))
    else:
        st.caption("尚未上傳履歷。 সন")

# --------------------------------------------------------
# Prompt 組合器
# --------------------------------------------------------
def build_system_prompt(job: str, style: str, resume_info=None, rag_snippets=None):
    style_map = {"普通": "語氣專業、正常面試流程。", "嚴格": "語氣直接、有壓力、深度追問。", "溫和": "語氣友善、講解式、鼓勵學生。"}
    style_desc = style_map[style]
    resume_context = ""
    if resume_info:
        skills = ", ".join(resume_info.get("skills", []))
        resume_context += f"候選人技能：{skills or '（無）'}\n"
        if resume_info.get("projects"):
            resume_context += "專案經驗：\n"
            for p in resume_info["projects"]:
                resume_context += f"- {p['title']}: {p['description']}\n"
        summary = resume_info.get("summary", "")
        if summary:
            resume_context += f"自我介紹摘要：{summary}\n"
    rag_context = ""
    if rag_snippets:
        rag_context += "\n以下為與職缺相關的重要技術知識片段（RAG）：\n"
        for i, snip in enumerate(rag_snippets, start=1):
            rag_context += f"[片段 {i}]\n{snip}\n\n"
    return f"""
你是一位專業「{job}」領域的面試官。
**你的最重要規則是：每次只問一個問題。** 你的回覆內容必須只能包含一個問題，不能包含任何其他文字或第二個問題。如果違反此規則，整個系統將會失敗。

請嚴格遵守以下所有原則：
1. **使用者要求結束：** 如果候選人明確表示想結束面試（例如說出「我不想回答了」、「結束面試」等），你必須立刻停止提問，並直接回傳 `[END_INTERVIEW]` 指令。
2. **自動結束面試**：當你判斷面試應結束時（核心問題問完、持續答非所問、或表現過差），也同樣回傳 `[END_INTERVIEW]` 指令。
3. 使用繁體中文。
4. 若候選人回答不完整，適度追問技術細節。
5. 風格：{style_desc}
6. 題目深度比一般面試更偏工程實作、技術理解。

根據以下候選人資訊與背景：
{resume_context}

{rag_context}

請開始面試，第一題請對方自我介紹。
""".strip()

# --------------------------------------------------------
# LLM 主回覆 function
# --------------------------------------------------------
def call_llm(job: str, style: str, history, resume_info=None):
    query_parts = [f"職缺：{job}"]

    last_q, last_a = None, None
    for role, msg in reversed(history):
        if role == "assistant" and last_q is None: last_q = msg
        elif role == "user" and last_a is None: last_a = msg
        if last_q and last_a: break

    if last_q: query_parts.append("上一題：" + last_q[:100])
    if last_a: query_parts.append("上一答：" + last_a[:100])

    if resume_info:
        skills = resume_info.get("skills", [])
        if skills: query_parts.append("技能：" + ", ".join(skills))

    rag_query = "；".join(query_parts)

    role_map = {"後端工程師": ["algorithms", "datastructures", "system_design", "database"], "AI 工程師": ["ai_ml", "algorithms", "computer_arch"], "資料工程師": ["database", "system_design"], "前端工程師": ["algorithms", "system_design"]}
    preferred_tags = role_map.get(job, [])

    raw_snips = rag.retrieve(job, rag_query, top_k=5)
    rag_snippets = sorted(raw_snips, key=lambda x: any(tag in x.lower() for tag in preferred_tags), reverse=True)[:3]

    system_prompt = build_system_prompt(
        job,
        style,
        resume_info=resume_info,
        rag_snippets=rag_snippets,
    )

    msgs = [{"role": "system", "content": system_prompt}] + [{"role": r, "content": c} for r, c in history]

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=msgs,
    )
    return resp.choices[0].message.content

# --------------------------------------------------------
# 評分與建議產生函式
# --------------------------------------------------------
def run_grading():
    if not st.session_state.qa_list:
        st.warning("尚未回答任何題目，無法評分。 সন")
        return
    with st.spinner("AI 正在分析你的整場面試…"):
        result = grade_interview(st.session_state.qa_list, job_role, st.session_state.resume_info)
        st.session_state.grade_result = result
        if st.session_state.candidate_id:
            interview_id = save_interview(candidate_id=st.session_state.candidate_id, job_role=job_role, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), summary=result["overall"].get("summary", ""))
            for qa in st.session_state.qa_list:
                save_qa(interview_id, qa["question"], qa["answer"])
            save_scores(interview_id, result["overall"])
    with st.spinner("AI 正在產生個人化建議…"):
        suggestions = generate_suggestions(st.session_state.qa_list, st.session_state.grade_result["overall"])
        st.session_state.suggestions = suggestions
    st.success("評分與建議皆已完成！請向下查看結果。 সন")

# --------------------------------------------------------
# 主流程
# --------------------------------------------------------
for role, content in st.session_state.messages:
    st.chat_message(role).markdown(content)

if not st.session_state.started:
    if st.button("▶️ 開始面試"):
        first_reply = call_llm(
            job_role,
            interview_style,
            [],
            resume_info=st.session_state.resume_info,
        )
        st.session_state.messages.append(("assistant", first_reply))
        st.session_state.last_question = first_reply
        st.session_state.started = True
        if st.session_state.voice_mode:
            audio_bytes = synthesize_speech(first_reply)
            if audio_bytes:
                st.session_state.tts_audio_bytes = audio_bytes
        st.rerun()
else:
    if "tts_audio_bytes" in st.session_state and st.session_state.tts_audio_bytes:
        st.audio(st.session_state.tts_audio_bytes, format="audio/opus")
        st.session_state.tts_audio_bytes = None

    if not st.session_state.grade_result:
        st.markdown("### 回答本題")
        audio_rec = st.audio_input("🎤 按下開始錄音 → 對著麥克風回答")
        voice_answer = None
        if audio_rec:
            with st.spinner("Whisper 正在辨識你的語音…"):
                voice_answer = speech_to_text(audio_rec)
            if voice_answer:
                st.success("語音辨識成功！ সন")
                st.write("你的語音內容：", voice_answer)
        
        audio_file = st.file_uploader("（可選）上傳語音檔 mp3/wav/m4a", type=["mp3","wav","m4a"])
        if audio_file and not voice_answer:
            with st.spinner("Whisper 正在辨識你的語音…"):
                voice_answer = speech_to_text(audio_file)
            if voice_answer:
                st.success("語音辨識成功！ সন")
                st.write("你的語音內容：", voice_answer)

        text_answer = st.chat_input("請輸入你的回答…")
        user_input = voice_answer if voice_answer else text_answer

        if user_input and user_input != st.session_state.last_processed_input:
            st.session_state.last_processed_input = user_input
            source_audio = audio_rec if audio_rec else audio_file
            if source_audio and user_input == voice_answer:
                with st.spinner("正在分析你的語音特徵…"):
                    analysis_result = analyze_voice(source_audio)
                    st.session_state.voice_analysis_results.append(analysis_result)
            else:
                st.session_state.voice_analysis_results.append(None)

            if st.session_state.last_question:
                st.session_state.qa_list.append({"question": st.session_state.last_question, "answer": user_input})
            
            st.session_state.messages.append(("user", user_input))
            
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("AI 面試官正在思考下一題..."):
                assistant_reply = call_llm(
                    job_role,
                    interview_style,
                    st.session_state.messages,
                    resume_info=st.session_state.resume_info,
                )

            if "[END_INTERVIEW]" in assistant_reply:
                st.info("好的，今天的面試差不多到此結束。我們將開始為您評分。 সন")
                run_grading()
                st.rerun()
            else:
                questions = [q.strip() for q in assistant_reply.split('\n') if q.strip()]
                first_question = questions[0] if questions else "抱歉，我好像沒有想到問題，可以請您再說一次嗎？"
                
                st.session_state.messages.append(("assistant", first_question))
                with st.chat_message("assistant"):
                    st.markdown(first_question)
                st.session_state.last_question = first_question

                if st.session_state.voice_mode:
                    tts_audio = synthesize_speech(first_question)
                    if tts_audio:
                        st.audio(tts_audio, format="audio/opus")
                # No rerun here to wait for next user input
        
        st.markdown("---")
        st.subheader("📊 面試評分")
        if st.button("📊 結束面試並進行 AI 評分"):
            run_grading()
            st.rerun()

if st.session_state.grade_result:
    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]
    tech, comm, struct, rel, ps, gp = (overall.get(k, 0) for k in ["technical", "communication", "structure", "relevance", "problem_solving", "growth_potential"])

    st.markdown("### ⭐ 整體評分")
    st.write(f"- 技術（technical）：**{tech} / 5**")
    st.write(f"- 表達（communication）：**{comm} / 5**")
    st.write(f"- 結構（structure）：**{struct} / 5**")
    st.write(f"- 相關性（relevance）：**{rel} / 5**")
    st.write(f"- 解題能力（problem_solving）：**{ps} / 5**")
    st.write(f"- 潛力（growth_potential）：**{gp} / 5**")
    st.markdown("#### 📝 整體評論")
    st.write(overall.get("summary", "N/A"))

    st.markdown("### 📌 本次面試雷達圖")
    categories = ["technical", "communication", "structure", "relevance", "problem_solving", "growth_potential"]
    labels_zh = ["技術", "表達", "結構", "相關", "解題", "潛力"]
    scores = [tech, comm, struct, rel, ps, gp]
    
    # 修正維度不匹配的 bug
    values = scores + scores[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_zh)
    ax.set_ylim(0, 5)
    plt.tight_layout()
    st.pyplot(fig)

    if st.session_state.selected_history_interview_id:
        ref_scores = get_scores(st.session_state.selected_history_interview_id)
        st.markdown("### 🔄 與歷史面試比較")
        if ref_scores:
            ref_vals = [ref_scores.get(k, 0) for k in categories]
            ref_plot = ref_vals + ref_vals[:1]
            fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax2.plot(angles, ref_plot, "r--", label="歷史")
            ax2.plot(angles, values, "b-", label="本次")
            ax2.fill(angles, values, alpha=0.25)
            ax2.set_thetagrids(angles[:-1] * 180 / np.pi, labels_zh)
            ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            ax2.set_ylim(0, 5)
            plt.tight_layout()
            st.pyplot(fig2)
            st.caption("虛線：歷史紀錄；實線：本次面試")

    st.markdown("### 📝 逐題回饋")
    for i, item in enumerate(per_question):
        s = item.get("score", {})
        st.markdown(f"#### 第 {i+1} 題")
        st.markdown(f"**題目：** {item.get('question', 'N/A')}")
        st.markdown(f"**回答：** {item.get('answer', 'N/A')}")
        st.write(f"技術 {s.get('technical',0)}/5 ｜ 表達 {s.get('communication',0)}/5 ｜ 結構 {s.get('structure',0)}/5 ｜ 相關 {s.get('relevance',0)}/5 ｜ 解題 {s.get('problem_solving',0)}/5 ｜ 潛力 {s.get('growth_potential',0)}/5")
        st.markdown(f"**回饋：** {item.get('feedback', 'N/A')}")
        st.markdown("---")

    if st.session_state.voice_analysis_results:
        st.markdown("### 🎤 逐題語音特徵回饋")
        for i, r in enumerate(st.session_state.voice_analysis_results):
            if r:
                st.markdown(f"#### 第 {i+1} 題的語音")
                if "error" in r:
                    st.warning(r["error"])
                else:
                    st.write(f"- **音調分析**：{r['pitch']}")
                    st.write(f"- **音量分析**：{r['volume']}")
                    st.write(f"- **語速分析**：{r['speech_rate']}")
                st.markdown("---")
    
    if st.session_state.suggestions:
        st.markdown("### 💡 個人化建議")
        st.markdown(st.session_state.suggestions)
        st.markdown("---")

    st.markdown("### 💾 下載面試報告")
    
    radar_image_buffer = io.BytesIO()
    fig.savefig(radar_image_buffer, format='PNG', dpi=300)
    radar_image_buffer.seek(0)

    report_data = {
        "candidate_id": st.session_state.candidate_id,
        "job_role": job_role,
        "timestamp": datetime.now().strftime("%Y-%m-%d"),
        "overall_scores": overall,
        "summary": overall.get("summary", "N/A"),
        "radar_chart_image": radar_image_buffer.read(),
        "qa_list": st.session_state.qa_list,
        "per_question_feedback": per_question,
        "voice_analysis_results": st.session_state.voice_analysis_results,
        "suggestions": st.session_state.suggestions,
    }

    pdf_bytes = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            export_pdf(tmp.name, report_data)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
    except Exception as e:
        st.error(f"產生 PDF 報告時發生錯誤: {e}")

    if pdf_bytes:
        st.download_button("📄 下載精緻 PDF 報告", data=pdf_bytes, file_name="interview_report_detailed.pdf", mime="application/pdf")

    def build_report_md():
        lines = ["# AI 面試官練習報告", f"- 受試者：{st.session_state.candidate_id}", f"- 職缺：{job_role}", f"- 日期：{datetime.now().strftime('%Y-%m-%d')}\n"]
        lines.append("## 整體評分")
        lines.append(f"- 技術：{tech}/5, 表達：{comm}/5, 結構：{struct}/5, 相關：{rel}/5, 解題：{ps}/5, 潛力：{gp}/5\n")
        lines.append("## 整體評論\n" + overall.get("summary", "N/A") + "\n")
        lines.append("## 逐題回饋")
        for i, item in enumerate(per_question):
            sc = item.get("score", {})
            lines.extend([f"### 第 {i+1} 題", f"- 題目：{item.get('question', 'N/A')}", f"- 回答：{item.get('answer', 'N/A')}", f"- 分數：技術 {sc.get('technical',0)}/5, 表達 {sc.get('communication',0)}/5, 結構 {sc.get('structure',0)}/5, 相關 {sc.get('relevance',0)}/5, 解題 {sc.get('problem_solving',0)}/5, 潛力 {sc.get('growth_potential',0)}/5", f"- 回饋：{item.get('feedback', 'N/A')}\n"])
        if st.session_state.suggestions:
            lines.extend(["## 個人化建議\n", st.session_state.suggestions])
        return "\n".join(lines)

    report_md = build_report_md()
    st.download_button("📄 下載 Markdown 報告", data=report_md, file_name="interview_report.md", mime="text/markdown")
    html_content = export_html(report_md)
    st.download_button("🌐 下載 HTML 報告", data=html_content, file_name="interview_report.html", mime="text/html")