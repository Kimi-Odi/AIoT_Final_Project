import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import librosa

# 自訂模組
from resume_parser import parse_resume
from grader import grade_interview
from pdf_export import export_pdf
from html_export import export_html
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
    raise RuntimeError("請在 .env 設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# 字型
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 初始化資料庫
init_db()

# ====== 語音功能 ======
def synthesize_speech(text: str) -> bytes:
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
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
            response_format="verbose_json"  # ⭐ 取得每段 timestamps
        )
        return resp.text
    except Exception as e:
        st.error(f"Whisper 錯誤：{e}")
        return ""

FILLERS = ["嗯", "呃", "那個", "就是", "你知道", "like", "you know", "um", "uh"]

def analyze_speech_features(whisper_resp, audio_bytes):
    result = {}

    # ======================
    # 1) 語速 WPM
    # ======================
    total_words = len(whisper_resp["text"].split())
    total_time = whisper_resp["segments"][-1]["end"] - whisper_resp["segments"][0]["start"]
    wpm = (total_words / total_time) * 60 if total_time > 0 else 0
    result["wpm"] = round(wpm, 2)

    # ======================
    # 2) 停頓比例（silence ratio）
    # ======================
    silences = []
    segs = whisper_resp["segments"]
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i-1]["end"]
        if gap > 0.2:   # >0.2s 視為停頓
            silences.append(gap)

    total_silence = sum(silences)
    result["silence_ratio"] = round(total_silence / total_time, 3)

    # ======================
    # 3) 音量穩定度（Volume Stability）
    # ======================
    # 讀取音訊為 numpy 陣列
    import soundfile as sf
    import io
    y, sr = sf.read(io.BytesIO(audio_bytes))

    frame = librosa.feature.rms(y=y)[0]  # Root Mean Square energy
    vol_std = np.std(frame)
    vol_mean = np.mean(frame)
    stability = 1 - (vol_std / (vol_mean + 1e-9))
    result["volume_stability"] = round(float(stability), 3)

    # ======================
    # 4) 填充詞比例 filler ratio
    # ======================
    filler_count = 0
    for f in FILLERS:
        filler_count += whisper_resp["text"].count(f)

    filler_ratio = filler_count / max(total_words, 1)
    result["filler_ratio"] = round(filler_ratio, 3)

    return result


# ====== RAG ======
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

@st.cache_resource
def get_rag():
    return SimpleRAG("knowledge")

rag = get_rag()

# ====== Streamlit UI ======
st.set_page_config(page_title="AI 虛擬面試官", page_icon="🧑‍🏫")
st.title("🧑‍🏫 AI 虛擬面試官（履歷 + RAG + 語音 + 歷史紀錄）")

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
init_state("play_tts_first_question", False)


# ====== Sidebar ======
with st.sidebar:
    st.header("面試設定")

    candidate_id = st.text_input(
        "受試者 ID（姓名 / 學號）",
        value=st.session_state.candidate_id,
    )
    st.session_state.candidate_id = candidate_id

    if candidate_id:
        save_candidate(candidate_id)

    job_role = st.selectbox(
        "應徵職缺",
        ["後端工程師", "AI 工程師", "資料工程師", "前端工程師"],
    )

    interview_style = st.selectbox(
        "面試風格", ["普通", "嚴格", "溫和"]
    )

    st.markdown("---")
    st.subheader("上傳履歷（PDF）")
    uploaded_resume = st.file_uploader("PDF 履歷", type=["pdf"])

    st.markdown("---")
    st.subheader("語音模式")
    st.session_state.voice_mode = st.checkbox("啟用語音（TTS + Whisper）", value=False)

    st.markdown("---")
    st.subheader("歷史紀錄")
    history_list = []
    if candidate_id:
        history_list = get_interviews(candidate_id)
        if history_list:
            labels = [f"{h['timestamp']}｜{h['job_role']}" for h in history_list]
            idx = st.session_state.selected_history_interview_id
            default_idx = 0
            if idx:
                for i, h in enumerate(history_list):
                    if h["interview_id"] == idx:
                        default_idx = i
                        break

            picked = st.selectbox("選擇紀錄：", labels, index=default_idx)
            st.session_state.selected_history_interview_id = history_list[
                labels.index(picked)
            ]["interview_id"]
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
        st.caption("尚未上傳履歷。")


# --------------------------------------------------------
# Prompt 組合器（含電資 RAG）
# --------------------------------------------------------
def build_system_prompt(job: str, style: str, resume_info=None, rag_snippets=None):

    style_map = {
        "普通": "語氣專業、正常面試流程。",
        "嚴格": "語氣直接、有壓力、深度追問。",
        "溫和": "語氣友善、講解式、鼓勵學生。",
    }
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

請遵守以下原則：
1. 使用繁體中文。
2. 每次只問一題。
3. 若候選人回答不完整，適度追問技術細節。
4. 風格：{style_desc}
5. 題目深度比一般面試更偏工程實作、技術理解。

根據以下候選人資訊與背景：
{resume_context}

{rag_context}

請開始面試，第一題請對方自我介紹。
""".strip()


# --------------------------------------------------------
# LLM 主回覆 function（含 RAG + 上一輪 Q/A）
# --------------------------------------------------------
def call_llm(job: str, style: str, history, resume_info=None):

    # ===== 產生 RAG 查詢 =====
    query_parts = [f"職缺：{job}"]

    last_q, last_a = None, None
    for role, msg in reversed(history):
        if role == "assistant" and last_q is None:
            last_q = msg
        elif role == "user" and last_a is None:
            last_a = msg
        if last_q and last_a:
            break

    if last_q:
        query_parts.append("上一題：" + last_q[:100])
    if last_a:
        query_parts.append("上一答：" + last_a[:100])

    if resume_info:
        skills = resume_info.get("skills", [])
        if skills:
            query_parts.append("技能：" + ", ".join(skills))

    rag_query = "；".join(query_parts)

    # ===== 電資職缺 RAG 權重 =====
    role_map = {
        "後端工程師": ["algorithms", "datastructures", "system_design", "database"],
        "AI 工程師": ["ai_ml", "algorithms", "computer_arch"],
        "資料工程師": ["database", "system_design"],
        "前端工程師": ["algorithms", "system_design"],
    }
    preferred_tags = role_map.get(job, [])

    raw_snips = rag.retrieve(job, rag_query, top_k=5)
    rag_snippets = sorted(
        raw_snips,
        key=lambda x: any(tag in x.lower() for tag in preferred_tags),
        reverse=True
    )[:3]

    # ===== Build system prompt =====
    system_prompt = build_system_prompt(
        job,
        style,
        resume_info=resume_info,
        rag_snippets=rag_snippets,
    )

    # ===== 呼叫 OpenAI =====
    msgs = [{"role": "system", "content": system_prompt}]
    for r, c in history:
        msgs.append({"role": r, "content": c})

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=msgs,
    )
    return resp.choices[0].message.content


# --------------------------------------------------------
# 顯示歷史訊息（聊天框）
# --------------------------------------------------------
for role, content in st.session_state.messages:
    st.chat_message(role).markdown(content)
    # ===== 第一題 TTS 播放 =====
    if st.session_state.get("play_tts_first_question", False):
        st.session_state.play_tts_first_question = False  # 播一次就關掉
        first_question = st.session_state.last_question
        audio_bytes = synthesize_speech(first_question)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")



# --------------------------------------------------------
# 面試主流程（尚未開始）
# --------------------------------------------------------
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

        # ⭐設定旗標，下一輪 render 播放 TTS
        if st.session_state.voice_mode:
            st.session_state.play_tts_first_question = True

        st.rerun()




# --------------------------------------------------------
# 面試已開始 → 使用者回答
# --------------------------------------------------------
else:
    st.markdown("### 回答本題")

    # ===== 使用者語音回答（錄音 + Whisper） =====
    st.markdown("### 🎤 語音回答（錄音）")

    audio_rec = st.audio_input("按下開始錄音 → 對著麥克風回答")

    voice_answer = None

    if audio_rec:
        with st.spinner("Whisper 正在辨識你的語音…"):
            whisper_resp = speech_to_text(audio_rec)
            voice_answer = whisper_resp["text"]

            # ===== 語音特徵分析 =====
            analysis = analyze_speech_features(whisper_resp, audio_rec.getvalue())

            st.markdown("### 📊 語音特徵分析")
            st.write(f"- 語速（WPM）：{analysis['wpm']}")
            st.write(f"- 停頓比例：{analysis['silence_ratio']}")
            st.write(f"- 音量穩定度：{analysis['volume_stability']}")
            st.write(f"- 填充詞比例：{analysis['filler_ratio']}")

        if voice_answer:
            st.success("語音辨識成功！")
            st.write("你的語音內容：", voice_answer)

    # ===== 舊的上傳檔案功能（仍保留） =====
    audio_file = st.file_uploader("（可選）上傳語音檔 mp3/wav/m4a", type=["mp3","wav","m4a"])
    if audio_file and not voice_answer:
        with st.spinner("Whisper 正在辨識你的語音…"):
            voice_answer = speech_to_text(audio_file)
        if voice_answer:
            st.success("語音辨識成功！")
            st.write("你的語音內容：", voice_answer)


    # ====== 文字回答 ======
    text_answer = st.chat_input("請輸入你的回答…")

    # 語音優先於文字
    user_input = voice_answer if voice_answer else text_answer

    # 若沒有回答（語音/文字）則不進行
    if user_input:
        # 記錄 QA（上一題 + 使用者的回答）
        if st.session_state.last_question:
            st.session_state.qa_list.append({
                "question": st.session_state.last_question,
                "answer": user_input,
            })

        # 顯示使用者回答
        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        # 呼叫面試官
        assistant_reply = call_llm(
            job_role,
            interview_style,
            st.session_state.messages,
            resume_info=st.session_state.resume_info,
        )

        # 顯示 AI 回覆
        st.session_state.messages.append(("assistant", assistant_reply))
        st.chat_message("assistant").markdown(assistant_reply)

        # 更新 last_question
        st.session_state.last_question = assistant_reply

        # ===== 面試官語音出題（TTS） =====
        if st.session_state.voice_mode:
            tts_audio = synthesize_speech(assistant_reply)
            if tts_audio:
                st.audio(tts_audio, format="audio/mp3")


# --------------------------------------------------------
# 評分按鈕
# --------------------------------------------------------
st.markdown("---")
st.subheader("📊 面試評分")

if st.button("📊 結束面試並進行 AI 評分"):
    if not st.session_state.qa_list:
        st.warning("尚未回答任何題目，無法評分。")
    else:
        with st.spinner("AI 正在分析你的整場面試…"):
            # 產生評分
            result = grade_interview(
                st.session_state.qa_list,
                job_role,
                st.session_state.resume_info,
            )
            st.session_state.grade_result = result

            # 儲存到資料庫 interview.db
            if st.session_state.candidate_id:
                interview_id = save_interview(
                    candidate_id=st.session_state.candidate_id,
                    job_role=job_role,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    summary=result["overall"]["summary"],
                )

                # 儲存 QA
                for qa in st.session_state.qa_list:
                    save_qa(interview_id, qa["question"], qa["answer"])

                # 儲存分數
                save_scores(interview_id, result["overall"])

        st.success("評分完成！請向下查看結果。")


# --------------------------------------------------------
# 顯示評分結果
# --------------------------------------------------------
if st.session_state.grade_result:

    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]

    # 主分數
    tech = overall["technical"]
    comm = overall["communication"]
    struct = overall["structure"]
    rel = overall["relevance"]
    ps = overall["problem_solving"]
    gp = overall["growth_potential"]

    st.markdown("### ⭐ 整體評分")

    st.write(f"- 技術（technical）：**{tech} / 5**")
    st.write(f"- 表達（communication）：**{comm} / 5**")
    st.write(f"- 結構（structure）：**{struct} / 5**")
    st.write(f"- 相關性（relevance）：**{rel} / 5**")
    st.write(f"- 解題能力（problem_solving）：**{ps} / 5**")
    st.write(f"- 潛力（growth_potential）：**{gp} / 5**")

    st.markdown("#### 📝 整體評論")
    st.write(overall["summary"])

    # --------------------------------------------------------
    # 📈 本次面試雷達圖
    # --------------------------------------------------------
    st.markdown("### 📌 本次面試雷達圖")

    categories = ["technical", "communication", "structure",
                  "relevance", "problem_solving", "growth_potential"]
    labels_zh = ["技術", "表達", "結構", "相關", "解題", "潛力"]

    scores = [tech, comm, struct, rel, ps, gp]
    values = scores + scores[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories) + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels_zh)
    ax.set_ylim(0, 5)
    plt.tight_layout()
    st.pyplot(fig)

    # --------------------------------------------------------
    # 🔄 與歷史紀錄比較雷達圖（若有選取）
    # --------------------------------------------------------
    if st.session_state.selected_history_interview_id:
        ref_scores = get_scores(st.session_state.selected_history_interview_id)

        st.markdown("### 🔄 與歷史面試比較")

        if ref_scores:
            ref_vals = [
                ref_scores["technical"],
                ref_scores["communication"],
                ref_scores["structure"],
                ref_scores["relevance"],
                ref_scores["problem_solving"],
                ref_scores["growth_potential"],
            ]
            cur_vals = scores
            ref_plot = ref_vals + ref_vals[:1]
            cur_plot = cur_vals + cur_vals[:1]

            fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax2.plot(angles, ref_plot, "r--", label="歷史")
            ax2.plot(angles, cur_plot, "b-", label="本次")
            ax2.fill(angles, cur_plot, alpha=0.25)
            ax2.set_thetagrids(angles[:-1] * 180 / np.pi, labels_zh)
            ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            ax2.set_ylim(0, 5)

            plt.tight_layout()
            st.pyplot(fig2)
            st.caption("虛線：歷史紀錄；實線：本次面試")

    # --------------------------------------------------------
    # 逐題回饋
    # --------------------------------------------------------
    st.markdown("### 📝 逐題回饋")

    for i, item in enumerate(per_question, start=1):
        s = item["score"]
        st.markdown(f"#### 第 {i} 題")
        st.markdown(f"**題目：** {item['question']}")
        st.markdown(f"**回答：** {item['answer']}")
        st.write(
            f"技術 {s['technical']}/5 ｜ 表達 {s['communication']}/5 ｜ "
            f"結構 {s['structure']}/5 ｜ 相關 {s['relevance']}/5 ｜ "
            f"解題 {s['problem_solving']}/5 ｜ 潛力 {s['growth_potential']}/5"
        )
        st.markdown(f"**回饋：** {item['feedback']}")
        st.markdown("---")


    # --------------------------------------------------------
    # 面試報告下載（MD / PDF / HTML）
    # --------------------------------------------------------
    st.markdown("### 💾 下載面試報告")

    def build_report_md():
        lines = []
        lines.append("# AI 面試官練習報告\n")
        lines.append(f"- 受試者：{st.session_state.candidate_id}")
        lines.append(f"- 職缺：{job_role}")
        lines.append(f"- 日期：{datetime.now().strftime('%Y-%m-%d')}\n")

        lines.append("## 整體評分")
        lines.append(f"- 技術：{tech}/5")
        lines.append(f"- 表達：{comm}/5")
        lines.append(f"- 結構：{struct}/5")
        lines.append(f"- 相關：{rel}/5")
        lines.append(f"- 解題：{ps}/5")
        lines.append(f"- 潛力：{gp}/5\n")

        lines.append("## 整體評論\n" + overall["summary"] + "\n")

        lines.append("## 逐題回饋")
        for i, item in enumerate(per_question, start=1):
            sc = item["score"]
            lines.append(f"### 第 {i} 題")
            lines.append(f"- 題目：{item['question']}")
            lines.append(f"- 回答：{item['answer']}")
            lines.append(
                f"- 分數：技術 {sc['technical']}/5，表達 {sc['communication']}/5，"
                f"結構 {sc['structure']}/5，相關 {sc['relevance']}/5，"
                f"解題 {sc['problem_solving']}/5，潛力 {sc['growth_potential']}/5"
            )
            lines.append(f"- 回饋：{item['feedback']}\n")
        return "\n".join(lines)

    report_md = build_report_md()

    # Markdown 下載
    st.download_button(
        "📄 下載 Markdown 報告",
        data=report_md,
        file_name="interview_report.md",
        mime="text/markdown",
    )

    # PDF 下載
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        export_pdf(tmp.name, report_md)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

    st.download_button(
        "📄 下載 PDF 報告",
        data=pdf_bytes,
        file_name="interview_report.pdf",
        mime="application/pdf",
    )

    # HTML 下載
    html_content = export_html(report_md)
    st.download_button(
        "🌐 下載 HTML 報告",
        data=html_content,
        file_name="interview_report.html",
        mime="text/html",
    )
