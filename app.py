# ============================================================
# PART 1 — Imports、初始化、資料庫、語音（Whisper/TTS）、RAG
# ============================================================

import os
import json
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import librosa
import soundfile as sf

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
    get_qa,
)

# ====== 初始化 ======
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("請在 .env 中設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ====== 字型設定 ======
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ====== 初始化資料庫 ======
init_db()

# ============================================================
# ------------- 語音功能（Whisper + TTS） ---------------------
# ============================================================

def speech_to_text(file):
    """
    Whisper 語音辨識（回傳 Python dict，需要 verbose_json）
    """
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        response_format="verbose_json"
    )
    return resp.model_dump()   # ⭐ 回傳 dict（不是 Transcription 物件）


def synthesize_speech(text: str) -> bytes:
    """
    TTS — 文字轉語音
    """
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


# ============================================================
# ----------- 語音特徵分析：WPM / Silence / Volume / Fillers ----
# ============================================================

FILLERS = ["嗯", "呃", "那個", "就是", "like", "you know"]

def analyze_speech_features(whisper_resp, audio_bytes):
    """
    回傳 dict：
    {
      wpm,
      silence_ratio,
      volume_stability,
      filler_ratio
    }
    """

    result = {}

    # -------------------------
    # 1) 語速（WPM）
    # -------------------------
    total_words = len(whisper_resp["text"].split())
    segs = whisper_resp["segments"]
    total_time = segs[-1]["end"] - segs[0]["start"]
    wpm = (total_words / total_time) * 60 if total_time > 0 else 0
    result["wpm"] = round(wpm, 2)

    # -------------------------
    # 2) 停頓比例
    # -------------------------
    silences = []
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i-1]["end"]
        if gap > 0.25:
            silences.append(gap)

    total_silence = sum(silences)
    result["silence_ratio"] = round(total_silence / total_time, 3)

    # -------------------------
    # 3) 音量穩定度（Volume Stability）
    # -------------------------
    y, sr = sf.read(io.BytesIO(audio_bytes))
    frame_energy = librosa.feature.rms(y=y)[0]

    vol_mean = np.mean(frame_energy)
    vol_std = np.std(frame_energy)

    stability = 1 - (vol_std / (vol_mean + 1e-9))
    result["volume_stability"] = round(float(stability), 3)

    # -------------------------
    # 4) 填充詞比例
    # -------------------------
    filler_count = sum(whisper_resp["text"].count(f) for f in FILLERS)
    filler_ratio = filler_count / max(total_words, 1)
    result["filler_ratio"] = round(filler_ratio, 3)

    return result


# ============================================================
# ------------- RAG 知識庫載入（電資學生專用） ---------------
# ============================================================

class SimpleRAG:
    def __init__(self, folder="knowledge"):
        self.docs = []
        if not os.path.isdir(folder):
            return
        for fname in os.listdir(folder):
            if fname.endswith(".md"):
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    self.docs.append((fname, f.read()))

    def retrieve(self, job, query, top_k=3):
        if not self.docs:
            return []
        q = query.lower()
        scored = []
        for name, text in self.docs:
            score = sum(q.count(tok) for tok in q.split() if tok in text.lower())
            scored.append((score, text))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [x[1] for x in scored[:top_k] if x[0] > 0]


@st.cache_resource
def load_rag():
    return SimpleRAG("knowledge")


rag = load_rag()

# ============================================================
# -------------------- UI & Session 初始化 -------------------
# ============================================================

st.set_page_config(page_title="AI 虛擬面試官", page_icon="🧑‍🏫")
st.title("🧑‍🏫 AI 電資領域虛擬面試官（語音 + RAG + 履歷 + 評分）")

def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

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
init_state("last_speech_features", None)

# ============================================================
# PART 2 — 履歷解析、Prompt 生成、RAG、LLM 回覆
# ============================================================

# ------------------------------------------------------------
# Sidebar 設置
# ------------------------------------------------------------
with st.sidebar:
    st.header("面試設定")

    # 受試者 ID
    candidate_id = st.text_input("受試者 ID（姓名 / 學號）", value=st.session_state.candidate_id)
    st.session_state.candidate_id = candidate_id

    if candidate_id:
        save_candidate(candidate_id)

    job_role = st.selectbox(
        "應徵職缺",
        ["後端工程師", "AI 工程師", "資料工程師", "前端工程師"]
    )

    interview_style = st.selectbox(
        "面試風格",
        ["普通", "嚴格", "溫和"]
    )

    st.markdown("---")
    st.subheader("履歷上傳（PDF）")
    uploaded_resume = st.file_uploader("選擇 PDF 履歷", type=["pdf"])

    st.markdown("---")
    st.subheader("語音模式（TTS + Whisper）")
    st.session_state.voice_mode = st.checkbox("啟用語音模式", value=False)

    st.markdown("---")
    st.subheader("歷史紀錄")

    history = []
    if candidate_id:
        history = get_interviews(candidate_id)

    if history:
        options = [
            f"{h['timestamp']}｜{h['job_role']}｜ID:{h['interview_id']}"
            for h in history
        ]
        picked = st.selectbox("選擇一筆歷史紀錄：", options)
        idx = options.index(picked)
        st.session_state.selected_history_interview_id = history[idx]["interview_id"]
    else:
        st.caption("尚無歷史紀錄")

    st.markdown("---")
    if st.button("🔁 重置面試"):
        for key in [
            "messages", "started", "resume_info", "qa_list",
            "last_question", "grade_result", "last_speech_features"
        ]:
            st.session_state[key] = None if key == "resume_info" else []
        st.session_state.started = False
        st.rerun()


# ------------------------------------------------------------
# 履歷解析（PDF → JSON）
# ------------------------------------------------------------
if uploaded_resume and st.session_state.resume_info is None:
    with st.spinner("AI 正在解析你的履歷…"):
        st.session_state.resume_info = parse_resume(uploaded_resume)
    st.success("履歷解析完成！")

# 展示履歷解析內容
with st.expander("📄 履歷解析結果"):
    ri = st.session_state.resume_info
    if ri:
        st.markdown("### 🧩 技能")
        st.write(", ".join(ri.get("skills", [])) or "（無）")

        st.markdown("### 📚 專案")
        for p in ri.get("projects", []):
            st.markdown(f"**{p['title']}** — {p['description']}")
            st.caption("技術：" + ", ".join(p.get("tech_stack", [])))

        st.markdown("### 💼 工作經驗")
        for w in ri.get("work_experience", []):
            st.markdown(f"**{w['company']} / {w['position']} ({w['duration']})**")
            st.write(w["description"])

        st.markdown("### 🎓 學歷")
        for e in ri.get("education", []):
            st.markdown(f"- {e['school']} — {e['degree']} ({e['duration']})")

        st.markdown("### 📝 自我摘要")
        st.write(ri.get("summary", "（無）"))
    else:
        st.caption("尚未上傳履歷。")


# ------------------------------------------------------------
# Prompt 建構器（含 RAG）
# ------------------------------------------------------------
def build_system_prompt(job, style, resume_info=None, rag_snippets=None):

    style_desc = {
        "普通": "語氣專業，提問自然。",
        "嚴格": "語氣直接、追問細節、有壓力感。",
        "溫和": "語氣親切、鼓勵式提問。",
    }[style]

    # ===== 履歷內容 =====
    resume_context = ""
    if resume_info:
        skills = resume_info.get("skills", [])
        resume_context += f"候選人技能：{', '.join(skills)}\n" if skills else ""

        if resume_info.get("projects"):
            resume_context += "專案：\n"
            for p in resume_info["projects"]:
                resume_context += f"- {p['title']}: {p['description']}\n"

    # ===== RAG =====
    rag_context = ""
    if rag_snippets:
        rag_context += "\n以下為職缺相關的技術知識片段（RAG）：\n"
        for i, sn in enumerate(rag_snippets, 1):
            rag_context += f"[{i}] {sn}\n"

    return f"""
你是一位專業的 **{job}** 面試官。

面試風格：{style_desc}

請遵守規則：
1. 每次只問一題。
2. 問題需有技術深度，聚焦職缺能力。
3. 若候選人答不完整，追問更細。
4. 用繁體中文。

候選人資訊：
{resume_context}

技術知識（RAG）：
{rag_context}

開始面試，請提出第一題：自我介紹。
""".strip()


# ------------------------------------------------------------
# LLM 回覆（含 RAG 查詢）
# ------------------------------------------------------------
def call_llm(job, style, history, resume_info=None):

    # ---- RAG 查詢字串 ----
    query_parts = [f"職缺：{job}"]

    last_q = None
    last_a = None

    for role, msg in reversed(history):
        if role == "assistant" and last_q is None:
            last_q = msg
        elif role == "user" and last_a is None:
            last_a = msg
        if last_q and last_a:
            break

    if last_q: query_parts.append("上一題：" + last_q[:80])
    if last_a: query_parts.append("上一答：" + last_a[:80])

    if resume_info and resume_info.get("skills"):
        query_parts.append("技能：" + ", ".join(resume_info["skills"]))

    rag_query = "；".join(query_parts)

    # ---- 根據職缺自動排序 RAG ----
    role_pref = {
        "後端工程師": ["algorithms", "datastructures", "system_design", "database"],
        "AI 工程師": ["ai_ml", "algorithms", "computer_arch"],
        "資料工程師": ["database", "system_design"],
        "前端工程師": ["algorithms", "system_design"],
    }.get(job, [])

    raw_snippets = rag.retrieve(job, rag_query, top_k=5)
    rag_snippets = sorted(
        raw_snippets,
        key=lambda x: any(tag in x.lower() for tag in role_pref),
        reverse=True
    )[:3]

    # ---- System prompt ----
    system_prompt = build_system_prompt(
        job,
        style,
        resume_info=resume_info,
        rag_snippets=rag_snippets
    )

    # ---- Messages ----
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})

    # ---- 呼叫 OpenAI ----
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return resp.choices[0].message.content


# ============================================================
# PART 3 — 面試流程（開始面試 + 語音回答 + TTS + Whisper）
# ============================================================

# ------------------------------------------------------------
# 顯示歷史對話訊息
# ------------------------------------------------------------
for role, content in st.session_state.messages:
    st.chat_message(role).markdown(content)


# ------------------------------------------------------------
# 尚未開始面試
# ------------------------------------------------------------
if not st.session_state.started:

    if st.button("▶️ 開始面試"):

        # 生成第一題（通常是自我介紹）
        first_reply = call_llm(
            job_role,
            interview_style,
            [],
            resume_info=st.session_state.resume_info
        )

        st.session_state.messages.append(("assistant", first_reply))
        st.session_state.last_question = first_reply
        st.session_state.started = True

        # ⭐ 關鍵：第一題 TTS 必須延後一輪播放
        if st.session_state.voice_mode:
            st.session_state.play_tts_first_question = True

        st.rerun()


# ------------------------------------------------------------
# 第一題 TTS 播放（避免被 rerun 吃掉）
# ------------------------------------------------------------
if st.session_state.get("play_tts_first_question", False):
    st.session_state.play_tts_first_question = False   # 播一次就關掉

    text = st.session_state.last_question
    audio_bytes = synthesize_speech(text)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")


# ------------------------------------------------------------
# 面試已經開始 → 使用者回答（語音 / 文字）
# ------------------------------------------------------------
if st.session_state.started:

    st.markdown("### 🧑‍💬 請回答：")

    # ============================================================
    # 🎤（方式 1）使用者錄音回答（Streamlit 錄音按鈕）
    # ============================================================
    st.markdown("#### 🎤 語音回答（錄音）")

    audio_rec = st.audio_input("點擊錄音 → 說出你的答案")

    voice_answer = None

    if audio_rec:
        with st.spinner("Whisper 正在辨識語音…"):
            whisper_resp = speech_to_text(audio_rec)

        voice_answer = whisper_resp["text"]

        # ===== 語音特徵分析 =====
        speech_features = analyze_speech_features(whisper_resp, audio_rec.getvalue())
        st.session_state.last_speech_features = speech_features

        st.success("語音辨識完成！")

        st.markdown("### 🎧 語音特徵分析")
        st.write(f"- 語速（WPM）：{speech_features['wpm']}")
        st.write(f"- 停頓比例：{speech_features['silence_ratio']}")
        st.write(f"- 音量穩定度：{speech_features['volume_stability']}")
        st.write(f"- 填充詞比例：{speech_features['filler_ratio']}")

        st.markdown("---")

    # ============================================================
    # 🎤（方式 2）使用者上傳語音檔（備用）
    # ============================================================
    st.markdown("#### 📁 語音檔上傳回答（mp3 / wav / m4a）")
    audio_file = st.file_uploader("上傳語音檔案", type=["mp3", "wav", "m4a"])

    if audio_file and voice_answer is None:
        with st.spinner("Whisper 正在辨識語音…"):
            whisper_resp = speech_to_text(audio_file)

        voice_answer = whisper_resp["text"]

        speech_features = analyze_speech_features(whisper_resp, audio_file.read())
        st.session_state.last_speech_features = speech_features

        st.success("語音辨識成功！")


    # ============================================================
    # 📝（方式 3）文字回答
    # ============================================================
    st.markdown("#### ⌨️ 文字回答")
    text_answer = st.chat_input("請輸入你的回答…")

    # 語音優先於文字
    user_input = voice_answer if voice_answer else text_answer

    if user_input:

        # --------- 記錄上一題+使用者回答（QA） -----------
        st.session_state.qa_list.append({
            "question": st.session_state.last_question,
            "answer": user_input
        })

        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        # --------- 呼叫面試官取得下一題 ----------
        assistant_reply = call_llm(
            job_role,
            interview_style,
            st.session_state.messages,
            resume_info=st.session_state.resume_info,
        )

        st.session_state.messages.append(("assistant", assistant_reply))
        st.chat_message("assistant").markdown(assistant_reply)
        st.session_state.last_question = assistant_reply

        # --------- TTS 播放下一題 ----------
        if st.session_state.voice_mode:
            tts_audio = synthesize_speech(assistant_reply)
            if tts_audio:
                st.audio(tts_audio, format="audio/mp3")

# ============================================================
# PART 4 — AI 面試評分（含語音特徵 + 語音建議）
# ============================================================

# ------------------------------------------------------------
# 評分按鈕
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 面試評分（AI 分析）")

if st.button("📊 結束面試並進行 AI 評分"):

    if not st.session_state.qa_list:
        st.warning("你尚未回答任何題目，無法進行評分。")
    else:
        with st.spinner("AI 正在分析你的整場面試……"):

            # ⭐ 傳入語音特徵讓 grader 加權
            result = grade_interview(
                st.session_state.qa_list,
                job_role,
                st.session_state.resume_info,
                speech_features=st.session_state.last_speech_features
            )

            st.session_state.grade_result = result

            # ----------- 儲存到資料庫 -----------
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

        st.success("評分完成！向下捲動查看分析結果。")


# ------------------------------------------------------------
# 顯示評分結果
# ------------------------------------------------------------
if st.session_state.grade_result:

    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]

    # 拆分六大項目
    tech = overall["technical"]
    comm = overall["communication"]
    struct = overall["structure"]
    rel = overall["relevance"]
    ps = overall["problem_solving"]
    gp = overall["growth_potential"]

    st.markdown("## ⭐ 整體評分")

    st.write(f"- 技術能力（Technical）：**{tech}/5**")
    st.write(f"- 表達能力（Communication）：**{comm}/5**")
    st.write(f"- 回答結構（Structure）：**{struct}/5**")
    st.write(f"- 相關性（Relevance）：**{rel}/5**")
    st.write(f"- 解題能力（Problem Solving）：**{ps}/5**")
    st.write(f"- 成長潛力（Growth Potential）：**{gp}/5**")

    st.markdown("### 📝 整體評論")
    st.write(overall["summary"])


    # ============================================================
    # 🎤 語音特徵區段（若有語音回答）
    # ============================================================
    st.markdown("## 🎤 語音表達能力分析")

    features = st.session_state.last_speech_features

    if features:
        st.write(f"- 語速（WPM）：**{features['wpm']}**")
        st.write(f"- 停頓比例：**{features['silence_ratio']}**")
        st.write(f"- 音量穩定度：**{features['volume_stability']}**")
        st.write(f"- 填充詞比例：**{features['filler_ratio']}**")
    else:
        st.caption("（本次沒有語音回答，因此無法進行語音分析。）")


    # ============================================================
    # 🎤 AI 語音表達改善建議（D）
    # ============================================================
    from grader import generate_speech_feedback

    st.markdown("## 🎧 語音改善建議（AI 生成）")

    speech_fb = generate_speech_feedback(features)
    st.write(speech_fb)


# ============================================================
# PART 5 — 雷達圖 + 歷史比較 + 逐題回饋
# ============================================================

if st.session_state.grade_result:

    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]

    # 各項分數
    tech = overall["technical"]
    comm = overall["communication"]
    struct = overall["structure"]
    rel = overall["relevance"]
    ps = overall["problem_solving"]
    gp = overall["growth_potential"]

    # ============================================================
    # 📈 雷達圖（本次面試）
    # ============================================================
    st.markdown("## 📊 本次面試雷達圖")

    categories = ["technical", "communication", "structure",
                  "relevance", "problem_solving", "growth_potential"]
    labels_zh = ["技術", "表達", "結構", "相關性", "解題", "潛力"]

    scores = [tech, comm, struct, rel, ps, gp]
    values = scores + scores[:1]
    angles = np.linspace(0, 2*np.pi, len(categories) + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels_zh)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    st.pyplot(fig)

    # ============================================================
    # 🔄 歷史比較雷達圖
    # ============================================================
    if st.session_state.selected_history_interview_id:

        st.markdown("## 🔄 與歷史面試比較")

        ref_scores = get_scores(st.session_state.selected_history_interview_id)

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
            ax2.plot(angles, ref_plot, "r--", linewidth=1.8, label="歷史紀錄")
            ax2.plot(angles, cur_plot, "b-", linewidth=2.2, label="本次面試")
            ax2.fill(angles, cur_plot, alpha=0.25)

            ax2.set_thetagrids(angles[:-1] * 180/np.pi, labels_zh)
            ax2.set_ylim(0, 5)
            ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
            plt.tight_layout()

            st.pyplot(fig2)
            st.caption("提示：虛線代表歷史紀錄，實線代表本次面試。")

    # ============================================================
    # 📝 逐題回饋（Question-by-Question）
    # ============================================================
    st.markdown("## 📝 逐題回饋（AI 分析）")

    for i, item in enumerate(per_question, 1):

        s = item["score"]

        st.markdown(f"### 第 {i} 題")
        st.markdown(f"**題目：** {item['question']}")
        st.markdown(f"**你的回答：** {item['answer']}")

        st.write(
            f"- 技術：{s['technical']}/5 ｜ "
            f"表達：{s['communication']}/5 ｜ "
            f"結構：{s['structure']}/5 ｜ "
            f"相關性：{s['relevance']}/5 ｜ "
            f"解題：{s['problem_solving']}/5 ｜ "
            f"潛力：{s['growth_potential']}/5"
        )

        st.markdown(f"**AI 回饋：** {item['feedback']}")
        st.markdown("---")

# ============================================================
# PART 6 — 面試報告下載（Markdown / PDF / HTML）
# ============================================================

if st.session_state.grade_result:

    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]

    tech = overall["technical"]
    comm = overall["communication"]
    struct = overall["structure"]
    rel = overall["relevance"]
    ps = overall["problem_solving"]
    gp = overall["growth_potential"]

    sf = st.session_state.last_speech_features
    from grader import generate_speech_feedback

    st.markdown("## 💾 下載本次面試報告")

    # ------------------ 建立 Markdown 內容 ------------------
    def build_report_md():
        lines = []
        lines.append("# AI 虛擬面試官面試報告\n")
        lines.append(f"- 受試者：{st.session_state.candidate_id}")
        lines.append(f"- 應徵職缺：{job_role}")
        lines.append(f"- 日期：{datetime.now().strftime('%Y-%m-%d')}\n")

        # 整體評分
        lines.append("## 整體評分")
        lines.append(f"- 技術能力（Technical）：{tech}/5")
        lines.append(f"- 表達能力（Communication）：{comm}/5")
        lines.append(f"- 回答結構（Structure）：{struct}/5")
        lines.append(f"- 相關性（Relevance）：{rel}/5")
        lines.append(f"- 解題能力（Problem Solving）：{ps}/5")
        lines.append(f"- 成長潛力（Growth Potential）：{gp}/5\n")

        # 整體評論
        lines.append("## 整體評論")
        lines.append(overall["summary"] + "\n")

        # 語音分析（如果有）
        if sf:
            lines.append("## 語音表達分析")
            lines.append(f"- 語速（WPM）：{sf['wpm']}")
            lines.append(f"- 停頓比例：{sf['silence_ratio']}")
            lines.append(f"- 音量穩定度：{sf['volume_stability']}")
            lines.append(f"- 填充詞比例：{sf['filler_ratio']}\n")

            lines.append("## 語音改善建議（AI）")
            lines.append(generate_speech_feedback(sf) + "\n")
        else:
            lines.append("## 語音表達分析")
            lines.append("本次未提供語音回答，因此無語音分析與建議。\n")

        # 逐題回饋
        lines.append("## 逐題回饋（Question-by-Question）")
        for i, item in enumerate(per_question, 1):
            s = item["score"]
            lines.append(f"### 第 {i} 題")
            lines.append(f"- 題目：{item['question']}")
            lines.append(f"- 回答：{item['answer']}")
            lines.append(
                f"- 分數：技術 {s['technical']}/5，"
                f"表達 {s['communication']}/5，"
                f"結構 {s['structure']}/5，"
                f"相關性 {s['relevance']}/5，"
                f"解題 {s['problem_solving']}/5，"
                f"潛力 {s['growth_potential']}/5"
            )
            lines.append(f"- AI 回饋：{item['feedback']}\n")

        return "\n".join(lines)

    report_md = build_report_md()

    # ------------------ Markdown 下載 ------------------
    st.download_button(
        "📘 下載 Markdown 報告",
        data=report_md,
        file_name="interview_report.md",
        mime="text/markdown",
    )

    # ------------------ PDF 下載 ------------------
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

    # ------------------ HTML 下載 ------------------
    html_content = export_html(report_md)
    st.download_button(
        "🌐 下載 HTML 報告",
        data=html_content,
        file_name="interview_report.html",
        mime="text/html",
    )
