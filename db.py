# db.py
import sqlite3
from typing import List, Dict, Any

DB_FILE = "interview.db"


def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # 1. 使用者（受試者）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT UNIQUE,
            name TEXT
        )
    """)

    # 2. 面試紀錄（每一場面試）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT,
            job_role TEXT,
            timestamp TEXT,
            summary TEXT
        )
    """)

    # 3. Q&A（每場面試的所有題目）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS qa_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            question TEXT,
            answer TEXT
        )
    """)

    # 4. 分數表（6 項評分）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER,
            technical INTEGER,
            communication INTEGER,
            structure INTEGER,
            relevance INTEGER,
            problem_solving INTEGER,
            growth_potential INTEGER
        )
    """)

    conn.commit()
    conn.close()


# --- 寫入工具 ---

def save_candidate(candidate_id: str, name: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO candidates (candidate_id, name) VALUES (?, ?)",
        (candidate_id, name),
    )
    conn.commit()
    conn.close()


def save_interview(candidate_id: str, job_role: str, timestamp: str, summary: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interviews (candidate_id, job_role, timestamp, summary) VALUES (?, ?, ?, ?)",
        (candidate_id, job_role, timestamp, summary),
    )
    conn.commit()
    interview_id = cur.lastrowid
    conn.close()
    return interview_id


def save_qa(interview_id: int, q: str, a: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO qa_records (interview_id, question, answer) VALUES (?, ?, ?)",
        (interview_id, q, a),
    )
    conn.commit()
    conn.close()


def save_scores(interview_id: int, overall: Dict[str, Any]):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO scores 
        (interview_id, technical, communication, structure, relevance, problem_solving, growth_potential)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            interview_id,
            overall.get("technical", 0),
            overall.get("communication", 0),
            overall.get("structure", 0),
            overall.get("relevance", 0),
            overall.get("problem_solving", 0),
            overall.get("growth_potential", 0),
        ),
    )
    conn.commit()
    conn.close()


# --- 查詢工具 ---

def get_interviews(candidate_id: str) -> List[Dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, job_role, timestamp, summary FROM interviews WHERE candidate_id=? ORDER BY id DESC",
        (candidate_id,),
    )
    rows = cur.fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "interview_id": row[0],
            "job_role": row[1],
            "timestamp": row[2],
            "summary": row[3],
        })
    return result


def get_scores(interview_id: int) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM scores WHERE interview_id=?", (interview_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}

    return {
        "technical": row[2],
        "communication": row[3],
        "structure": row[4],
        "relevance": row[5],
        "problem_solving": row[6],
        "growth_potential": row[7],
    }


def get_qa(interview_id: int) -> List[Dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT question, answer FROM qa_records WHERE interview_id=?", (interview_id,))
    rows = cur.fetchall()
    conn.close()

    return [{"question": q, "answer": a} for q, a in rows]
