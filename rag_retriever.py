# rag_retriever.py
import os
import glob
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 將中文職缺名稱映射成內部 job_type
JOB_ROLE_TO_TYPE = {
    "後端工程師": "backend",
    "AI 工程師": "ai",
    "資料工程師": "data",
    "前端工程師": "frontend",
}


@dataclass
class Chunk:
    text: str
    job_type: str  # backend / ai / data / frontend / common
    embedding: np.ndarray


class RagRetriever:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.chunks: List[Chunk] = []
        self._build_index()

    # 建立向量索引（只在啟動時跑一次）
    def _build_index(self):
        pattern = os.path.join(self.knowledge_dir, "*.md")
        files = glob.glob(pattern)

        raw_chunks: List[tuple[str, str]] = []

        for path in files:
            filename = os.path.basename(path)

            if filename.startswith("backend"):
                job_type = "backend"
            elif filename.startswith("ai"):
                job_type = "ai"
            elif filename.startswith("data"):
                job_type = "data"
            elif filename.startswith("frontend"):
                job_type = "frontend"
            else:
                job_type = "common"

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # 以段落切（空行），再進一步切成長度約 400 字的 chunk
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for para in paragraphs:
                # 再依長度切片
                step = 400
                for i in range(0, len(para), step):
                    piece = para[i : i + step].strip()
                    if piece:
                        raw_chunks.append((piece, job_type))

        if not raw_chunks:
            return

        texts = [c[0] for c in raw_chunks]
        job_types = [c[1] for c in raw_chunks]

        # 呼叫 OpenAI embeddings
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        for emb, text, jt in zip(resp.data, texts, job_types):
            vec = np.array(emb.embedding, dtype=np.float32)
            self.chunks.append(Chunk(text=text, job_type=jt, embedding=vec))

    # 計算 cosine similarity
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, job_role_chinese: str, query: str, top_k: int = 3) -> List[str]:
        if not self.chunks:
            return []

        job_type = JOB_ROLE_TO_TYPE.get(job_role_chinese, None)

        # 先算 query 向量
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)

        scored: List[tuple[float, Chunk]] = []

        for c in self.chunks:
            # 若有 job_type，優先使用同類型 + common
            if job_type is not None and c.job_type not in (job_type, "common"):
                continue

            sim = self._cosine_sim(q_vec, c.embedding)
            scored.append((sim, c))

        # 沒東西就回空
        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c.text for (score, c) in scored[:top_k]]
        return top_chunks
