# src/coach.py
from __future__ import annotations

import os
import re
import json
import math
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Optional deps:
# - pypdf (or PyPDF2) for PDF text extraction
# - sentence_transformers for embeddings
# - llama_cpp for local LLM generation (GGUF)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore


@dataclass
class CoachConfig:
    pdf_path: str = "docs/Archery The Art of Repetition (Simon Needham ).pdf"
    cache_dir: str = ".cache/coach"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_chars: int = 900
    chunk_overlap: int = 140
    top_k: int = 6

    # generation mode:
    # "rules" | "rag" | "rag_llm"
    mode: str = "rag"

    # optional local LLM:
    gguf_path: str = "models/llm.gguf"
    llm_ctx: int = 2048
    llm_max_tokens: int = 420
    llm_temperature: float = 0.3


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _read_pdf_text(pdf_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Please `pip install pypdf`.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append(f"[page {i+1}]\n{txt}")
    return _clean_text("\n\n".join(pages))


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append(text[i:j])
        i = max(i + 1, j - overlap)
    return chunks


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


class _Embedder:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`.")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


class _LocalLLM:
    def __init__(self, gguf_path: str, ctx: int):
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed. `pip install llama-cpp-python`.")
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
        self.llm = Llama(model_path=gguf_path, n_ctx=ctx, verbose=False)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</END>"],
        )
        return (out["choices"][0]["text"] or "").strip()


class CoachRAG:
    """
    RAG coach:
      - load PDF (cached)
      - chunk
      - embed + store
      - retrieve top-k relevant chunks per query
      - synthesize “repetition-first” coaching text
    """

    def __init__(self, cfg: CoachConfig):
        self.cfg = cfg
        _ensure_dir(cfg.cache_dir)
        self._embedder: Optional[_Embedder] = None
        self._llm: Optional[_LocalLLM] = None

        self._chunks: List[str] = []
        self._emb: Optional[np.ndarray] = None
        self._index_loaded = False

    def _lazy_embedder(self) -> _Embedder:
        if self._embedder is None:
            self._embedder = _Embedder(self.cfg.embed_model)
        return self._embedder

    def _lazy_llm(self) -> _LocalLLM:
        if self._llm is None:
            self._llm = _LocalLLM(self.cfg.gguf_path, self.cfg.llm_ctx)
        return self._llm

    def _cache_paths(self) -> Tuple[str, str, str]:
        pdf_hash = _sha1(os.path.abspath(self.cfg.pdf_path) + str(os.path.getmtime(self.cfg.pdf_path)))
        base = os.path.join(self.cfg.cache_dir, f"idx_{pdf_hash}")
        return base + ".json", base + ".npy", base + ".meta.json"

    def build_or_load(self) -> None:
        if self._index_loaded:
            return

        idx_json, emb_npy, meta_json = self._cache_paths()

        if os.path.exists(idx_json) and os.path.exists(emb_npy) and os.path.exists(meta_json):
            with open(idx_json, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)
            self._emb = np.load(emb_npy).astype(np.float32)
            self._index_loaded = True
            return

        # Build
        text = _read_pdf_text(self.cfg.pdf_path)
        chunks = _chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap)

        emb = self._lazy_embedder().encode(chunks)

        with open(idx_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        np.save(emb_npy, emb)

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pdf_path": self.cfg.pdf_path,
                    "embed_model": self.cfg.embed_model,
                    "chunk_chars": self.cfg.chunk_chars,
                    "chunk_overlap": self.cfg.chunk_overlap,
                    "created_at": time.time(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        self._chunks = chunks
        self._emb = emb
        self._index_loaded = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self.build_or_load()
        assert self._emb is not None

        top_k = top_k or self.cfg.top_k
        q_emb = self._lazy_embedder().encode([query])[0]

        sims = []
        for i in range(len(self._chunks)):
            sims.append((i, float(np.dot(self._emb[i], q_emb))))  # normalized
        sims.sort(key=lambda x: x[1], reverse=True)

        out = []
        for idx, sim in sims[:top_k]:
            out.append({"rank": len(out) + 1, "sim": sim, "chunk": self._chunks[idx]})
        return out

    # -----------------------------
    # Synthesis
    # -----------------------------
    def _make_query(self, metrics: Dict[str, Any], shape: str, handedness: str, lang: str) -> str:
        # Keep it simple but informative for retrieval
        spread = metrics.get("spread", None)
        slope = metrics.get("slope_deg", None)
        offset = metrics.get("offset", {}) or {}
        dx = offset.get("dx", 0.0)
        dy = offset.get("dy", 0.0)
        return (
            f"archery repetition coaching. shape={shape}, handedness={handedness}, "
            f"spread={spread:.1f} px, slope={slope:.0f} deg, offset(dx={dx:.1f}, dy={dy:.1f}). "
            f"Need one-cue shot script, pass/fail, fallback, micro-drill. language={lang}."
        )

    def _fallback_compose(self, base_advice: Dict[str, Any], retrieved: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
        """
        No LLM. Still produce flexible output:
        - keep base advice’s single cue
        - attach retrieved “principles” as supporting bullets
        """
        cue = base_advice.get("single_cue") or base_advice.get("cue") or ""
        title = base_advice.get("title") or "Coaching"

        # Extract 2–3 short “principle lines” from retrieved chunks (very conservative)
        def pick_lines(txt: str) -> List[str]:
            lines = [l.strip() for l in re.split(r"[\n\.]", txt) if l.strip()]
            # prefer lines mentioning repetition / form / consistency / let down
            key = ["repeat", "repetition", "consistent", "let down", "routine", "shot", "form", "process",
                   "反復", "繰り返", "一致", "下ろ", "ルーティン", "フォーム", "過程", "重复", "一致", "放下", "流程"]
            scored = []
            for l in lines:
                s = sum(1 for k in key if k.lower() in l.lower())
                if 10 <= len(l) <= 140:
                    scored.append((s, l))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [l for _, l in scored[:3]]

        support = []
        for r in retrieved[:3]:
            support.extend(pick_lines(r["chunk"]))
        support = support[:5]

        # Minimal repeated-structure output
        if lang == "zh":
            pass_fail = "通过：这一靶能在同一节奏下稳定复现口令；失败：开始补偿/加力/拖时。"
            fallback = "崩了就放下重来；只保留“口令 + 节奏”，先别追分。"
            mental = "“同一件事，用同一种方式。”"
            script = f"【本靶唯一口令】{cue}\n【通过/失败】{pass_fail}\n【崩了就回滚】{fallback}"
            if support:
                script += "\n【来自书的提醒】" + "\n- " + "\n- ".join(support)
        elif lang == "ja":
            pass_fail = "PASS：同じテンポで合図を再現できる。FAIL：補正/力み/時間超過が出る。"
            fallback = "崩れたら必ず下ろす。合図＋テンポだけ守る（得点は捨てる）。"
            mental = "「同じことを、同じように」"
            script = f"【このエンドの一言】{cue}\n【合格/失敗】{pass_fail}\n【崩れたら】{fallback}"
            if support:
                script += "\n【本からのヒント】" + "\n- " + "\n- ".join(support)
        else:
            pass_fail = "PASS: you can repeat the cue with the same tempo. FAIL: you start compensating / tensing / overrunning time."
            fallback = "If it breaks, always let down. Protect cue + tempo; ignore score."
            mental = '"Same thing. Same way."'
            script = f"[One cue] {cue}\n[PASS/FAIL] {pass_fail}\n[If it breaks] {fallback}"
            if support:
                script += "\n[Book reminders]\n- " + "\n- ".join(support)

        out = dict(base_advice)
        out.update(
            {
                "title": title,
                "single_cue": cue,
                "pass_fail": pass_fail,
                "fallback": fallback,
                "mental_phrase": mental,
                "script": script,
                "rag": {"top_k": len(retrieved), "snippets": retrieved},
            }
        )
        return out

    def _llm_compose(self, base_advice: Dict[str, Any], retrieved: List[Dict[str, Any]], metrics: Dict[str, Any], shape: str, handedness: str, lang: str) -> Dict[str, Any]:
        """
        With local LLM:
        - feed retrieved chunks as context
        - ask for structured JSON to avoid verbose hallucination
        """
        cue = base_advice.get("single_cue") or base_advice.get("cue") or ""
        title = base_advice.get("title") or "Coaching"

        context = "\n\n".join([f"(ctx {i+1}, sim={r['sim']:.3f})\n{r['chunk']}" for i, r in enumerate(retrieved[: self.cfg.top_k])])
        prompt = f"""
You are an archery coach. Use ONLY the provided context excerpts as inspiration. Do not quote long passages.
Task: produce a repetition-first coaching script for the next end.

User stats:
- shape: {shape}
- handedness: {handedness}
- metrics: {json.dumps(metrics, ensure_ascii=False)}

Base cue (must keep consistent unless clearly wrong):
- cue: {cue}

Output MUST be valid JSON with keys:
title, single_cue, pass_fail, fallback, drill (name, how, duration_s), mental_phrase, script.
Language: {lang}  (zh=Chinese, ja=Japanese, en=English)
Keep single_cue short. pass_fail observable. fallback actionable. drill 30-120s.
End with </END>.

Context:
{context}
</END>
""".strip()

        llm = self._lazy_llm()
        txt = llm.generate(prompt, max_tokens=self.cfg.llm_max_tokens, temperature=self.cfg.llm_temperature)

        # best-effort JSON parse
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            obj = json.loads(txt[start : end + 1])
        except Exception:
            # fall back to non-LLM synthesis
            return self._fallback_compose(base_advice, retrieved, lang)

        # merge
        out = dict(base_advice)
        out.update(obj)
        out["rag"] = {"top_k": len(retrieved), "snippets": retrieved}
        return out

    def enhance_advice(
        self,
        base_advice: Dict[str, Any],
        metrics: Dict[str, Any],
        shape: str,
        handedness: str,
        lang: str,
    ) -> Dict[str, Any]:
        """
        Entry point used by UI:
          - mode rules: return base_advice
          - mode rag: RAG compose without LLM
          - mode rag_llm: RAG + local LLM compose
        """
        if self.cfg.mode == "rules":
            return base_advice

        query = self._make_query(metrics, shape, handedness, lang)
        retrieved = self.retrieve(query, top_k=self.cfg.top_k)

        if self.cfg.mode == "rag_llm":
            try:
                return self._llm_compose(base_advice, retrieved, metrics, shape, handedness, lang)
            except Exception:
                return self._fallback_compose(base_advice, retrieved, lang)

        return self._fallback_compose(base_advice, retrieved, lang)
