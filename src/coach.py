# src/coach.py
from __future__ import annotations

import os
import re
import json
import time
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

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

    # "rules" | "rag" | "rag_llm"
    mode: str = "rag"

    # local LLM (optional)
    gguf_path: str = "models/llm.gguf"
    llm_ctx: int = 2048
    llm_max_tokens: int = 420
    llm_temperature: float = 0.3

    # routing
    router: str = "fine"  # "coarse" | "fine"


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
            raise RuntimeError("sentence-transformers not installed. Please `pip install sentence-transformers`.")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class _Index:
    def __init__(self, cfg: CoachConfig):
        self.cfg = cfg
        _ensure_dir(cfg.cache_dir)
        self.idx_path = os.path.join(cfg.cache_dir, f"idx-{_sha1(cfg.pdf_path)}-{_sha1(cfg.embed_model)}.json")
        self.vec_path = os.path.join(cfg.cache_dir, f"vec-{_sha1(cfg.pdf_path)}-{_sha1(cfg.embed_model)}.npy")

        self.chunks: List[str] = []
        self.vecs: Optional[np.ndarray] = None

    def load_or_build(self) -> None:
        if os.path.exists(self.idx_path) and os.path.exists(self.vec_path):
            with open(self.idx_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)["chunks"]
            self.vecs = np.load(self.vec_path)
            return

        text = _read_pdf_text(self.cfg.pdf_path)
        chunks = _chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap)

        emb = _Embedder(self.cfg.embed_model)
        vecs = emb.embed(chunks)

        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False)

        np.save(self.vec_path, vecs)

        self.chunks = chunks
        self.vecs = vecs

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.vecs is None:
            raise RuntimeError("Index not loaded.")
        emb = _Embedder(self.cfg.embed_model)
        qv = emb.embed([query])[0]
        sims = [(_cosine_sim(qv, self.vecs[i]), i) for i in range(len(self.chunks))]
        sims.sort(reverse=True)
        out = []
        for s, i in sims[:top_k]:
            out.append({"score": float(s), "text": self.chunks[i]})
        return out


def _short_profile(profile: Dict[str, Any]) -> str:
    return _clean_text(
        f"""
name: {profile.get('name','')}
bow: {profile.get('bow','')}
experience_months: {profile.get('experience_months','')}
dominant_eye: {profile.get('dominant_eye','')}
goals: {profile.get('goals','')}
recurring_issues: {profile.get('recurring_issues','')}
constraints: {profile.get('constraints','')}
style: {profile.get('language_style','tight')}
""".strip()
    )


def _recent_log_summary(log: List[Dict[str, Any]], k: int = 3) -> str:
    if not log:
        return ""
    tail = log[-k:]
    lines = []
    for e in tail:
        dist = e.get("distance_m", "")
        face = e.get("target_face", "")
        scoring = e.get("scoring", {}) or {}
        metrics = e.get("metrics", {}) or {}
        lines.append(
            f"- dist={dist}m face={face} total={scoring.get('total','')} avg={scoring.get('avg','')} "
            f"spread={metrics.get('spread','')} slope={metrics.get('slope_deg','')} dx={metrics.get('offset',{}).get('dx','')} dy={metrics.get('offset',{}).get('dy','')}"
        )
    return "\n".join(lines)


def _route_topics(cfg: CoachConfig, metrics: Dict[str, Any], shape: str, scoring: Dict[str, Any]) -> List[str]:
    """
    Fine router: choose sharper topics so RAG retrieves the right pages.
    """
    topics = ["shot process", "repetition", "consistency"]

    spread = float(metrics.get("spread", 0.0) or 0.0)
    slope = float(metrics.get("slope_deg", 0.0) or 0.0)
    offset = metrics.get("offset", {}) or {}
    dx = float(offset.get("dx", 0.0) or 0.0)
    dy = float(offset.get("dy", 0.0) or 0.0)

    avg = float(scoring.get("avg", 0.0) or 0.0)

    if cfg.router == "coarse":
        if spread > 55:
            topics += ["form breakdown", "shot rhythm", "reset drill"]
        else:
            topics += ["fine aiming", "hold stability"]
        return topics

    # fine routing
    if spread > 70:
        topics += ["tension management", "let down", "tempo reset", "subtraction"]
    elif spread > 45:
        topics += ["anchor consistency", "string picture", "expand through clicker"]
    else:
        topics += ["micro-aim", "follow-through", "quiet bow hand"]

    # offset routing
    if abs(dx) > abs(dy) and abs(dx) > 12:
        topics += ["windage", "torque", "bow hand pressure", "grip"]
    if abs(dy) >= abs(dx) and abs(dy) > 12:
        topics += ["shoulder line", "front shoulder", "draw length", "anchor height"]

    # slope routing
    if 20 <= abs(slope) <= 70:
        topics += ["release", "expansion", "back tension", "string alignment"]

    # score routing
    if avg >= 9.0:
        topics += ["do less", "repeat the same", "protect the process"]
    elif avg >= 7.0:
        topics += ["one cue", "single variable", "confidence"]
    else:
        topics += ["rebuild", "simple drill", "no-score ends"]

    # shape routing
    if shape:
        topics.append(f"group shape {shape}")

    # de-dup
    out = []
    seen = set()
    for t in topics:
        k = t.strip().lower()
        if k and k not in seen:
            out.append(t)
            seen.add(k)
    return out


class CoachRAG:
    def __init__(self, cfg: CoachConfig):
        self.cfg = cfg
        self._index: Optional[_Index] = None
        self._llm: Optional[Any] = None

    def _ensure_index(self) -> None:
        if self.cfg.mode == "rules":
            return
        if self._index is None:
            self._index = _Index(self.cfg)
            self._index.load_or_build()

    def _ensure_llm(self) -> None:
        if self.cfg.mode != "rag_llm":
            return
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed. Please `pip install llama-cpp-python`.")
        if not os.path.exists(self.cfg.gguf_path):
            raise FileNotFoundError(f"GGUF not found: {self.cfg.gguf_path}")
        if self._llm is None:
            self._llm = Llama(
                model_path=self.cfg.gguf_path,
                n_ctx=self.cfg.llm_ctx,
                n_threads=max(2, os.cpu_count() or 4),
                verbose=False,
            )

    def enhance_advice(
        self,
        *,
        base_advice: Dict[str, Any],
        metrics: Dict[str, Any],
        shape: str,
        handedness: str,
        lang: str,
        scoring: Dict[str, Any],
        user_profile: Dict[str, Any],
        log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Return a dict compatible with rules.py UI, plus optional rag fields.
        """
        if self.cfg.mode == "rules":
            return dict(base_advice)

        self._ensure_index()

        topics = _route_topics(self.cfg, metrics, shape, scoring)
        profile_txt = _short_profile(user_profile)
        hist = _recent_log_summary(log, k=3)

        query = _clean_text(
            f"""
You are an archery coach. Retrieve from the book and give actionable next-end advice.

LANG={lang}
HANDEDNESS={handedness}

PROFILE:
{profile_txt}

RECENT LOG:
{hist}

METRICS:
{json.dumps(metrics, ensure_ascii=False)}

SCORING:
{json.dumps(scoring, ensure_ascii=False)}

TOPICS:
{', '.join(topics)}

Need: one cue, pass/fail, if it breaks, one 30-60s drill, a short shot script.
""".strip()
        )

        hits = self._index.search(query, top_k=self.cfg.top_k) if self._index else []
        snippets = "\n\n".join([f"[retrieval {i+1} score={h['score']:.3f}]\n{h['text']}" for i, h in enumerate(hits)])

        # rag-only: template synthesis without LLM
        if self.cfg.mode == "rag":
            out = dict(base_advice)
            out["rag"] = {"topics": topics, "top_k": self.cfg.top_k, "hits": hits[:3]}
            # Make the wording more flexible but deterministic:
            out["single_cue"] = base_advice.get("single_cue") or base_advice.get("cue") or _fallback_cue(lang, topics)
            out["pass_fail"] = base_advice.get("pass_fail") or _fallback_pass_fail(lang, metrics)
            out["fallback"] = base_advice.get("fallback") or _fallback_fallback(lang)
            out["drill"] = base_advice.get("drill") or _fallback_drill(lang, topics)
            out["script"] = base_advice.get("script") or _fallback_script(lang, out["single_cue"])
            return out

        # rag + local LLM
        self._ensure_llm()
        prompt = _build_prompt(lang, base_advice, query, snippets)

        resp = self._llm(
            prompt,
            max_tokens=self.cfg.llm_max_tokens,
            temperature=self.cfg.llm_temperature,
            stop=["</json>"],
        )
        text = (resp["choices"][0]["text"] or "").strip()

        out = dict(base_advice)
        parsed = _try_parse_json(text)
        if isinstance(parsed, dict):
            out.update(parsed)
        out["rag"] = {"topics": topics, "top_k": self.cfg.top_k, "hits": hits[:3]}
        return out


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _build_prompt(lang: str, base_advice: Dict[str, Any], query: str, snippets: str) -> str:
    # Force a stable JSON schema so UI stays compatible
    return _clean_text(
        f"""
You are a concise archery coach. Use the retrieved book snippets as grounding.
Return STRICT JSON only, no extra text.

Required keys:
title, single_cue, pass_fail, fallback, drill{{name,how,duration_s}}, script, mental_phrase

Language: {lang}

Base advice:
{json.dumps(base_advice, ensure_ascii=False)}

Query context:
{query}

Retrieved:
{snippets}

Return:
<json>
{{...}}
</json>
""".strip()
    )


def _fallback_cue(lang: str, topics: List[str]) -> str:
    if lang == "ja":
        return "次の6射は「同じテンポ」だけを守る。"
    if lang == "zh":
        return "下一组只盯一件事：节奏一致。"
    return "Next end: protect ONE thing—identical tempo."


def _fallback_pass_fail(lang: str, metrics: Dict[str, Any]) -> str:
    if lang == "ja":
        return "PASS: セットアップ→アンカーまで違和感なし / FAIL: 3秒で落ち着かなければ必ず下ろす"
    if lang == "zh":
        return "PASS：举弓到锚点无明显别扭；FAIL：3秒内不稳定就必须放下重来"
    return "PASS: set-up→anchor feels identical; FAIL: if not settling within 3s, always let down."


def _fallback_fallback(lang: str) -> str:
    if lang == "ja":
        return "崩れたら：矢は撃たず、空打ちでフォームだけ1回リセット。"
    if lang == "zh":
        return "一旦崩了：别硬放箭，空拉一次把动作重置。"
    return "If it breaks: no shot—one blank-bale rep to reset the feel."


def _fallback_drill(lang: str, topics: List[str]) -> Dict[str, Any]:
    if lang == "ja":
        return {"name": "60秒：テンポ固定", "how": "3秒以内に決まらなければ下ろす。×6回。", "duration_s": 60}
    if lang == "zh":
        return {"name": "60秒：节奏固定", "how": "3秒内不稳定就放下重来。做6次。", "duration_s": 60}
    return {"name": "60s: Tempo lock", "how": "If not settling within 3s, let down. x6 reps.", "duration_s": 60}


def _fallback_script(lang: str, cue: str) -> str:
    if lang == "ja":
        return f"セット→引き分け→アンカー→{cue}→リリース→フォロー"
    if lang == "zh":
        return f"举弓→开弓→锚点→{cue}→撒放→随动"
    return f"Set → Draw → Anchor → {cue} → Release → Follow-through"
