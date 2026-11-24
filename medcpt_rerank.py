#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medcpt_rerank.py — MedCPT cross-encoder 재순위 (2026-11-08 팀 결정)

무엇
    검색 결과 상위 N건을 (영어 질의, 제목+초록) 쌍으로 MedCPT cross-encoder(ncbi/MedCPT-Cross-Encoder, NCBI 공개,
    PubMed 검색 기록으로 학습)에 넣어 점수 순으로 다시 매긴다. API 비용이 없다(로컬·코랩 GPU).
    MeSH 효과가 아니라 시스템 개선이다 — 논문에도 그렇게 따로 적는다.

영어 질의를 어디서 얻나 (rerank_query)
    1) LLM 질의 확장이 켜져 있으면 그 english_claim
    2) 주장이 이미 영어면 주장 그대로
    3) 둘 다 아니면(한국어, LLM 없음) 리졸버가 찾은 MeSH 표목명 + 관계어의 영어 + 주장 속 영문 용어를 이은 키워드 질의.
       MedCPT 는 PubMed 사용자 질의(대개 짧은 키워드)로 학습되어 이런 질의도 받는다. 비용 0.

평가기 신호는 바꾸지 않는다 — 재순위는 이미 고른 문헌의 순서만 바꾼다.
"""
from __future__ import annotations

import re

MODEL = "ncbi/MedCPT-Cross-Encoder"
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def doc_text(title: str | None, abstract: str | None) -> str:
    """MedCPT 사용 예처럼 제목 다음에 초록을 잇는다."""
    t, a = (title or "").strip(), (abstract or "").strip()
    if t and a:
        return t + ("" if t[-1] in ".?!" else ".") + " " + a
    return t or a


def is_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isascii()) / len(letters) >= 0.9


def keyword_query(claim: str, concepts: list[dict], predicate_map: dict | None = None) -> str:
    """한국어 주장 → 영어 키워드 질의 (LLM 없음). 표목명 · 관계어 첫 단어 · 주장 속 영문 용어."""
    parts: list[str] = []
    for c in concepts:
        nm = c.get("name")
        if nm and nm not in parts:
            parts.append(nm)
    for key, eng in (predicate_map or {}).items():
        if key in claim:
            w = eng.split()[0]
            if w not in parts:
                parts.append(w)
    for w in _ASCII_WORD.findall(claim):
        if w not in parts and w.lower() not in (p.lower() for p in parts):
            parts.append(w)
    return " ".join(parts)


def rerank_query(claim: str, concepts: list[dict], llm_info: dict | None = None,
                 predicate_map: dict | None = None) -> tuple[str, str]:
    """(영어 질의, 출처) — 출처는 llm · claim · mesh_keywords · none."""
    if llm_info and (llm_info.get("english_claim") or "").strip():
        return llm_info["english_claim"].strip(), "llm"
    if is_english(claim):
        return claim.strip(), "claim"
    q = keyword_query(claim, concepts, predicate_map)
    return (q, "mesh_keywords") if q else ("", "none")


class MedCPTReranker:
    """transformers 가 있는 환경(코랩·로컬)에서만 만든다. 처음 쓸 때 모델을 내려받는다(약 440MB)."""

    def __init__(self, model: str = MODEL, device: str | None = None, max_len: int = 512, batch: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.max_len, self.batch, self.name = max_len, batch, model

    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(docs), self.batch):
                pairs = [[query, d] for d in docs[i:i + self.batch]]
                enc = self.tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=self.max_len)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out.extend(self.model(**enc).logits.squeeze(dim=1).float().cpu().tolist())
        return out


class FakeReranker:
    """모델 없이 경로만 확인하는 가짜 채점기: 질의 단어가 문서에 몇 번 나오는지. 수치는 뜻이 없다."""
    name = "fake"

    def score(self, query: str, docs: list[str]) -> list[float]:
        qs = {w.lower() for w in _ASCII_WORD.findall(query)}
        return [float(sum(1 for w in _ASCII_WORD.findall(d) if w.lower() in qs)) for d in docs]


def make_reranker(spec: str, **kw):
    if spec == "medcpt":
        return MedCPTReranker(**kw)
    if spec == "fake":
        return FakeReranker()
    raise ValueError(f"모르는 재순위기: {spec} (medcpt · fake)")


def rerank(order: list, scores: list[float]) -> list:
    """점수 내림차순, 동점이면 원래 순위."""
    return [d for _, _, d in sorted(((-s, i, d) for i, (d, s) in enumerate(zip(order, scores))), key=lambda x: (x[0], x[1]))]
