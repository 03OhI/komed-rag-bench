#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claim_extract.py — 기사 본문 → 검증 가능한 주장 문장 (End-to-end 의 첫 단계)

    from claim_extract import extract_claims
    claims = extract_claims(text, retriever, llm=None, max_claims=3)
    → [{"claim": "...", "sentence": "원문 문장", "n_concepts": 2, "concepts": ["오메가-3→Fatty Acids, Omega-3", ...]}, ...]

규칙 (mine_claims.py 와 같은 관문 — 평가 주장 30건을 뽑을 때 쓴 기준을 시스템도 그대로 쓴다)
    1) 문장 15~120자 · 인용부호/기자명/URL/연락처 없음
    2) 관계어(낮춘다·예방·위험·효과…) 포함 — 검증 가능한 명제
    3) retrieve.Retriever.concepts() 로 MeSH 개념 1개 이상 (2개 이상 우선)
    후보를 개념 수 → 등장 순서로 정렬해 상위를 고른다.

LLM 정규화 (선택 — llm 을 주면)
    긴 기사 문장은 개념이 너무 많이 잡혀 교집합이 안 되고, 수식어가 많아 BM25 질의어가 흐려진다.
    "누가/무엇이 – 무엇에 – 어떤 효과" 만 남긴 단문으로 다시 쓴다. 평가 주장 R01~R15 도 사람이 이렇게 줄였다.
    다시 쓴 주장은 원문 문장과 함께 보여 준다 — 시스템이 무엇을 검증했는지 사용자가 확인할 수 있어야 한다.
"""
from __future__ import annotations

import re

MED_HINT = re.compile(
    "건강|질환|질병|치료|예방|증상|환자|의료|영양|섭취|복용|면역|염증|혈압|혈당|콜레스테롤|"
    "체중|비만|다이어트|수면|피로|관절|골다공|당뇨|고혈압|암|심장|간|위장|장|피부|모발|"
    "비타민|오메가|프로바이오틱|유산균|효능|효과|개선|완화|감소|억제|백신|약|성분")
REL = re.compile(
    "낮춘|낮추|낮아|감소|줄인|줄여|줄이|높인|높이|높아|증가|늘린|예방|개선|완화|억제|"
    "유발|악화|도움|효과|효능|위험|치료|완치|권장|필요|중요")
BAD = re.compile(r"[\"“”'‘’]|기자|@|http|www\.|010-|문의|출시|판매|가격|원\)|㈜|주식회사")
# 인용·평가·전망 문장은 명제가 아니다 (LLM 정규화 없이 규칙만 쓸 때 거른다)
ATTRIB = re.compile(r"(평가|설명|전망|계획|입장|방침|분석|관측|목표)(이|으로)?다\.?$|밝혔다|말했다|전했다|강조했다|덧붙였다|설명했다|소개했다")
# 리졸버가 의학 개념으로 잡는 1음절 동음이의어 — 검색 계층(retrieve.py) 동결 해제 후 관문으로 옮길 것
JUNK_SPANS = {"이", "점", "등", "시", "상", "것", "수", "때", "곳", "말", "중", "대", "전"}
_PART = re.compile(r"(에서는|으로는|에게서|이라는|라는|에서|으로|에게|부터|까지|이나|든지|은|는|이|가|을|를|의|에|와|과|도|만|로)$")
_SENT = re.compile(r"(?<=[.!?다])\s+")


def real_concepts(concepts: list[dict]) -> list[dict]:
    """1음절 동음이의어(이→Tooth, 점→Nevus, 등→Back …)를 뺀 개념 목록."""
    out = []
    for c in concepts or []:
        span = _PART.sub("", (c.get("span") or "").strip())
        if span in JUNK_SPANS:
            continue
        out.append(c)
    return out

TOOL = {
    "name": "submit_claims",
    "description": "기사 문장을 검증 가능한 단문 주장으로 다시 쓴다.",
    "input_schema": {"type": "object", "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "후보 문장 번호"},
            "claim": {"type": "string", "description": "검증 가능한 단문 (개입/노출 – 대상 – 결과), 40자 안팎"},
            "verifiable": {"type": "boolean", "description": "문헌으로 검증할 수 있는 명제인가"}},
            "required": ["n", "claim", "verifiable"]}}},
        "required": ["claims"]}}

SYSTEM = """당신은 한국어 건강뉴스 문장에서 **문헌으로 검증 가능한 주장**을 추리는 편집자다.
각 후보 문장을 '개입·노출 – (대상) – 결과' 구조의 단문으로 다시 쓴다. 예:
  "아르기닌은 신체 신진대사와 해독 작용에 관여하는 주요 아미노산으로, 간 기능 개선과 혈관 기능 향상, 근육 회복을 돕는 성분으로 잘 알려져 있다."
  → "아르기닌은 간 기능 개선과 혈관 기능 향상, 근육 회복을 돕는 아미노산이다."
규칙: 원문에 없는 내용을 넣지 않는다 · 의학 용어(성분·질환·지표)는 원문 표기를 유지한다 · 제품명·상표는 그대로 둔다(범위 초과 판정에 필요) ·
제품 홍보 문구·연락처·가격만 있는 문장은 verifiable=false."""


def mock_claims(system: str, prompt: str, tool: dict) -> dict:
    """키 없이 돌릴 때: 후보 문장을 그대로 주장으로 쓴다 (verifiable=True)."""
    items = re.findall(r"^\[(\d+)\] (.+)$", prompt, re.M)
    return {"claims": [{"n": int(n), "claim": t.strip(), "verifiable": True} for n, t in items]}


def sentences(text: str):
    for s in _SENT.split(re.sub(r"\s+", " ", text or "")):
        s = s.strip()
        if 15 <= len(s) <= 120:
            yield s


def candidates(text: str, retriever, limit: int = 8) -> list[dict]:
    out = []
    for i, s in enumerate(sentences(text)):
        if not (MED_HINT.search(s) and REL.search(s)) or BAD.search(s) or ATTRIB.search(s):
            continue
        cons = real_concepts(retriever.concepts(s))
        if not cons:
            continue
        out.append({"pos": i, "sentence": s, "n_concepts": len(cons),
                    "concepts": [f"{c['span']}→{c['name']}" for c in cons]})
    out.sort(key=lambda d: (-min(d["n_concepts"], 3), d["pos"]))    # 개념 3개 이상은 동급, 그다음 등장 순서
    return out[:limit]


def extract_claims(text: str, retriever, llm=None, max_claims: int = 3) -> list[dict]:
    cands = candidates(text, retriever, limit=max(8, max_claims * 3))
    if not cands:
        return []
    if llm is None:
        return [{"claim": c["sentence"], **c} for c in cands[:max_claims]]
    prompt = "[후보 문장]\n" + "\n".join(f"[{i + 1}] {c['sentence']}" for i, c in enumerate(cands)) + \
             f"\n\n검증 가능한 것만 골라 최대 {max_claims}개를 submit_claims 로 제출하라. 후보 번호 n 을 함께 적는다."
    try:
        out = llm.call(SYSTEM, prompt, TOOL)
    except Exception:                            # noqa: BLE001 — LLM 실패 시 규칙 결과로
        return [{"claim": c["sentence"], **c} for c in cands[:max_claims]]
    res = []
    for item in out.get("claims", []):
        n = int(item.get("n") or 0)
        if not item.get("verifiable") or not (1 <= n <= len(cands)) or not item.get("claim"):
            continue
        c = cands[n - 1]
        res.append({"claim": item["claim"].strip(), **c})
        if len(res) >= max_claims:
            break
    return res or [{"claim": c["sentence"], **c} for c in cands[:max_claims]]


if __name__ == "__main__":
    import json, sys
    from retrieve import Retriever
    txt = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else \
        "오메가3의 EPA·DHA가 기억력 개선과 혈중 중성지질 개선에 도움을 준다. 특히 40대 이상은 꾸준히 복용하는 것이 좋다. 자세한 문의는 02-000-0000."
    r = Retriever(boost=0.0, verbose=False)
    print(json.dumps(extract_claims(txt, r), ensure_ascii=False, indent=1))
