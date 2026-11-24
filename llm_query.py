#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_query.py — 검색 질의 LLM 번역·확장 (2026-11-08 팀 결정, E3_MeSH검색_개선안_1108.md §5)

무엇을 하나
    주장(한국어 또는 영어)과 리졸버가 찾은 MeSH 개념을 LLM 에 주고, 한 번의 호출로 셋을 받는다.
        english_claim : 주장을 영어로 옮긴 문장 (이미 영어면 그대로)
        passage       : PubMed 초록 문체의 짧은 영어 의사 문서 (지지·반박 근거를 모두 담는 중립 서술)
        keywords      : 영어 검색어 5~12개
    retrieve.py 는 이것을 BM25 질의어로만 보탠다. 개념 식별자(MeSH UI)는 지금처럼 동결 사전에서만 온다.

무엇을 하지 않나
    · 리졸버를 바꾸지 않는다 — 한국어 → MeSH 연결은 번역 없이 그대로(추계 원고·E4 의 주장 유지).
    · 평가기 신호(개념 수·후보 수·1위 점수·모드)를 바꾸지 않는다 — retrieve.py 가 원래 질의어로 계산한다.
    · 주장이 참인지 판단하지 않는다 — 프롬프트가 중립 서술을 요구한다(관련성은 지지 여부가 아니다).

재현성 (질의 시점 번역을 뺐던 11/2 이전의 이유에 대한 대응)
    temperature 0 + komed_llm 캐시(같은 모델·프롬프트·도구면 같은 출력) · 출력은 검색 결과에 함께 기록.

MeSH 정보 있음/없음 (use_mesh)
    E3 에서 "MeSH 덕분인가"를 가르려고 둔다. 있음 = 정의(scope note) · 상위/하위 표목 · entry term 을 프롬프트에 넣음.
    없음 = 주장만 준다. 두 조건은 같은 LLM 이다.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

from bm25_util import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
DESC_INFO = "mesh_desc_info.pkl"     # build_mesh_desc_info.py 가 desc2026.xml 에서 만든다

TOOL = {
    "name": "search_text",
    "description": "English search text for a BM25 search engine over PubMed titles and abstracts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "english_claim": {"type": "string",
                              "description": "The claim in plain English. Translate it if it is not English; keep it as is if it is."},
            "passage": {"type": "string",
                        "description": "An 80-150 word passage written like a PubMed abstract on the topic of the claim, "
                                       "using standard biomedical terminology. Neutral: mention what studies would measure, "
                                       "and allow for both supporting and contradicting findings."},
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "5-12 English search terms or short phrases, preferring MeSH-style vocabulary."},
        },
        "required": ["english_claim", "passage", "keywords"],
    },
}

SYSTEM_BASE = ("You prepare search text for a BM25 search engine over PubMed titles and abstracts. "
               "Your output is used only to find relevant studies, including studies that contradict the claim. "
               "Do not judge whether the claim is true and do not invent study results, numbers, or citations.")
SYSTEM_MESH = SYSTEM_BASE + (" MeSH concept information linked to the claim is given; use it to choose correct "
                             "biomedical vocabulary (preferred names, broader and narrower terms, synonyms).")

_info = None
_info_lock = threading.Lock()


def mesh_info(path: str | None = None) -> dict:
    global _info
    with _info_lock:
        if _info is None:
            p = path or os.path.join(HERE, DESC_INFO)
            with open(p, "rb") as f:
                info = pickle.load(f)
            ch: dict[str, list[str]] = {}
            for tn, u in sorted(info["_tree2ui"].items()):   # 하위 표목: 트리 번호 한 마디 아래
                if "." in tn:
                    ch.setdefault(tn.rsplit(".", 1)[0], []).append(u)
            info["_children"] = ch
            _info = info
    return _info


def concept_context(uis: list[str], max_rel: int = 4, max_entries: int = 5) -> list[dict]:
    """UI 목록 → 프롬프트용 MeSH 정보. 상위 = 트리 번호의 마지막 마디를 뗀 것, 하위 = 한 마디 아래."""
    info = mesh_info()
    t2u = info["_tree2ui"]
    out, seen = [], set()
    for ui in uis:
        if ui in seen or ui not in info:
            continue
        seen.add(ui)
        v = info[ui]
        parents, children = [], []
        for tn in v["trees"]:
            if "." in tn:
                p = t2u.get(tn.rsplit(".", 1)[0])
                if p and info[p]["name"] not in parents:
                    parents.append(info[p]["name"])
        for tn in v["trees"]:
            for u in info["_children"].get(tn, ()):
                nm = info[u]["name"]
                if nm not in children:
                    children.append(nm)
        entries = [e for e in v["entries"] if ", " not in e][:max_entries]   # 'Neoplasms, Breast' 같은 도치형은 뺀다
        out.append({"ui": ui, "name": v["name"], "scope": v["scope"][:300],
                    "broader": parents[:max_rel], "narrower": children[:max_rel], "also": entries})
    return out


def build_prompt(claim: str, concepts: list[dict], use_mesh: bool) -> tuple[str, str]:
    if not use_mesh:
        return SYSTEM_BASE, f"Claim: {claim}\n"
    ctx = concept_context([c["ui"] for c in concepts])
    lines = [f"Claim: {claim}", "", "MeSH concepts linked to the claim:"]
    if not ctx:
        lines.append("(none)")
    for c in ctx:
        lines.append(f"- {c['name']} ({c['ui']}): {c['scope']}")
        if c["broader"]:
            lines.append(f"  broader: {'; '.join(c['broader'])}")
        if c["narrower"]:
            lines.append(f"  narrower: {'; '.join(c['narrower'])}")
        if c["also"]:
            lines.append(f"  also called: {'; '.join(c['also'])}")
    return SYSTEM_MESH, "\n".join(lines) + "\n"


def _cache_key(llm, system: str, prompt: str) -> str:
    # komed_llm.LLM.call 과 같은 키 — prefetch 로 채운 캐시를 call 이 그대로 쓴다
    return hashlib.sha1(f"{llm.label}\n{system}\n{prompt}\n{TOOL['name']}".encode("utf-8")).hexdigest()


def expand(llm, claim: str, concepts: list[dict], use_mesh: bool = True) -> dict:
    system, prompt = build_prompt(claim, concepts, use_mesh)
    out = llm.call(system, prompt, TOOL)
    kws = out.get("keywords") or []
    if isinstance(kws, str):
        kws = [kws]
    return {"english_claim": str(out.get("english_claim") or ""), "passage": str(out.get("passage") or ""),
            "keywords": [str(k) for k in kws][:12], "use_mesh": use_mesh, "model": llm.label}


def prefetch(llm, items: list[tuple[str, list[dict], bool]], workers: int = 8) -> int:
    """여러 (주장, 개념, use_mesh) 를 병렬로 미리 호출해 캐시에 넣는다. 반환: 새로 부른 횟수.
    캐시 파일 쓰기만 잠근다(komed_llm 의 줄 단위 jsonl 형식 그대로)."""
    lock = threading.Lock()
    todo = {}
    for claim, cons, use_mesh in items:
        system, prompt = build_prompt(claim, cons, use_mesh)
        k = _cache_key(llm, system, prompt)
        if k not in llm.cache and k not in todo:
            todo[k] = (system, prompt)

    def one(kv):
        k, (system, prompt) = kv
        last = None
        for _ in range(3):
            try:
                out = llm._dispatch(system, prompt, TOOL)
                break
            except Exception as e:             # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"LLM 호출 실패 ({llm.label}): {last}")
        with lock:
            llm.cache[k] = out
            llm.n_calls += 1
            if llm.cache_path:
                import time
                with open(llm.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": out, "m": llm.label, "t": time.strftime("%Y-%m-%d %H:%M")},
                                       ensure_ascii=False) + "\n")
        return 1

    if not todo:
        return 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return sum(ex.map(one, todo.items()))


def terms(exp: dict, w_claim: float = 1.0, w_passage: float = 0.3) -> dict[str, float]:
    """LLM 출력 → BM25 질의어 가중. 영어 주장·검색어는 w_claim, 의사 문서는 출현 횟수(최대 3)에 비례해 w_passage."""
    q: dict[str, float] = {}
    for t in tokenize(exp.get("english_claim", "") + " " + " ".join(exp.get("keywords") or [])):
        q[t] = max(q.get(t, 0.0), w_claim)
    cnt: dict[str, int] = {}
    for t in tokenize(exp.get("passage", "")):
        cnt[t] = cnt.get(t, 0) + 1
    for t, c in cnt.items():
        q[t] = max(q.get(t, 0.0), w_passage * min(c, 3))
    return q


def merge(base: dict[str, float], extra: dict[str, float]) -> dict[str, float]:
    """원래 질의어와 LLM 질의어를 합친다. 같은 용어는 큰 가중을 쓴다."""
    q = dict(base)
    for t, w in extra.items():
        if w > q.get(t, 0.0):
            q[t] = w
    return q


def mock_fn(system: str, prompt: str, tool: dict) -> dict:
    """키 없이 경로를 돌려 보는 가짜 응답: 주장과 MeSH 표목명을 그대로 돌려준다."""
    claim = prompt.split("\n", 1)[0].replace("Claim: ", "")
    names = [ln[2:].split(" (", 1)[0] for ln in prompt.splitlines() if ln.startswith("- ")]
    return {"english_claim": claim, "passage": " ".join([claim] + names), "keywords": names or [claim]}
