#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm25_util.py — PubMed 초록 BM25 공용 유틸

인덱스 빌드(build_pubmed_index.py)와 질의(retrieve.py)는 **반드시 같은**
토크나이저를 써야 df/dl 이 맞는다. 그래서 한 곳에 모아둔다.

정규화 방침
    복수형만 정리한다. -ing/-ed 는 건드리지 않는다.
    reduces→reduce / reduced→reduc / reducing→reduc 처럼 활용형끼리 어긋나
    오히려 매칭이 깨지기 때문이다. 동사 활용은 retrieve.py 의 술어 사전이
    표면형을 나열하는 쪽으로 처리한다. (규칙이 단순해 논문에 그대로 적을 수 있다)
"""
from __future__ import annotations

import math
import re
import unicodedata

TITLE_W = 3      # 제목 가중 (토큰 반복 횟수)
MESH_W = 2       # MeSH 표목 가중 — 통제어휘라 리졸브 결과와 직접 맞물린다

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")

# 초록 구조 라벨과 논문 상투어. IDF 로도 눌리지만 df 표를 키우지 않게 미리 뺀다.
STOP = set("""
a an and are as at be been being but by can could did do does for from had has have
he her his if in into is it its of on or our she that the their them there these they
this to was we were what when which who will with would you your
background method methods result results conclusion conclusions objective objectives
purpose aim aims design setting participants intervention interventions
study studies trial trials patient patients subject subjects group groups
significant significantly compare compared comparison versus vs using used use
show shown showed report reported found find data analysis analyses
p n ci sd se et al however also may might such than then thus while whether
""".split())


def stem(w: str) -> str:
    """복수형 최소 정규화. triglycerides→triglyceride, studies→study"""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """
    홑글자·한두 자리 숫자는 앞 낱말에 붙여 합성어를 하나 더 만든다.

    이게 없으면 'Vitamin D' 가 그냥 {vitamin} 이 된다('d' 는 짧아서 버려지므로).
    그러면 비타민 D 질의에 **비타민 K 논문이 1위로 올라온다** — 실측으로 그랬다.
    같은 문제가 Type 1/Type 2 당뇨, Hepatitis B/C, Omega-3 에 전부 걸린다.
    'vitamind' 같은 합성 토큰은 변별력이 매우 높아 이 혼동을 한 번에 없앤다.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    out: list[str] = []
    prev: str | None = None
    for t in _TOKEN.findall(text):
        if t in STOP:
            prev = None
            continue
        if len(t) <= 2 and prev:
            out.append(prev + t)          # vitamin+d · type+2 · omega+3 · covid+19
        if len(t) < 2:                    # 홑글자 자체는 변별력이 없어 버린다
            prev = None
            continue
        s = stem(t)
        if s in STOP:
            prev = None
            continue
        out.append(s)
        prev = s
    return out


def doc_tokens(d: dict) -> list[str]:
    """한 레코드의 BM25 대상 토큰열. 제목·MeSH 표목에 가중을 준다."""
    return (tokenize(d.get("title") or "") * TITLE_W
            + tokenize(" ".join(d.get("mesh") or [])) * MESH_W
            + tokenize(d.get("abstract") or ""))


def idf(df_t: int, n_docs: int) -> float:
    return math.log(1.0 + (n_docs - df_t + 0.5) / (df_t + 0.5))


def bm25(q_terms, tf: dict, dl: int, df: dict, n_docs: int, avgdl: float,
         k1: float = K1, b: float = B) -> float:
    """
    q_terms : {용어: 가중} 또는 용어 집합(가중 1.0).

    가중을 두는 이유 — 질의어는 출처가 다르고 신뢰도도 다르다. MeSH 표목명은
    주장의 핵심 개념이지만, 별칭에서 흘러나온 토큰('fish oils'→'oil')이나
    관계어('lower','reduce')는 그만한 대접을 받으면 안 된다. 가중이 없으면
    관계어 9개가 개념어 4개를 눌러 엉뚱한 문서가 올라온다.
    """
    if not isinstance(q_terms, dict):
        q_terms = {t: 1.0 for t in q_terms}
    s = 0.0
    norm = k1 * (1.0 - b + b * dl / avgdl)
    for t, w in q_terms.items():
        f = tf.get(t, 0)
        if not f:
            continue
        s += w * idf(df.get(t, 0), n_docs) * f * (k1 + 1.0) / (f + norm)
    return s
