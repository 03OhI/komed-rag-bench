#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve.py — 검색 계층 (MeSH UI 필터 + BM25, 전역 BM25 와 RRF 융합)

파이프라인
    한국어 주장
      1) 개념 추출   : 어절 n-gram(3→2→1) 최장일치 → KoResolver → MeSH UI
      2) 후보 좁히기 : ui2docs 로 문서 후보 축소. 축이 둘 이상이면 **교집합** 우선
      3) 랭킹        : 후보 안에서 BM25 (전역 df/avgdl 사용) = 게이트 순위
                       + 같은 질의어의 전역 BM25 상위 100 (색인 postings) 을 RRF(k=60) 로 융합 → top-k
                       (11/2 E3 NFCorpus 실측: 게이트 단독 nDCG@10 −0.034 vs BM25, 게이트+RRF −0.003 ·
                        Recall@100 +0.026 → 게이트+RRF 채택. E3_NFCorpus_결과_1102.md)
      4) (다음 단계) Retrieval Evaluator 로 correct/ambiguous/incorrect 판정
                       — 평가기의 네 신호(n_concepts · mode · n_scored · top_score)는 **게이트 쪽 후보에서
                       그대로** 계산한다. 10/23 에 맞춘 임계값과 호환되게 두려는 것이다.

포스팅이 없는 옛 색인(version 1)이면 경고를 내고 게이트 단독(융합 없음)으로 돌아간다.
    python build_pubmed_index.py --force   # 색인을 다시 만들면 융합이 켜진다

한국어 주장을 영문 초록에 어떻게 잇는가
    개념 식별자는 번역 없이 리졸버가 정한다. 리졸버가 돌려준 **MeSH 표목의 영문 표면형**이 기본 다리다.
        "오메가-3" → Fatty Acids, Omega-3 → {fatty, acid, omega, 3, ...}
    관계어(낮춘다/증가/예방…)만 사전으로 표면형을 붙인다. 활용형을 나열해두는
    쪽이 어간추출로 억지로 맞추는 것보다 예측 가능하다.

    (11/8 팀 결정) 검색 질의어에는 LLM 번역·확장을 더할 수 있다 — llm_query.py, 기본 꺼짐.
        llm_mode = off    : 위 규칙만 (11/2 E3 의 시스템 행과 같다)
                   global : 전역 BM25 다리의 질의어에 LLM 질의어(영어 주장·검색어·의사 문서)를 더한다
                   leg    : LLM 질의어만으로 만든 전역 BM25 순위를 RRF 의 셋째 다리로 더한다
                   all    : global + 게이트 후보의 순위도 합친 질의어로 매긴다
    어느 모드든 개념 식별자와 평가기 신호(n_concepts · mode · n_scored · top_score)는 원래 질의어로 계산한다.
    어느 모드를 켤지는 NFCorpus dev 에서 고른다(E3_MeSH검색_개선안_1108.md).

    (11/8 팀 결정) 재순위: reranker=medcpt_rerank.MedCPTReranker 를 주면 융합 순위 상위 rerank_top 건을
    MedCPT cross-encoder 로 다시 매긴다(기본 꺼짐, API 비용 없음). 영어 질의는 LLM 번역 → 영어 주장 그대로 →
    MeSH 표목명 키워드 순으로 고른다(medcpt_rerank.rerank_query). 평가기 신호는 바꾸지 않는다.

BM25 를 먼저 두는 이유
    의존성이 적고 빠르며 무엇보다 **기준선이 명확하다**. 임베딩 트랙을 먼저 넣으면
    비교 대상이 없어 개선을 주장할 수 없다. 임베딩은 이 위에 두 번째 트랙으로
    올려 MeSH 리졸버와 동일하게 RRF(k=60)로 융합한다 — 설계 일관성.

사용
    python build_pubmed_index.py          # 최초 1회 (수집이 늘면 다시)
    python retrieve.py "오메가-3가 중성지방을 낮춘다"
    python retrieve.py "..." -k 10 --explain --boost 1.0
    python retrieve.py "..." --no-fuse          # 게이트 단독 (E3 의 gate 행과 같은 규칙)
    python retrieve.py "..." --llm gemini --llm-mode global --explain   # LLM 질의 확장 (키: GEMINI_API_KEY 또는 gemini_key.txt)
    python retrieve.py "..." --rerank medcpt --explain                  # MedCPT 재순위 (pip install transformers torch)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

from bm25_util import bm25, doc_tokens, tokenize, idf, K1, B   # noqa: E402
from mesh_resolve import normalize, _load_lookup          # noqa: E402

INDEX = os.environ.get("KOMED_INDEX", "pubmed_index.pkl")   # 11/10: 쌍 보강 색인 등 다른 색인으로 바꿔 볼 때만 환경변수로 지정
KB = "mesh_lookup.pkl"

MAX_NGRAM = 3          # 어절 n-gram 최대 길이
KO_MIN_SCORE = 88.0    # ko_resolver 퍼지 임계값과 맞춤
MAX_UI_RATIO = 0.20    # 코퍼스의 20% 넘게 붙은 UI 는 변별력이 없다고 보고 제외
MIN_INTERSECT = 20     # 교집합이 이보다 작으면 2축 교집합 → 합집합으로 물러난다.
                       # 3축을 다 만족하는 문서만 보면 6건까지 줄어 정작 근거를 놓친다.
MAX_CANDS = 20000
FUSE_K = 60            # RRF 상수 (bench/run_nfcorpus.py 의 gate_rrf 와 동일)
FUSE_TOP = 100         # 융합에 넣는 게이트 순위·전역 순위의 깊이 (E3 에서 top=100)
LLM_MODES = ("off", "global", "leg", "all")   # 11/8 — 맨 위 설명 참고. 기본 off
RERANK_TOP = 30        # MedCPT 재순위할 상위 건수 (판독은 3건을 읽는다). 11/9 NFCorpus dev: 상위 30 0.323 · 상위 100 0.327,
                       # 차이 0.005 안이라 사전 규칙대로 30 (nDCG@3 은 둘 다 0.373)

# 질의어 가중 — 출처별로 신뢰도가 다르다 (bm25_util.bm25 참고)
W_NAME = 1.00        # MeSH 표목명 = 주장의 핵심 개념
W_ALIAS = 0.35       # 별칭에서 흘러나온 토큰
W_PRED = 0.30        # 관계어 (낮춘다 → lower/reduce/…)
W_ASCII = 1.00       # 주장에 그대로 박힌 영문 용어
ALIAS_MAX_DF = 0.02  # 코퍼스 2% 넘게 나오는 흔한 토큰은 별칭 확장에서 제외

# ── 오탐 방지 ────────────────────────────────────────────────────────────────
# 실제 기사 문장에 걸어보면 리졸버가 일상어를 의학 개념으로 바꿔버린다(실측):
#     국내→Palate  신체→Cadaver  이들→Id  대상→Metabolism  지난→Diagnosis
#     시→Poetry    등→Back       친구→Friends  전신→Pronephros
# 앞 여섯은 2음절에 대한 자모 퍼지(score 89~91)가 원인이고, 뒤 넷은 한국어
# 동음이의어가 사전에 정말 들어 있어서 exact 로 잡힌다. 손으로 만든 주장에는
# 안 나타나고 실제 기사에서만 터진다.
#
# 사전을 고치는 대신 **검색 계층에서** 두 관문으로 막는다. ko_resolver 의 계약
# (method enum, score 스케일)은 건드리지 않는다.
MIN_SPAN_FOR_FUZZY = 4   # 3음절 이하에서 자모 퍼지는 신뢰할 수 없다
MIN_UI_DOCS = 30         # 코퍼스에 문헌이 이만큼도 없는 개념은 애초에 근거가 못 된다.
                         # Poetry 0 · Pronephros 0 · Id 0 · Friends 1 · Palate 17
                         # ↔ Pain 1,103 · Psoriasis 310 · Methotrexate 303

# 근거등급 가중 (기본 0 = 순수 BM25 기준선). --boost 로 켠다.
EVIDENCE_W = {
    "Meta-Analysis": 0.30,
    "Systematic Review": 0.25,
    "Randomized Controlled Trial": 0.15,
    "Practice Guideline": 0.20,
}

# 관계어 사전 — 주장의 '방향'을 초록 본문과 잇는다. 활용형을 그대로 나열한다.
PREDICATE_KO = {
    # 한 개념에 활용형을 여러 키로 적어둔다. '높인다' 에는 '높이' 가 부분문자열로
    # 들어있지 않다(높/인/다) — 어간만 넣으면 조용히 안 걸린다.
    "낮추": "lower lowering reduce reduces reduced reduction decrease decreased",
    "낮춘": "lower lowering reduce reduces reduced reduction decrease decreased",
    "낮아": "lower lowering reduce reduced decrease decreased decline",
    "낮췄": "lower lowering reduce reduced decrease decreased",
    "감소": "reduce reduces reduced reduction decrease decreased lower lowering",
    "줄이": "reduce reduces reduced reduction decrease decreased",
    "높이": "increase increases increased raise elevated elevation higher",
    "높인": "increase increases increased raise elevated elevation higher risk",
    "높아": "increase increases increased elevated higher",
    "높였": "increase increases increased raise elevated",
    "늘리": "increase increases increased raise",
    "늘린": "increase increases increased raise",
    "줄인": "reduce reduces reduced reduction decrease decreased lower",
    "줄여": "reduce reduces reduced reduction decrease decreased lower",
    "악화": "worsen worsened aggravate exacerbate deterioration progression",
    "증가": "increase increases increased elevated elevation higher",
    "예방": "prevent prevents prevention preventive prophylaxis",
    "치료": "treatment therapy therapeutic treat treated",
    "개선": "improve improves improved improvement benefit",
    "억제": "inhibit inhibits inhibition suppress suppression",
    "완화": "alleviate relief ameliorate attenuate",
    "유발": "induce induces induced cause causes causing",
    "위험": "risk hazard odds",
    "효과": "effect effects efficacy effective effectiveness",
    "관련": "association associated relationship correlation",
    "사망": "mortality death survival fatal",
    "부작용": "adverse safety side toxicity",
    "발생": "incidence occurrence onset",
    "완치": "cure cured remission",
    "도움": "benefit beneficial improve",
}

_WS = re.compile(r"[\s,·/()\[\]{}\"'“”‘’]+")
_ASCII = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{2,}$")


def rrf(lists, k: int = FUSE_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. lists = [[doc, ...], ...] (순위순). 반환: [(doc, 점수)] 점수 내림차순.
    bench/run_nfcorpus.py 의 rrf 와 같은 식이다 — 같은 문서가 여러 목록에 있으면 1/(k+순위) 를 더한다."""
    sc: dict[int, float] = {}
    for lst in lists:
        for r, d in enumerate(lst, 1):
            sc[d] = sc.get(d, 0.0) + 1.0 / (k + r)
    return sorted(sc.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, index_path: str = INDEX, boost: float = 0.0, verbose: bool = True,
                 fuse: bool = True, llm=None, llm_mode: str = "off", llm_mesh: bool = True,
                 w_llm_claim: float = 1.0, w_llm_passage: float = 0.3,
                 reranker=None, rerank_top: int = RERANK_TOP, rerank_keywords: bool = False):
        assert llm_mode in LLM_MODES, llm_mode
        self.reranker = reranker            # medcpt_rerank.MedCPTReranker (또는 None) — 11/8, 상위 rerank_top 건의 순서만 바꾼다
        self.rerank_top = rerank_top
        self.rerank_keywords = rerank_keywords   # MeSH 키워드 질의로도 재순위할지. 11/9 dev: 이득 없음(−0.003) → 기본 False
        self.llm = llm                      # komed_llm.LLM (또는 None)
        self.llm_mode = llm_mode if llm is not None else "off"
        self.llm_mesh = llm_mesh            # LLM 프롬프트에 MeSH 정의·계층을 줄지 (E3 의 MeSH 있음/없음)
        self.w_llm_claim, self.w_llm_passage = w_llm_claim, w_llm_passage
        if not os.path.isfile(index_path):
            sys.exit(f"[에러] {index_path} 없음. 먼저: python build_pubmed_index.py")
        t0 = time.time()
        with open(index_path, "rb") as f:
            self.idx = pickle.load(f)
        stale = [p for p, m, s in self.idx["sources"]
                 if not os.path.isfile(p) or os.stat(p).st_size != s]
        if stale and verbose:
            print(f"  ⚠ 인덱스가 낡았습니다({', '.join(stale)}). "
                  f"python build_pubmed_index.py --force", file=sys.stderr)
        self.lookup = _load_lookup(KB)
        from ko_resolver import KoResolver
        self.ko = KoResolver()
        self.boost = boost
        self.postings = self.idx.get("postings")          # version 2 색인에만 있다
        self.fuse = bool(fuse) and self.postings is not None
        if fuse and self.postings is None and verbose:
            print("  ⚠ 색인에 postings 가 없어 전역 RRF 융합을 끕니다(게이트 단독). "
                  "python build_pubmed_index.py --force 로 색인을 다시 만드세요.", file=sys.stderr)
        self._fh = {}
        self._rcache: dict[str, dict | None] = {}   # span → 리졸브 결과. n-gram 이 반복돼 적중률이 높다
        self.rejected: list[tuple] = []             # 관문에서 걸러낸 개념 (진단용)
        if verbose:
            print(f"  [로드 {time.time()-t0:.1f}초 · 문서 {self.idx['n_docs']:,}건 · "
                  f"UI {len(self.idx['ui2docs']):,}개 · 융합 {'게이트+전역 RRF' if self.fuse else '없음(게이트 단독)'}]",
                  file=sys.stderr)

    # ---------------- 1) 개념 추출 ----------------
    def concepts(self, claim: str) -> list[dict]:
        self.rejected = []
        words = [w for w in _WS.split(claim.strip()) if w]
        taken = [False] * len(words)
        out = []
        for n in range(MAX_NGRAM, 0, -1):
            for i in range(len(words) - n + 1):
                if any(taken[i:i + n]):
                    continue
                span = " ".join(words[i:i + n])
                hit = self._resolve(span)
                if hit:
                    hit["span"] = span
                    out.append(hit)
                    for j in range(i, i + n):
                        taken[j] = True
        # 원문 등장 순서로 되돌린다
        out.sort(key=lambda h: claim.find(h["span"]))
        seen, uniq = set(), []
        for h in out:
            if h["ui"] not in seen:
                seen.add(h["ui"])
                uniq.append(h)
        return uniq

    def _resolve(self, span: str):
        if span in self._rcache:
            return self._rcache[span]
        hit = self._resolve_uncached(span)
        self._rcache[span] = hit
        return hit

    def _hangul_len(self, s: str) -> int:
        return sum(1 for c in s if "\uac00" <= c <= "\ud7a3")

    def _covered(self, ui: str) -> bool:
        return len(self.idx["ui2docs"].get(ui, ())) >= MIN_UI_DOCS

    def _resolve_uncached(self, span: str):
        r = self.ko.resolve(span, k=1)
        if r and float(r[0].get("score", 0)) >= KO_MIN_SCORE:
            h = dict(r[0])
            method = str(h.get("method", ""))
            short = self._hangul_len(span) < MIN_SPAN_FOR_FUZZY
            if short and "fuzzy" in method:
                self.rejected.append((span, h["name"], "짧은span+퍼지"))
            elif not self._covered(h["ui"]):
                self.rejected.append((span, h["name"], "코퍼스 문헌 부족"))
            else:
                h["track"] = "ko"
                return h
        if _ASCII.match(span):                      # 영문 약어·용어 (BMI, statin …)
            v = self.lookup.get(normalize(span))
            if v and self._covered(v["ui"]):
                return {"ui": v["ui"], "name": v["name"], "score": 100.0,
                        "method": "exact", "track": "en"}
        return None

    # ---------------- 2) 후보 좁히기 ----------------
    def _candidates(self, uis: list[str]):
        u2d, n = self.idx["ui2docs"], self.idx["n_docs"]
        usable = [(u, u2d[u]) for u in uis
                  if u in u2d and len(u2d[u]) <= MAX_UI_RATIO * n]
        if not usable:
            return [], "없음", []
        sets = [(u, set(p)) for u, p in usable]
        if len(sets) >= 2:
            inter = set.intersection(*[s for _, s in sets])
            if len(inter) >= MIN_INTERSECT:
                return sorted(inter), "교집합(전체)", [u for u, _ in sets]
            best, pair = None, None
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    cur = sets[i][1] & sets[j][1]
                    if len(cur) >= MIN_INTERSECT and (best is None or len(cur) < len(best)):
                        best, pair = cur, (sets[i][0], sets[j][0])
            if best:
                return sorted(best), "교집합(2축)", list(pair)
        union = set()
        for _, s in sets:
            union |= s
        return sorted(union)[:MAX_CANDS], "합집합", [u for u, _ in sets]

    # ---------------- 3) 질의어 ----------------
    def _query_terms(self, claim: str, concepts: list[dict]) -> dict[str, float]:
        df, n = self.idx["df"], self.idx["n_docs"]
        q: dict[str, float] = {}

        def add(t: str, w: float):
            if w > q.get(t, 0.0):
                q[t] = w

        for c in concepts:
            surfs = self.idx["ui2terms"].get(c["ui"]) or [c["name"]]
            for t in tokenize(surfs[0]):            # 0번 = 표목명
                add(t, W_NAME)
            for surf in surfs[1:]:                  # 별칭은 변별력 있는 것만
                for t in tokenize(surf):
                    if df.get(t, 0) <= ALIAS_MAX_DF * n:
                        add(t, W_ALIAS)
        for key, eng in PREDICATE_KO.items():
            if key in claim:
                for t in tokenize(eng):
                    add(t, W_PRED)
        for w in _WS.split(claim):
            if _ASCII.match(w):
                for t in tokenize(w):
                    add(t, W_ASCII)
        return q

    # ---------------- 4) 랭킹 ----------------
    def _read(self, i: int) -> dict:
        fi = self.idx["offs_file"][i]
        path = self.idx["files"][fi]
        fh = self._fh.get(fi)
        if fh is None:
            fh = self._fh[fi] = open(path, "rb")
        fh.seek(self.idx["offs_byte"][i])
        return json.loads(fh.readline().decode("utf-8"))

    def _global_top(self, qterms: dict[str, float], top: int = FUSE_TOP) -> list[tuple[int, float]]:
        """같은 질의어로 코퍼스 전체를 BM25 채점한 상위 top. bm25_util.bm25 와 같은 식을 포스팅 위에서 누적한다.
        반환: [(문서 id, 점수)] 점수 내림차순. --boost 가 켜져 있으면 상위 3·top 만 문서를 읽어 가중을 적용한다."""
        df, n, avgdl, dl = (self.idx["df"], self.idx["n_docs"], self.idx["avgdl"], self.idx["dl"])
        acc: dict[int, float] = {}
        for t, w in qterms.items():
            pl = self.postings.get(t)
            if not pl:
                continue
            wi = w * idf(df.get(t, 0), n)
            docs, tfs = pl
            for i, f in zip(docs, tfs):
                acc[i] = acc.get(i, 0.0) + wi * f * (K1 + 1.0) / (f + K1 * (1.0 - B + B * dl[i] / avgdl))
        ranked = sorted(acc.items(), key=lambda x: -x[1])
        if self.boost:
            head = []
            for i, sc in ranked[:3 * top]:
                d = self._read(i)
                w = max((EVIDENCE_W.get(pt, 0.0) for pt in (d.get("pubtypes") or [])), default=0.0)
                head.append((i, sc * (1.0 + w * self.boost)))
            head.sort(key=lambda x: -x[1])
            return head[:top]
        return ranked[:top]

    def _hit(self, r: int, s: float, d: dict, src: str, rrf_score: float | None) -> dict:
        return {
            "rank": r, "score": round(s, 3), "pmid": d.get("pmid"),
            "year": d.get("year"), "journal": d.get("journal"),
            "stratum": d.get("stratum", "BASE"),
            "pubtypes": [p for p in (d.get("pubtypes") or [])
                         if p in EVIDENCE_W or "Review" in p][:3],
            "title": d.get("title"),
            "abstract": d.get("abstract"),
            "src": src,                                        # gate · global · both
            "rrf": round(rrf_score, 5) if rrf_score is not None else None,
        }

    # ---------------- 5) 재순위 (11/8, 선택) ----------------
    def _pool(self, k: int) -> int:
        return max(k, self.rerank_top) if self.reranker is not None else k

    def _rerank(self, claim: str, cons: list[dict], llm_info: dict | None, hits: list[dict], k: int):
        """상위 rerank_top 건을 MedCPT 점수로 다시 매기고 k 건만 남긴다. 평가기 신호와는 무관하다."""
        if self.reranker is None or not hits:
            return hits[:k], None
        import medcpt_rerank as mr_
        q, qsrc = mr_.rerank_query(claim, cons, llm_info, PREDICATE_KO)
        if not q:
            return hits[:k], {"model": self.reranker.name, "query": "", "query_src": qsrc, "n": 0}
        if qsrc == "mesh_keywords" and not self.rerank_keywords:
            # 11/9 NFCorpus dev: MeSH 표목명 키워드 질의로 재순위하면 nDCG@10 −0.003 [−0.018, +0.013](상위 30)로 이득이 없었다.
            # 영어 문장 질의(LLM 번역 또는 영어 주장)가 있을 때만 재순위한다.
            return hits[:k], {"model": self.reranker.name, "query": q, "query_src": qsrc, "n": 0,
                              "skipped": "영어 문장 질의가 없어 재순위하지 않음(11/9 dev 결과)"}
        head, tail = hits[:self.rerank_top], hits[self.rerank_top:]
        sc = self.reranker.score(q, [mr_.doc_text(h.get("title"), h.get("abstract")) for h in head])
        for h, x in zip(head, sc):
            h["ce"] = round(float(x), 4)
        head = mr_.rerank(head, sc)
        out = (head + tail)[:k]
        for r, h in enumerate(out, 1):
            h["rank"] = r
        return out, {"model": self.reranker.name, "query": q, "query_src": qsrc, "n": len(head)}

    def search(self, claim: str, k: int = 10) -> dict:
        t0 = time.time()
        cons = self.concepts(claim)
        if not cons:
            return {"claim": claim, "concepts": [], "hits": [],
                    "reason": "MeSH 리졸브 실패 — 의학 개념을 못 찾았습니다",
                    "mode": "없음", "n_cands": 0, "query_terms": [], "ms": 0}

        cands, mode, used = self._candidates([c["ui"] for c in cons])
        qterms = self._query_terms(claim, cons)

        # 11/8 LLM 질의 확장 — 질의어만 보탠다. 개념·후보·평가기 신호는 위의 원래 값 그대로.
        llm_info, lq = None, {}
        if self.llm_mode != "off" and qterms:
            import llm_query
            exp = llm_query.expand(self.llm, claim, cons, self.llm_mesh)
            lq = llm_query.terms(exp, self.w_llm_claim, self.w_llm_passage)
            llm_info = {"mode": self.llm_mode, **exp, "n_terms": len(lq)}
        q_merged = {**qterms, **{t: w for t, w in lq.items() if w > qterms.get(t, 0.0)}} if lq else qterms
        q_glob = q_merged if self.llm_mode in ("global", "all") else qterms   # 전역 다리의 질의어
        q_gate = q_merged if self.llm_mode == "all" else qterms                # 게이트 후보 순위의 질의어

        if not cands or not qterms:
            out = {"claim": claim, "concepts": cons, "hits": [],
                   "reason": "후보 문헌 없음" if not cands else "질의어 없음",
                   "mode": mode, "n_cands": len(cands),
                   "query_terms": sorted(qterms.items()), "ms": (time.time() - t0) * 1000}
            if llm_info:
                out["llm"] = llm_info
            if not cands and qterms and self.fuse:
                # 게이트 후보가 없을 때의 폴백: 전역 순위만 제시한다. 평가기 신호는 그대로(점수 붙은 후보 0건)
                # 두어 10/23 임계값대로 incorrect 로 가게 한다 — 근거가 부족하면 멈추는 설계.
                glob = self._global_top(q_merged, self._pool(k))
                out["hits"] = [self._hit(r, sc, self._read(i), "global", None)
                               for r, (i, sc) in enumerate(glob, 1)]
                out["hits"], out["rerank"] = self._rerank(claim, cons, llm_info, out["hits"], k)
                out["reason"] = "후보 문헌 없음 — 전역 순위만 제시(폴백)"
                out["fusion"] = {"on": True, "k": FUSE_K, "top": FUSE_TOP, "n_gate": 0, "n_global": len(glob), "n_both": 0}
                out["signals"] = {"top_score": 0.0, "gap_1_2": None, "n_scored": 0, "mode": mode, "n_concepts": len(cons)}
                out["ms"] = (time.time() - t0) * 1000
            return out

        df, n, avgdl, dl = (self.idx["df"], self.idx["n_docs"],
                            self.idx["avgdl"], self.idx["dl"])
        scored = []          # 평가기 신호용: 원래 질의어 점수 (11/8 이전과 같다)
        gate_rank = []       # 게이트 후보의 순위용: llm_mode=all 이면 합친 질의어 점수
        for i in cands:
            d = self._read(i)
            tf = collections.Counter(t for t in doc_tokens(d) if t in q_gate)
            if not any(t in qterms for t in tf):     # 원래 질의어가 하나도 없으면 11/8 이전처럼 뺀다
                continue
            w = 0.0
            if self.boost:
                w = max((EVIDENCE_W.get(pt, 0.0) for pt in (d.get("pubtypes") or [])),
                        default=0.0)
            s = bm25(qterms, tf, dl[i], df, n, avgdl) * (1.0 + w * self.boost)
            scored.append((s, i, d))
            if q_gate is not qterms:
                gate_rank.append((bm25(q_gate, tf, dl[i], df, n, avgdl) * (1.0 + w * self.boost), i, d))
        scored.sort(key=lambda x: -x[0])
        if q_gate is qterms:
            gate_rank = scored
        else:
            gate_rank.sort(key=lambda x: -x[0])

        fusion = {"on": False}
        if self.fuse:
            # 게이트 순위(상위 FUSE_TOP) 와 전역 BM25 순위(상위 FUSE_TOP) 를 RRF 로 합친다.
            # llm_mode=leg 이면 LLM 질의어만의 전역 순위를 셋째 다리로 더한다.
            gate_docs = {i: (s, d) for s, i, d in gate_rank[:FUSE_TOP]}
            glob = self._global_top(q_glob, FUSE_TOP)
            gscore = dict(glob)
            lists = [list(gate_docs), [i for i, _ in glob]]
            lglob = []
            if self.llm_mode == "leg" and lq:
                lglob = self._global_top(lq, FUSE_TOP)
                lists.append([i for i, _ in lglob])
            lscore = dict(lglob)
            fused = rrf(lists, FUSE_K)
            hits = []
            for r, (i, rs) in enumerate(fused[:self._pool(k)], 1):
                if i in gate_docs:
                    s, d = gate_docs[i]
                    src = "both" if i in gscore else "gate"
                elif i in gscore:
                    s, d, src = gscore[i], self._read(i), "global"
                else:
                    s, d, src = lscore[i], self._read(i), "llm"
                hits.append(self._hit(r, s, d, src, rs))
            fusion = {"on": True, "k": FUSE_K, "top": FUSE_TOP, "n_gate": len(gate_docs), "n_global": len(glob),
                      "n_both": sum(1 for i in gate_docs if i in gscore),
                      "global_top_score": round(glob[0][1], 3) if glob else 0.0}
            if lglob:
                fusion["n_llm"] = len(lglob)
        else:
            hits = [self._hit(r, s, d, "gate", None) for r, (s, i, d) in enumerate(gate_rank[:self._pool(k)], 1)]
        hits, rr_info = self._rerank(claim, cons, llm_info, hits, k)

        top = scored[0][0] if scored else 0.0
        return {
            "claim": claim, "concepts": cons, "hits": hits, "mode": mode,
            "used_uis": used, "n_cands": len(cands), "n_scored": len(scored),
            "fusion": fusion,
            "query_terms": sorted(qterms.items(), key=lambda x: (-x[1], x[0])),
            "llm": llm_info,        # LLM 질의 확장의 출력(영어 주장·의사 문서·검색어). 꺼져 있으면 None
            "rerank": rr_info,      # 재순위 정보(모델·영어 질의·질의 출처·건수). 꺼져 있으면 None
            # ↓ 다음 단계(Retrieval Evaluator)가 쓸 원신호. 임계값은 여기서 정하지 않는다.
            #   llm_mode 와 무관하게 원래 질의어로 계산한다(10/23 임계값 호환).
            "signals": {
                "top_score": round(top, 3),
                "gap_1_2": round(top - scored[1][0], 3) if len(scored) > 1 else None,
                "n_scored": len(scored),
                "mode": mode,
                "n_concepts": len(cons),
            },
            "ms": (time.time() - t0) * 1000,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claim", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--boost", type=float, default=0.0,
                    help="근거등급 가중 (0=순수 BM25 기준선, 1.0=EVIDENCE_W 그대로)")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-fuse", action="store_true", help="전역 RRF 융합 없이 게이트 단독 (E3 의 gate 행)")
    ap.add_argument("--llm", default=None, help="LLM 질의 확장 공급자[:모델] (gemini · anthropic · mock). 없으면 끔")
    ap.add_argument("--llm-mode", default="global", choices=LLM_MODES[1:], help="--llm 을 줄 때의 모드")
    ap.add_argument("--llm-no-mesh", action="store_true", help="LLM 프롬프트에 MeSH 정의·계층을 넣지 않음")
    ap.add_argument("--rerank", default=None, choices=("medcpt", "fake"), help="상위 --rerank-top 건을 재순위 (transformers 필요)")
    ap.add_argument("--rerank-top", type=int, default=RERANK_TOP)
    ap.add_argument("--rerank-keywords", action="store_true", help="영어 문장 질의가 없을 때 MeSH 키워드 질의로도 재순위 (11/9 dev: 이득 없음)")
    a = ap.parse_args()

    llm = None
    if a.llm:
        from komed_llm import LLM
        import llm_query
        prov, _, model = a.llm.partition(":")
        llm = LLM(prov, model or None, mock_fn=llm_query.mock_fn if prov == "mock" else None)
    rr = None
    if a.rerank:
        import medcpt_rerank
        rr = medcpt_rerank.make_reranker(a.rerank)
    r = Retriever(boost=a.boost, fuse=not a.no_fuse, llm=llm,
                  llm_mode=a.llm_mode if llm else "off", llm_mesh=not a.llm_no_mesh,
                  reranker=rr, rerank_top=a.rerank_top, rerank_keywords=a.rerank_keywords)
    res = r.search(" ".join(a.claim), k=a.k)

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 78)
    print(f"주장: {res['claim']}")
    print("=" * 78)
    print("개념: " + (" / ".join(f"{c['span']}→{c['name']}({c['ui']},{c['method']})"
                                 for c in res["concepts"]) or "(없음)"))
    if res["hits"]:
        fu = res.get("fusion") or {}
        print(f"후보: {res['n_cands']:,}건 [{res['mode']}] · 점수산출 {res.get('n_scored', 0):,}건 "
              f"· {res['ms']:.0f}ms"
              + (f" · 융합 게이트 {fu.get('n_gate')} + 전역 {fu.get('n_global')} (겹침 {fu.get('n_both')})" if fu.get("on") else " · 융합 없음"))
    if a.explain:
        qt = res["query_terms"]
        print(f"질의어({len(qt)}): " + "  ".join(f"{t}:{w:.2f}" for t, w in qt))
        print(f"신호: {res.get('signals')}")
        if res.get("llm"):
            L = res["llm"]
            print(f"LLM({L['model']}, {L['mode']}, MeSH {'있음' if L['use_mesh'] else '없음'}): {L['english_claim']}")
            print(f"   검색어: {', '.join(L['keywords'])}")
            print(f"   의사 문서: {L['passage'][:300]}")
        if res.get("rerank"):
            R = res["rerank"]
            print(f"재순위({R['model']}, 상위 {R['n']}건, 질의 출처 {R['query_src']}): {R['query']}"
                  + (f"  [{R['skipped']}]" if R.get("skipped") else ""))
    if not res["hits"]:
        print(f"\n결과 없음 — {res.get('reason', '')}")
        return
    for h in res["hits"]:
        pt = ("[" + ", ".join(h["pubtypes"]) + "] ") if h["pubtypes"] else ""
        print(f"\n{h['rank']}. {h['score']:>7.2f}{'  CE ' + format(h['ce'], '.2f') if 'ce' in h else ''}  {h['year']}  {pt}"
              f"PMID {h['pmid']}  <{h['stratum']}>  [{h.get('src', 'gate')}]")
        print(f"   {h['title']}")
        print(f"   {(h['abstract'] or '')[:230]}…")


if __name__ == "__main__":
    main()
