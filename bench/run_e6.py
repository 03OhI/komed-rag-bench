#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_e6.py — E6 CLI entry point. 2026-10-23부터 실행은 run_e6_v2.main 으로 위임한다.
현행: closed/stdrag/bm25rules/gatefree/gaterules/ours, 4범주, 독립 gold 등록, 오류 분리.
아래 v1 함수는 v2가 사용하는 검색·파이프라인 어댑터 및 이전 구현 추적용이다.
현행 명령은 README_bench.md 맨 위와 E1_E6_재설계_20261023.md 참고.

이하 2026-09-26 v1 기록:

    closed   LLM-only — 문헌 없이 판정 (run_verdict.py 의 CLOSED 프롬프트)
    stdrag   Standard RAG — MeSH 게이트 없이 전역 BM25 상위 8편 → 같은 LLM 자유 판정 (run_verdict.py 의 FREE 프롬프트)
             질의어는 우리 리졸버가 만든 것과 같다(retrieve._query_terms) — 차이는 게이트·검색 평가기·판독·규칙이 없다는 것뿐
    ours     KoMedRAG.verify_claim 그대로 (MeSH 게이트 → 검색 평가기 → 판독 → 고정 규칙), 뉴스 폴백 끔

    python run_e6.py --llm mock --limit 3 --out /tmp/e6test          # 키 없이 구조 확인
    python run_e6.py --llm gemini:gemini-3.1-pro-preview             # 본실행 → preds/e6_<cond>_<llm>.jsonl
    python run_e6.py --llm ... --repeat 3 --no-cache                 # 3회 일치율
    python report.py                                                 # E6 표(truthfulness · 정확도 · 근거 없음 거부율 · 뒤집힘)

정답 = 사전 등록 기대 범주 (eval_claims.tsv 의 expected, claim_expected.tsv 가 있으면 덮어씀) → 3라벨 대응 (2026-09-26 사전 등록)
    지지 → SUPPORT · 반박 → REFUTE · 범위 초과(과장, 띄어쓰기 있음) → SUPPORT (E1 의 ours 대응과 같다)
    범위 밖 · 근거 없음 · 범위초과(띄어쓰기 없음 = C15, 의학 개념 없음) → NEI
    부분 · 논란 · 빈칸 → 채점 제외(주장별 표에만)
동결 파일(retrieve.py · bm25_util.py · 색인)은 읽기만 한다. 전역 BM25 는 이 파일 안에서 한 번에 계산해 캐시한다.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import verdict  # noqa: E402
from bm25_util import bm25, doc_tokens  # noqa: E402
from llm_client import make_llm  # noqa: E402
from run_verdict import CAT2LABEL, CLOSED_SYSTEM, FREE_SYSTEM, FREE_TOOL, _norm  # noqa: E402

GOLD_MAP = {"지지": "SUPPORT", "반박": "REFUTE", "범위 초과": "SUPPORT",
            "범위 밖": "NEI", "근거 없음": "NEI", "근거없음": "NEI", "범위초과": "NEI"}
EXCLUDED = {"부분", "논란", ""}
READING_LINE = "각 문헌에 대해 relevant/stance/key_sentence/coding 을, 주장에 대해 claim_coding 을, 그리고 explanation_ko 를 submit_reading 으로 제출하라."


def load_claims(limit: int = 0) -> list[dict]:
    rows = list(csv.DictReader(open(os.path.join(MESH, "eval_claims.tsv"), encoding="utf-8"), delimiter="\t"))
    ce = os.path.join(MESH, "claim_expected.tsv")
    if os.path.isfile(ce):
        for e in csv.DictReader(open(ce, encoding="utf-8"), delimiter="\t"):
            for r in rows:
                if r["id"] == e["id"] and (e.get("expected") or "").strip():
                    r["expected"] = e["expected"].strip()
    out = []
    for r in rows:
        exp = (r.get("expected") or "").strip()
        gold = None if exp in EXCLUDED else GOLD_MAP.get(exp)
        out.append({"id": r["id"], "claim": r["claim"], "expected_raw": exp, "gold": gold})
    return out[:limit] if limit else out


def doc_fields(d: dict) -> dict:
    return {"pmid": d.get("pmid"), "year": d.get("year"), "journal": d.get("journal"),
            "pubtypes": (d.get("pubtypes") or [])[:3], "title": d.get("title"), "abstract": d.get("abstract")}


def stdrag_hits(retr, claims: list[dict], k: int, cache_path: str) -> dict:
    """게이트 없는 전역 BM25 상위 k — 색인 전체를 한 번 훑어 모든 주장을 같이 채점한다(캐시)."""
    cache = {}
    if os.path.isfile(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    todo = [c for c in claims if c["id"] not in cache]
    if not todo:
        return cache
    qmap = {}
    for c in todo:
        cons = retr.concepts(c["claim"])
        qmap[c["id"]] = retr._query_terms(c["claim"], cons)
    union = set()
    for q in qmap.values():
        union |= set(q)
    idx = retr.idx
    df, n, avgdl, dl = idx["df"], idx["n_docs"], idx["avgdl"], idx["dl"]
    heaps = {cid: [] for cid in qmap}
    t0 = time.time()
    if union:
        for i in range(n):
            d = retr._read(i)
            cnt = Counter(t for t in doc_tokens(d) if t in union)
            if not cnt:
                continue
            for cid, q in qmap.items():
                tf = {t: c for t, c in cnt.items() if t in q}
                if not tf:
                    continue
                s = bm25(q, tf, dl[i], df, n, avgdl)
                h = heaps[cid]
                if len(h) < k:
                    heapq.heappush(h, (s, i))
                elif s > h[0][0]:
                    heapq.heapreplace(h, (s, i))
    for cid, h in heaps.items():
        top = sorted(h, key=lambda x: -x[0])
        cache[cid] = [{"rank": r, "score": round(s, 3), **doc_fields(retr._read(i))} for r, (s, i) in enumerate(top, 1)]
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  전역 BM25: 주장 {len(todo)}건 · 질의어 {len(union)}개 · 문헌 {n:,}건 훑음 · {time.time() - t0:.0f}초 → {os.path.basename(cache_path)}")
    return cache


def make_ours(retr, llm, k: int):
    """KoMedRAG.verify_claim 을 그대로 쓰되 LLM 만 벤치 클라이언트로 (문체 모델·뉴스 폴백은 부르지 않는다)."""
    from pipeline import KoMedRAG
    from retrieval_evaluator import load_thresholds
    sysm = KoMedRAG.__new__(KoMedRAG)
    sysm.retriever, sysm.k, sysm.use_news = retr, k, False
    sysm.thresholds, sysm.rules, sysm.llm = load_thresholds(), verdict.load_rules(), llm
    sysm.style, sysm.boost = None, getattr(retr, "boost", 0.0)
    return sysm


def run_one(c: dict, cond: str, llm, rules: dict, hits_std: dict, sysm) -> dict:
    t0 = time.time()
    rec = {"id": c["id"], "dataset": "e6", "cond": cond, "llm": llm.label, "claim": c["claim"],
           "expected_raw": c["expected_raw"], "gold": c["gold"]}
    try:
        if cond == "ours":
            r = sysm.verify_claim(c["claim"])
            f = r["fact"]
            ev = [e for e in f.get("evidence", []) if e.get("relevant")]
            rec.update({"pred": CAT2LABEL[f["category"]], "category": f["category"], "nei_reason": f.get("nei_reason"),
                        "evaluator": r["evaluator"]["verdict"], "n_hits": len(r["retrieval"]["hits"]),
                        "n_relevant": f.get("n_relevant", 0), "exceeded": [x["dim"] for x in f.get("exceeded", [])],
                        "key_sentences": [e["key_sentence"] for e in ev], "verbatim": [e["verbatim"] for e in ev],
                        "pmids": [e.get("pmid") for e in ev], "explanation_ko": f.get("explanation_ko", "")})
        elif cond == "stdrag":
            hits = hits_std.get(c["id"], [])
            prompt = verdict._prompt(c["claim"], hits, rules).replace(READING_LINE, "label / key_sentence / reason 을 submit_verdict 로 제출하라.")
            out = llm.call(FREE_SYSTEM, prompt, FREE_TOOL)
            lab = str(out.get("label", "")).upper()
            ks = out.get("key_sentence", "") or ""
            allabs = " ".join((h.get("abstract") or "") + " " + (h.get("title") or "") for h in hits)
            rec.update({"pred": lab if lab in ("SUPPORT", "REFUTE", "NEI") else "NEI", "n_hits": len(hits),
                        "key_sentences": [ks], "verbatim": [bool(ks) and _norm(ks) in _norm(allabs)],
                        "pmids": [h.get("pmid") for h in hits], "reason": out.get("reason", "")})
        elif cond == "closed":
            out = llm.call(CLOSED_SYSTEM, f"[주장] {c['claim']}\n\nlabel / reason 을 submit_verdict 로 제출하라.", FREE_TOOL)
            lab = str(out.get("label", "")).upper()
            rec.update({"pred": lab if lab in ("SUPPORT", "REFUTE", "NEI") else "NEI", "reason": out.get("reason", "")})
        else:
            raise ValueError(cond)
    except Exception as e:  # noqa: BLE001
        rec.update({"pred": "NEI", "error": str(e)[:500]})
    rec["sec"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="provider[:model]  예) gemini:gemini-3.1-pro-preview · mock")
    ap.add_argument("--cond", nargs="+", default=["closed", "stdrag", "ours"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k-std", type=int, default=8, help="Standard RAG 문헌 수 (E6 설계: BM25 상위 8)")
    ap.add_argument("--k-ours", type=int, default=10, help="우리 파이프라인 검색 상위 k (pipeline 기본 10)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "preds"))
    ap.add_argument("--hits-cache", default=os.path.join(HERE, "e6_stdrag_hits.json"), help="Standard RAG 전역 BM25 결과 캐시")
    a = ap.parse_args()

    claims = load_claims(a.limit)
    n_scored = sum(1 for c in claims if c["gold"])
    print(f"[E6] 주장 {len(claims)}건 (채점 대상 {n_scored} · 제외 {len(claims) - n_scored}: 부분·논란·빈칸)  llm={a.llm}  cond={a.cond}")
    os.makedirs(a.out, exist_ok=True)
    rules = verdict.load_rules()
    tag = a.llm.replace(":", "-").replace("/", "_")

    from retrieve import Retriever
    cwd = os.getcwd()
    os.chdir(MESH)                                   # 색인의 문서 파일 경로가 mesh 기준 상대 경로
    try:
        retr = Retriever(verbose=False)
        hits_std = stdrag_hits(retr, claims, a.k_std, a.hits_cache) if "stdrag" in a.cond else {}
        for cond in a.cond:
            for r in range(1, a.repeat + 1):
                suffix = f"_r{r}" if a.repeat > 1 else ""
                path = os.path.join(a.out, f"e6_{cond}_{tag}{suffix}.jsonl")
                done = set()
                if os.path.isfile(path):
                    for line in open(path, encoding="utf-8"):
                        try:
                            rec0 = json.loads(line)
                            if not rec0.get("error"):
                                done.add(rec0["id"])
                        except Exception:
                            pass
                todo = [c for c in claims if c["id"] not in done]
                print(f"  {cond} run{r}: {len(todo)} to do ({len(done)} done) → {os.path.basename(path)}")
                if not todo:
                    continue
                llm = make_llm(a.llm, cache=not a.no_cache)
                sysm = make_ours(retr, llm, a.k_ours) if cond == "ours" else None
                n_err = 0
                with open(path, "a", encoding="utf-8") as f:
                    for c in todo:
                        rec = run_one(c, cond, llm, rules, hits_std, sysm)
                        rec["run"] = r
                        n_err += bool(rec.get("error"))
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                if n_err:
                    print(f"  ⚠ {cond}: 오류 {n_err}건 (pred=NEI, 'error' 필드) — 같은 명령을 다시 돌리면 오류분만 재시도된다.")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    # v1 helpers retained for traceability. CLI uses the prospective four-label design.
    from run_e6_v2 import main as main_v2
    main_v2()
