#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerank_nfcorpus.py — E3: 시스템 게이트+RRF 순위의 상위 N건을 MedCPT cross-encoder 로 재순위해 NFCorpus 에서 잰다 (11/8)

입력 (bench/ 기준, run_system_on_nfcorpus.py 가 만든 것)
    preds_v2/ir_nfcorpus[_dev]_system_gate_rrf.trec      시스템 게이트+전역 RRF 순위 (재순위의 바탕)
    preds_v2/ir_nfcorpus[_dev]_system_concepts.jsonl     질의별 MeSH 개념 이름 (mesh 질의 변형용)
    preds_v2/ir_nfcorpus[_dev]_bm25_queries.jsonl        하네스 BM25 질의별 지표 (Δ 의 기준)

질의 변형 (--variants)
    raw   NFCorpus 질의 원문(영어). 한국어 기사에서는 LLM 영어 번역이 있을 때에 해당
    mesh  리졸버가 찾은 MeSH 표목명만 이은 키워드 질의. 한국어 기사에서 LLM 없이(비용 0) 갈 때에 해당. 개념이 없으면 원문으로

사용
    python rerank_nfcorpus.py --split dev                         # 코랩 GPU 권장 (MedCPT 를 처음에 내려받음)
    python rerank_nfcorpus.py --split dev --scorer fake           # 모델 없이 경로만 확인
    python rerank_nfcorpus.py --split test --variants raw --final # dev 에서 고정한 뒤 한 번만
출력
    report_ir_nfcorpus[_dev]_rerank_<scorer>_top<N>.md · preds_v2/ir_nfcorpus[_dev]_system_gate_rrf_<scorer>_top<N>_<변형>.trec
    rerank_cache_<scorer>.jsonl  (질의·문서 쌍 점수. 질의마다 fsync. 같은 명령을 다시 돌리면 매긴 쌍은 건너뛰고 이어서 돈다)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_nfcorpus as rn                      # noqa: E402  지표·부트스트랩은 하네스와 같은 함수
from medcpt_rerank import doc_text, make_reranker, rerank   # noqa: E402


def read_trec(path: str) -> dict[str, list[str]]:
    run: dict[str, list[tuple[int, str]]] = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            run.setdefault(p[0], []).append((int(p[3]), p[2]))
    return {q: [d for _, d in sorted(v)] for q, v in run.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--beir", default=os.path.join(HERE, "data", "nfcorpus"))
    ap.add_argument("--preds", default=os.path.join(HERE, "preds_v2"))
    ap.add_argument("--variants", default="raw,mesh")
    ap.add_argument("--top", type=int, default=100, help="재순위할 상위 건수")
    ap.add_argument("--scorer", default="medcpt", choices=("medcpt", "fake"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--final", action="store_true", help="test 분할은 dev 에서 고정한 뒤 한 번만")
    ap.add_argument("--cache", default=None, help="점수 캐시 경로 (기본 bench/rerank_cache_<scorer>.jsonl). 구글 드라이브에 두면 세션이 끊겨도 이어서 돈다")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        sys.exit("[중단] 시험 분할은 dev 에서 변형·N 을 고정한 뒤 --final 로 한 번만 (11/8 규칙)")

    name = "nfcorpus" if a.split == "test" else f"nfcorpus_{a.split}"
    corpus, queries, qrels = rn.load_beir(a.beir, a.split)
    base = read_trec(os.path.join(a.preds, f"ir_{name}_system_gate_rrf.trec"))
    cpath = os.path.join(a.preds, f"ir_{name}_system_concepts.jsonl")
    cmap = {}
    if os.path.isfile(cpath):
        for line in open(cpath, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cmap[r["id"]] = r.get("concepts") or []
    bm = {}
    for line in open(os.path.join(a.preds, f"ir_{name}_bm25_queries.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            bm[r["id"]] = r
    qids = list(queries)                 # 시스템 표(run_system_on_nfcorpus)와 같은 분모: 순위가 없는 질의도 0점으로 넣는다
    n_empty = sum(1 for q in qids if not base.get(q))
    base = {q: base.get(q, []) for q in qids}

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if "mesh" in variants and not cmap:
        sys.exit(f"[중단] {cpath} 가 없다 — run_system_on_nfcorpus.py --split {a.split} 를 먼저 돌린다")

    def qtext(qid, v):
        if v == "mesh" and cmap.get(qid):
            return " ".join(cmap[qid]), "mesh"
        return queries[qid], "raw"

    scorer = make_reranker(a.scorer, **({"device": a.device, "max_len": a.max_len, "batch": a.batch} if a.scorer == "medcpt" else {}))
    cache_path = a.cache or os.path.join(HERE, f"rerank_cache_{a.scorer}.jsonl")
    cache: dict[str, float] = {}
    if os.path.isfile(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(line); cache[r["k"]] = r["s"]
            except Exception:
                pass
    key = lambda q, d: hashlib.sha1(f"{scorer.name}\t{a.max_len}\t{q}\t{d}".encode("utf-8")).hexdigest()

    runs, srcs = {}, {}
    t0, done, total = time.time(), 0, len(qids) * len(variants)
    if os.path.isfile(cache_path) and os.path.getsize(cache_path):
        with open(cache_path, "rb") as f:          # 끊긴 세션이 줄 중간에서 멈췄으면 줄을 바꿔 둔다
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                open(cache_path, "a", encoding="utf-8").write("\n")
    print(f"[캐시] {cache_path} · 이미 매긴 쌍 {len(cache):,}", flush=True)
    with open(cache_path, "a", encoding="utf-8") as cf:
        for v in variants:
            run, src = {}, {"mesh": 0, "raw": 0}
            for qid in qids:
                q, s = qtext(qid, v)
                src[s] += 1
                head, tail = base[qid][:a.top], base[qid][a.top:]
                todo = [d for d in head if key(q, d) not in cache]
                if todo:
                    sc = scorer.score(q, [doc_text(corpus[d]["title"], corpus[d]["text"]) for d in todo])
                    for d, x in zip(todo, sc):
                        k = key(q, d); cache[k] = x
                        cf.write(json.dumps({"k": k, "s": x}) + "\n")
                    cf.flush()
                    try:
                        os.fsync(cf.fileno())           # 질의마다 디스크(드라이브)에 확정 — 세션이 끊겨도 여기까지는 남는다
                    except OSError:
                        pass                            # fsync 를 지원하지 않는 파일 시스템이면 flush 만
                run[qid] = rerank(head, [cache[key(q, d)] for d in head]) + tail
                done += 1
                if done % 25 == 0 or done == total:
                    el = time.time() - t0
                    print(f"  [{done}/{total}] {el:.0f}초 · 남은 시간 약 {el / done * (total - done):.0f}초", flush=True)
            runs[v], srcs[v] = run, src

    def m(run):
        return ([rn.ndcg(run[q], qrels[q], 10) for q in qids], [rn.recall(run[q], qrels[q], 100) for q in qids],
                [rn.ndcg(run[q], qrels[q], 3) for q in qids])

    b_nd = [bm[q]["ndcg10"] for q in qids]
    b_rc = [bm[q]["recall100"] for q in qids]
    s_nd, s_rc, s_n3 = m(base)
    mean = statistics.mean
    lines = [f"# E3 재순위 — {name} (질의 {len(qids)}, 그중 시스템 순위가 빈 질의 {n_empty} · 재순위 상위 {a.top} · {scorer.name} · max_len {a.max_len})", "",
             f"하네스 BM25: nDCG@10 {mean(b_nd):.3f} · Recall@100 {mean(b_rc):.3f}", "",
             "| 조건 | nDCG@10 [95% CI] | nDCG@3 | Recall@100 | Δ nDCG@10 vs BM25 [95% CI] | Δ nDCG@10 vs 시스템 게이트+RRF [95% CI] | Δ nDCG@3 vs 시스템 [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    lo, hi = rn.boot_ci(s_nd); dlo, dhi = rn.boot_diff(b_nd, s_nd)
    lines.append(f"| 시스템 게이트+RRF (재순위 없음) | {mean(s_nd):.3f} [{lo:.3f}, {hi:.3f}] | {mean(s_n3):.3f} | {mean(s_rc):.3f} | "
                 f"{mean(s_nd)-mean(b_nd):+.3f} [{dlo:+.3f}, {dhi:+.3f}] | — | — |")
    for v in variants:
        nd, rc, n3 = m(runs[v])
        lo, hi = rn.boot_ci(nd); dlo, dhi = rn.boot_diff(b_nd, nd)
        slo, shi = rn.boot_diff(s_nd, nd); tlo, thi = rn.boot_diff(s_n3, n3)
        lines.append(f"| + 재순위 · 질의 {v} | {mean(nd):.3f} [{lo:.3f}, {hi:.3f}] | {mean(n3):.3f} | {mean(rc):.3f} | "
                     f"{mean(nd)-mean(b_nd):+.3f} [{dlo:+.3f}, {dhi:+.3f}] | {mean(nd)-mean(s_nd):+.3f} [{slo:+.3f}, {shi:+.3f}] | "
                     f"{mean(n3)-mean(s_n3):+.3f} [{tlo:+.3f}, {thi:+.3f}] |")
        lines.append(f"| ↳ 질의 출처 | {srcs[v]} | | | | | |")
        with open(os.path.join(a.preds, f"ir_{name}_system_gate_rrf_{a.scorer}_top{a.top}_{v}.trec"), "w", encoding="utf-8") as f:
            for qid in qids:
                for rk, d in enumerate(runs[v][qid], 1):
                    f.write(f"{qid} Q0 {d} {rk} {1.0/rk:.6f} rerank_{a.scorer}_top{a.top}_{v}\n")
    lines += ["", f"재순위는 상위 {a.top}건 안의 순서만 바꾸므로 상위 100건의 문서 집합과 Recall@100 은 그대로다. 질의 raw = 영어 원문(한국어 기사라면 LLM 번역이 있을 때), "
              "mesh = MeSH 표목명 키워드(한국어 기사에서 LLM 없이 갈 때, 개념 0건이면 원문). 재순위는 MeSH 효과가 아니라 시스템 개선으로 보고한다.",
              f"점수 캐시: {os.path.basename(cache_path)} ({len(cache):,}쌍) · {time.time()-t0:.0f}초"]
    rp = os.path.join(HERE, f"report_ir_{name}_rerank_{a.scorer}_top{a.top}.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("→", rp)


if __name__ == "__main__":
    main()
