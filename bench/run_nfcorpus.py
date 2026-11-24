#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nfcorpus.py — E3 검색 계층을 BEIR 형식 벤치마크(NFCorpus 등)에서 평가: 순수 BM25 vs MeSH 게이트 BM25 (논문 4.5.2 설계 그대로)

    python fetch_mesh_pmids.py --corpus data/nfcorpus/corpus.jsonl --out data/nfcorpus/mesh.json      # 먼저 (PMID → MeSH 표목)
    python run_nfcorpus.py --beir data/nfcorpus --mesh data/nfcorpus/mesh.json --systems bm25 bm25_mesh gate gate_rrf

시스템
    bm25        제목+본문 토큰만 (BEIR 의 BM25 와 같은 입력; 토크나이저는 bm25_util — 문헌 0.325 와 소폭 차이 가능)
    bm25_mesh   + MeSH 표목 토큰(가중 2, 제목 3 — build_pubmed_index 와 동일) — 게이트 없이 표목만 쓴 대조군
    gate        질의 → 어절 n-gram(3→2→1) 최장일치 → 영어 리졸버(exact; --fuzzy 로 퍼지 허용) → MeSH UI → ui2docs 후보 축소
                (교집합 ≥ MIN_INTERSECT → 2축 교집합 → 합집합; 코퍼스 20% 초과 UI 제외) → 후보 안 BM25(전역 df/avgdl) — retrieve.py 의 규칙과 상수 그대로.
                개념 0건이거나 후보 0건이면 bm25_mesh 로 폴백(Retrieval Evaluator 의 incorrect 처리와 같은 취지)
    gate_rrf    gate 와 bm25_mesh 의 RRF(k=60) — 게이트가 놓친 것을 되살리는 안전판

지표  nDCG@10(선형 이득, pytrec_eval 과 동일) · Recall@100 · P@10 · MRR@10 · 질의당 채점 문서 수 · ms/질의 · 질의 부트스트랩 95% CI · gate−bm25 짝지은 CI
출력  report_ir_<name>.md · preds/ir_<name>_<system>.trec (TREC run 형식 — trec_eval/pytrec_eval 로 재검증 가능)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from bm25_util import tokenize, bm25, TITLE_W, MESH_W  # noqa: E402
from protocol import DEFAULT_PREDS

MAX_UI_RATIO, MIN_INTERSECT, MAX_CANDS, RRF_K = 0.20, 20, 20000, 60   # retrieve.py 와 동일
W_NAME, W_QUERY = 1.00, 1.00
REFERENCE = {"nfcorpus": [("BM25 (BEIR 논문, Anserini multifield)", "0.325", "0.250"), ("Contriever-msmarco", "0.328", "0.300"),
                          ("SPLADE++ (EnsembleDistil)", "0.348", "-"), ("ColBERTv2", "0.338", "-"),
                          ("MedCPT retriever / +cross-encoder", "0.340 / 0.355", "-"),
                          ("BMQExpander (UMLS/MeSH 확장 + BM25, 2025)", "0.363", "-"),
                          ("bge-base-en-v1.5 / bge-large", "0.374 / 0.381", "0.337 / 0.365"), ("NV-Embed-v2 (7B)", "0.452", "0.438")],
             "scifact": [("BM25 (BEIR)", "0.665", "0.908"), ("MedCPT retriever", "0.724", "-"), ("bge-base-en-v1.5", "0.744", "0.963")]}


# ───────────────────────────────────────────────────────────── BEIR 로더
def load_beir(root: str, split: str = "test"):
    corpus, queries, qrels = {}, {}, collections.defaultdict(dict)
    for line in open(os.path.join(root, "corpus.jsonl"), encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title") or "", "text": d.get("text") or "", "url": (d.get("metadata") or {}).get("url")}
    for line in open(os.path.join(root, "queries.jsonl"), encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            queries[q["_id"]] = q.get("text") or ""
    with open(os.path.join(root, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].strip().lstrip("-").isdigit():
                continue  # header
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(float(parts[2]))
    queries = {k: v for k, v in queries.items() if k in qrels}
    return corpus, queries, qrels


# ───────────────────────────────────────────────────────────── 인덱스
class Index:
    def __init__(self, corpus: dict, mesh: dict | None, with_mesh: bool):
        self.ids = list(corpus)
        self.tf, self.dl, self.df = [], [], collections.Counter()
        self.ui2docs = collections.defaultdict(set)
        for i, did in enumerate(self.ids):
            d = corpus[did]
            m = (mesh or {}).get(did) or {}
            names = [x["name"] for x in m.get("mesh") or [] if x.get("name")]
            toks = tokenize(d["title"]) * TITLE_W + tokenize(d["text"])
            if with_mesh:
                toks += tokenize(" ".join(names)) * MESH_W
            tf = collections.Counter(toks)
            self.tf.append(tf)
            self.dl.append(len(toks))
            for t in tf:
                self.df[t] += 1
            for x in m.get("mesh") or []:
                if x.get("ui"):
                    self.ui2docs[x["ui"]].add(i)
        self.n = len(self.ids)
        self.avgdl = sum(self.dl) / max(1, self.n)

    def score(self, qterms: dict, cands=None):
        rng = range(self.n) if cands is None else cands
        out = []
        for i in rng:
            tf = self.tf[i]
            if not any(t in tf for t in qterms):
                continue
            out.append((bm25(qterms, tf, self.dl[i], self.df, self.n, self.avgdl), i))
        out.sort(key=lambda x: -x[0])
        return out

    def candidates(self, uis):
        usable = [(u, self.ui2docs[u]) for u in uis if u in self.ui2docs and len(self.ui2docs[u]) <= MAX_UI_RATIO * self.n]
        if not usable:
            return None, "없음"
        sets = [s for _, s in usable]
        if len(sets) >= 2:
            inter = set.intersection(*sets)
            if len(inter) >= MIN_INTERSECT:
                return sorted(inter), "교집합(전체)"
            best = None
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    cur = sets[i] & sets[j]
                    if len(cur) >= MIN_INTERSECT and (best is None or len(cur) < len(best)):
                        best = cur
            if best:
                return sorted(best), "교집합(2축)"
        union = set()
        for s in sets:
            union |= s
        return sorted(union)[:MAX_CANDS], "합집합"


# ───────────────────────────────────────────────────────────── 개념 추출 (영어)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
_SKIP = set("the a an of in on for and or to with without is are be does do can what how why which who when where"
            " effect effects risk benefit benefits health disease diseases".split())


def make_concept_extractor(kb: str, threshold: float, fuzzy: bool, mock_lookup=None, max_ngram: int = 3):
    import mesh_resolve as mr
    lookup = mock_lookup if mock_lookup is not None else mr._load_lookup(kb)
    res = mr.make_resolver(lookup=lookup, threshold=threshold, tree_prefixes=None)

    def extract(query: str):
        words = _WORD.findall(query)
        used = [False] * len(words)
        found = []
        for n in range(max_ngram, 0, -1):
            for s in range(0, len(words) - n + 1):
                if any(used[s:s + n]):
                    continue
                span = " ".join(words[s:s + n])
                if n == 1 and (span.lower() in _SKIP or len(span) <= 3):
                    continue
                r = res(span, k=1)
                if not r:
                    continue
                if r[0].get("method") != "exact" and not fuzzy:
                    continue
                found.append({"span": span, "ui": r[0]["ui"], "name": r[0]["name"], "method": r[0].get("method")})
                for j in range(s, s + n):
                    used[j] = True
        return found
    return extract


# ───────────────────────────────────────────────────────────── 지표
def ndcg(ranked, rel: dict, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked, rel, k=100):
    R = {d for d, r in rel.items() if r > 0}
    return len(R & set(ranked[:k])) / len(R) if R else 0.0


def precision(ranked, rel, k=10):
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def mrr(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1 / (i + 1)
    return 0.0


def boot_ci(vals, n_boot=2000, seed=0):
    rnd = random.Random(seed)
    n = len(vals)
    out = []
    for _ in range(n_boot):
        out.append(sum(vals[rnd.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot) - 1]


def boot_diff(a, b, n_boot=2000, seed=0):
    rnd = random.Random(seed)
    n = len(a)
    out = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        out.append(sum(b[i] - a[i] for i in idx) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot) - 1]


def rrf(lists, k=60, top=100):
    sc = collections.defaultdict(float)
    for lst in lists:
        for r, d in enumerate(lst, 1):
            sc[d] += 1.0 / (k + r)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])[:top]]


# ───────────────────────────────────────────────────────────── 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", required=True, help="BEIR 형식 폴더 (corpus.jsonl · queries.jsonl · qrels/test.tsv)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--mesh", default=None, help="fetch_mesh_pmids.py 출력")
    ap.add_argument("--systems", nargs="+", choices=["bm25", "bm25_mesh", "bm25_mesh_expand", "gate", "gate_rrf"],
                    default=["bm25", "bm25_mesh", "bm25_mesh_expand", "gate", "gate_rrf"])
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--kb", default=os.path.join(MESH, "mesh_lookup.pkl"))
    ap.add_argument("--threshold", type=float, default=85.0)
    ap.add_argument("--fuzzy", action="store_true", help="개념 추출에 퍼지 매칭 허용 (기본 exact 만)")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_PREDS)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.beir))
    corpus, queries, qrels = load_beir(a.beir, a.split)
    if not queries or not corpus or a.top < 100:
        ap.error("Require nonempty corpus/queries and --top >= 100 for Recall@100")
    if not a.mesh and any(s != "bm25" for s in a.systems):
        ap.error("--mesh is required for MeSH systems")
    if a.limit:
        queries = dict(list(queries.items())[:a.limit])
    mesh = json.load(open(a.mesh, encoding="utf-8")) if a.mesh else None
    n_mesh = sum(1 for v in (mesh or {}).values() if v.get("mesh"))
    print(f"[{name}] 문서 {len(corpus):,} · 질의 {len(queries)} · MeSH 있는 문서 {n_mesh:,}")
    mock_lookup = None
    if a.mock:
        mock_lookup = {"breast cancer": {"ui": "D001943", "name": "Breast Neoplasms", "trees": []},
                       "breast neoplasms": {"ui": "D001943", "name": "Breast Neoplasms", "trees": []},
                       "coffee": {"ui": "D003069", "name": "Coffee", "trees": []},
                       "liver neoplasms": {"ui": "D008113", "name": "Liver Neoplasms", "trees": []},
                       "liver cancer": {"ui": "D008113", "name": "Liver Neoplasms", "trees": []},
                       "vitamin d": {"ui": "D014807", "name": "Vitamin D", "trees": []}}
    idx_plain = Index(corpus, mesh, with_mesh=False)
    idx_mesh = Index(corpus, mesh, with_mesh=True)
    extract = make_concept_extractor(a.kb, a.threshold, a.fuzzy, mock_lookup) if any(s.startswith("gate") or s == "bm25_mesh_expand" for s in a.systems) else None
    os.makedirs(a.out, exist_ok=True)

    runs, stats = {}, {}
    for sysname in a.systems:
        t0 = time.time()
        run, n_scored, n_fallback, n_concepts = {}, [], 0, []
        query_ms, details = [], []
        for qid, qtext in queries.items():
            query_start = time.perf_counter()
            visited, mode, cons = idx_plain.n, "global", []
            qterms = {t: W_QUERY for t in tokenize(qtext)}
            if sysname == "bm25":
                scored = idx_plain.score(qterms)
            elif sysname == "bm25_mesh":
                scored = idx_mesh.score(qterms)
            elif sysname in ("gate", "gate_rrf", "bm25_mesh_expand"):
                cons = extract(qtext)
                n_concepts.append(len(cons))
                q2 = dict(qterms)
                for c in cons:
                    for t in tokenize(c["name"]):
                        q2[t] = max(q2.get(t, 0), W_NAME)
                cands, mode = idx_mesh.candidates([c["ui"] for c in cons]) if cons and sysname != "bm25_mesh_expand" else (None, "없음")
                if cands:
                    visited = len(cands)
                    scored = idx_mesh.score(q2, cands)
                    if not scored:
                        visited += idx_mesh.n
                        scored, mode = idx_mesh.score(q2), "폴백"
                else:
                    scored, mode = idx_mesh.score(q2), "폴백"
                if mode == "폴백" and sysname != "bm25_mesh_expand":
                    n_fallback += 1
                if sysname == "gate_rrf":
                    base = [idx_mesh.ids[i] for _, i in idx_mesh.score(qterms)[:a.top]]
                    gate = [idx_mesh.ids[i] for _, i in scored[:a.top]]
                    run[qid] = rrf([gate, base], RRF_K, a.top)
                    visited += idx_mesh.n
                    n_scored.append(visited)
                    elapsed = (time.perf_counter() - query_start) * 1000
                    query_ms.append(elapsed)
                    details.append({"id": qid, "visited": visited, "ms": elapsed, "mode": mode,
                                    "n_concepts": len(cons), "ranking": run[qid]})
                    continue
            else:
                raise ValueError(sysname)
            n_scored.append(visited)
            run[qid] = [idx_plain.ids[i] for _, i in scored[:a.top]]
            elapsed = (time.perf_counter() - query_start) * 1000
            query_ms.append(elapsed)
            details.append({"id": qid, "visited": visited, "ms": elapsed, "mode": mode,
                            "n_concepts": len(cons), "ranking": run[qid]})
        ms = (time.time() - t0) * 1000 / max(1, len(queries))
        runs[sysname] = run
        stats[sysname] = {"ms": ms, "scored": sum(n_scored) / max(1, len(n_scored)), "fallback": n_fallback,
                          "concepts": (sum(n_concepts) / len(n_concepts)) if n_concepts else None,
                          "p50": sorted(query_ms)[len(query_ms)//2],
                          "p95": sorted(query_ms)[min(len(query_ms)-1, int(.95*len(query_ms)))]}
        with open(os.path.join(a.out, f"ir_{name}_{sysname}_queries.jsonl"), "w", encoding="utf-8") as f:
            for row in details:
                row.update(ndcg10=ndcg(row["ranking"], qrels[row["id"]]),
                           recall100=recall(row["ranking"], qrels[row["id"]]), split=a.split)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(a.out, f"ir_{name}_{sysname}.trec"), "w", encoding="utf-8") as f:
            for qid, ranked in run.items():
                for r, d in enumerate(ranked, 1):
                    f.write(f"{qid} Q0 {d} {r} {1.0/r:.6f} {sysname}\n")

    lines = [f"# 검색 계층 — {name} (질의 {len(queries)} · 문서 {len(corpus):,} · MeSH 문서 {n_mesh:,})", "",
             "| 시스템 | nDCG@10 [95% CI] | Recall@100 | P@10 | MRR@10 | 질의당 문서 점검 횟수 | ms/질의 | 개념 수(평균) | 폴백 |",
             "|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for sysname, run in runs.items():
        qids = list(queries)
        nd = [ndcg(run[q], qrels[q], 10) for q in qids]
        rc = [recall(run[q], qrels[q], 100) for q in qids]
        pr = [precision(run[q], qrels[q], 10) for q in qids]
        mr_ = [mrr(run[q], qrels[q], 10) for q in qids]
        per[sysname] = nd
        st = stats[sysname]
        ci = boot_ci(nd)
        lines.append(f"| {sysname} | {sum(nd)/len(nd):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {sum(rc)/len(rc):.3f} | {sum(pr)/len(pr):.3f} | {sum(mr_)/len(mr_):.3f} | "
                     f"{st['scored']:.0f} | {st['ms']:.0f} | {('%.2f' % st['concepts']) if st['concepts'] is not None else '-'} | {st['fallback'] if st['concepts'] is not None else '-'} |")
    for ref, nd10, r100 in REFERENCE.get(name, []):
        lines.append(f"| (문헌) {ref} | {nd10} | {r100} | | | | | | |")
    if "bm25" in per:
        lines += ["", "**bm25 대비 짝지은 차이 (nDCG@10, 질의 부트스트랩 95% CI)**", "", "| 시스템 | Δ | 95% CI |", "|---|---|---|"]
        for s, nd in per.items():
            if s != "bm25":
                d = boot_diff(per["bm25"], nd)
                lines.append(f"| {s} | {sum(nd)/len(nd) - sum(per['bm25'])/len(per['bm25']):+.3f} | [{d[0]:+.3f}, {d[1]:+.3f}] |")
    lines += ["", "nDCG 는 선형 이득(pytrec_eval 기본)·log2(순위+1). 문헌 행은 논문 보고값. bm25 는 제목+본문만, bm25_mesh·gate 는 MeSH 표목 토큰 포함(제목 ×3 · 표목 ×2, 논문 4.5.2)."]
    if "bm25_mesh_expand" in per:
        base = "bm25_mesh_expand"
        lines += ["", "**동일 MeSH 색인·확장 질의의 게이트 비교**", "",
                  "| 시스템 | ΔnDCG@10 [95% CI] | 문서 점검 감소율 % | p50 / p95 ms |", "|---|---|---|---|"]
        for s in ("gate", "gate_rrf"):
            if s not in per:
                continue
            ci = boot_diff(per[base], per[s])
            reduction = 100 * (1 - stats[s]["scored"] / stats[base]["scored"])
            delta = sum(per[s])/len(per[s])-sum(per[base])/len(per[base])
            lines.append(f"| {s} | {delta:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {reduction:.1f} | {stats[s]['p50']:.2f} / {stats[s]['p95']:.2f} |")
        lines.append("gate가 게이트 단독 대조다. gate_rrf는 원 질의 전역 검색까지 결합한 별도 시스템이다. "
                     "점검 횟수는 양의 점수를 받은 문서 수가 아니라 실제 방문한 후보 수이며 폴백·RRF의 추가 방문을 포함한다. "
                     "지연은 탐색값이며 속도 우월성 주장 전 같은 장비에서 워밍업·순서 교대·반복이 필요하다. "
                     "비열등 허용폭은 dev에서 근거를 정해 별도 동결한다. 이 표는 비열등을 자동 선언하지 않는다.")
    rp = a.report_out or os.path.join(a.out, f"report_ir_{name}_v2.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("→", rp)


if __name__ == "__main__":
    main()
