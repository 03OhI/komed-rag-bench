#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_system_on_nfcorpus.py — E3 의 gate / gate_rrf 행을 **시스템 코드(retrieve.py) 그대로** NFCorpus 위에서 재현한다 (11/2).

왜: 논문의 E3 표는 bench/run_nfcorpus.py 하네스 값이다. "시스템은 게이트+RRF 를 쓴다"는 문장을 코드로 닫으려면
    실제 검색 모듈(retrieve.py — _candidates · _query_terms · 게이트 BM25 · 포스팅 전역 BM25 · RRF)이 같은 데이터에서 같은 수치를 내야 한다.

어떻게: NFCorpus 문서를 pubmed_db.jsonl 형식(mesh = 표목 이름)으로 바꿔 임시 폴더에 두고 build_pubmed_index.py(version 2) 로 색인한 뒤,
    retrieve.Retriever 의 한국어 개념 추출(concepts)만 하네스의 영어 exact n-gram 추출기로 바꿔 끼운다. 나머지는 retrieve.py 그대로.
    개념 0건·후보 0건이면 전역 순위를 쓴다(하네스 gate_rrf 가 bm25_mesh 로 폴백하는 것과 같은 규약; 실제 시스템은 그 자리에서 기권한다).
    MIN_UI_DOCS(코퍼스에 문헌 30건 미만인 개념 제외)는 코퍼스 10만 건 기준 값이라 3,633건 코퍼스에서는 0 이 같은 비율이다 — 기본 0, --min-ui-docs 로 바꿀 수 있다.

사용 (bench/ 에서):
    python run_system_on_nfcorpus.py --beir data/nfcorpus --mesh data/nfcorpus/mesh.json --preds preds_v2 [--split test] [--work data/nfcorpus_system]
    → report_ir_nfcorpus_system.md  (하네스 bm25 행 대비 Δ · 하네스 gate/gate_rrf 행과의 질의별 차이)
11/2 결과(test 323): 시스템 게이트 단독 0.291(하네스 0.292) · 시스템 게이트+RRF 0.320(하네스 0.323), Recall@100 0.277(0.272).

11/8 LLM 질의 확장 (E3_MeSH검색_개선안_1108.md §5) — dev 에서 고르기:
    python run_system_on_nfcorpus.py --split dev --llm gemini [--llm-modes global,leg,all] [--llm-mesh both] [--workers 8]
    → report_ir_nfcorpus_dev_system_llm_gemini.md · preds_v2/ir_nfcorpus_dev_system_llm_gemini_<모드>_<mesh|nomesh>.trec
    · LLM 출력은 bench/e3_llm_cache.jsonl 에 남는다(같은 주장·프롬프트면 다시 부르지 않음).
    · 시험 분할(test)에서 --llm 은 --final 을 줄 때만 돈다 — dev 에서 하나를 고정한 뒤 한 번만.
    · 개념 0건·후보 0건 질의의 폴백(전역 순위)에도 LLM 질의어를 더한다(retrieve.py 의 후보 0건 폴백과 같은 규약).
    · --llm mock 은 키 없이 경로만 확인한다(주장과 표목명을 그대로 돌려주는 가짜 응답).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)

STUB_KO = '''"""run_system_on_nfcorpus.py 가 쓰는 빈 KoResolver — retrieve.Retriever 가 import 만 하고, 개념 추출은 영어 추출기로 바꿔 낀다."""
class KoResolver:
    def resolve(self, span, k=1):
        return []
'''


CODE_FILES = ("retrieve.py", "build_pubmed_index.py", "bm25_util.py", "mesh_resolve.py", "mesh_lookup.pkl",
              "llm_query.py", "komed_llm.py", "mesh_desc_info.pkl", "medcpt_rerank.py")


def copy_code(work: str):
    """코드는 매번 복사한다(--no-prepare 여도) — 색인만 재사용하고 retrieve.py 는 늘 mesh/ 의 최신판."""
    os.makedirs(work, exist_ok=True)
    for f in CODE_FILES:
        src = os.path.join(MESH, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(work, f))
    open(os.path.join(work, "ko_resolver.py"), "w", encoding="utf-8").write(STUB_KO)


def prepare(work: str, beir: str, meshj: str):
    copy_code(work)
    mesh = json.load(open(meshj, encoding="utf-8"))
    n = 0
    with open(os.path.join(work, "pubmed_db.jsonl"), "w", encoding="utf-8") as out:
        for line in open(os.path.join(beir, "corpus.jsonl"), encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            names = [x["name"] for x in ((mesh.get(d["_id"]) or {}).get("mesh") or []) if x.get("name")]
            out.write(json.dumps({"pmid": d["_id"], "title": d.get("title") or "", "abstract": d.get("text") or "", "mesh": names,
                                  "query_ui": None, "year": None, "journal": None, "pubtypes": [], "stratum": "BASE"}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prepare] 문서 {n:,}건 → {work}/pubmed_db.jsonl", flush=True)
    subprocess.run([sys.executable, "build_pubmed_index.py", "--force"], cwd=work, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", default="data/nfcorpus")
    ap.add_argument("--mesh", default="data/nfcorpus/mesh.json")
    ap.add_argument("--preds", default="preds_v2", help="run_nfcorpus.py 가 쓴 *_queries.jsonl 폴더")
    ap.add_argument("--split", default="test")
    ap.add_argument("--work", default="data/nfcorpus_system")
    ap.add_argument("--min-ui-docs", type=int, default=0)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--no-prepare", action="store_true", help="임시 폴더·색인이 이미 있으면 건너뜀")
    ap.add_argument("--llm", default=None, help="LLM 질의 확장 공급자[:모델] (gemini · anthropic · mock). 없으면 11/2 와 같은 두 행만")
    ap.add_argument("--llm-modes", default="global,leg,all", help="retrieve.LLM_MODES 중 쉼표로")
    ap.add_argument("--llm-mesh", default="both", choices=("both", "on", "off"), help="프롬프트에 MeSH 정보를 줄지")
    ap.add_argument("--w-claim", type=float, default=1.0)
    ap.add_argument("--w-passage", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8, help="LLM 미리 부르기 병렬 수")
    ap.add_argument("--final", action="store_true", help="test 분할에서 LLM 조건을 돌릴 때만 (dev 에서 고정한 뒤 한 번)")
    a = ap.parse_args()
    if a.llm and a.split == "test" and not a.final:
        sys.exit("[중단] 시험 분할에서는 LLM 조건을 돌리지 않는다 — dev 에서 하나를 고정한 뒤 --final 로 한 번만 (11/8 규칙)")
    beir, meshj, preds, work = (os.path.abspath(p) for p in (a.beir, a.mesh, a.preds, a.work))
    if not a.no_prepare:
        prepare(work, beir, meshj)
    else:
        copy_code(work)

    sys.path.insert(0, HERE)
    import run_nfcorpus as rn
    import mesh_resolve as mr
    class _NoFuzzy:                     # exact 만 쓰므로 결과 동일, 별칭 25만 퍼지 탐색만 생략
        @staticmethod                   # (11/8) rapidfuzz 가 없는 환경에서도 돌도록 늘 바꿔 낀다
        def extract(*args, **kw):
            return []
    mr.process = _NoFuzzy()
    if not hasattr(mr, "fuzz"):         # 퍼지 호출의 인자(scorer=fuzz.token_sort_ratio)만 채운다 — 호출 결과는 위에서 버림
        mr.fuzz = type("_F", (), {"token_sort_ratio": staticmethod(lambda *a, **k: 0)})()
    mr._HAVE_RAPIDFUZZ = True
    sys.path.insert(0, work)
    cwd0 = os.getcwd()
    import retrieve                      # work/retrieve.py — import 시 work 로 chdir 한다
    assert os.path.dirname(os.path.abspath(retrieve.__file__)) == work, retrieve.__file__

    name = "nfcorpus" if a.split == "test" else f"nfcorpus_{a.split}"
    corpus, queries, qrels = rn.load_beir(beir, a.split)
    extract = rn.make_concept_extractor(os.path.join(work, "mesh_lookup.pkl"), 85.0, False, None)
    corpus_ids = [json.loads(l)["pmid"] for l in open(os.path.join(work, "pubmed_db.jsonl"), encoding="utf-8")]
    qids = list(queries)

    def load_bench(system):
        rows = {}
        for line in open(os.path.join(preds, f"ir_{name}_{system}_queries.jsonl"), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
        return rows
    bench = {s: load_bench(s) for s in ("bm25", "gate", "gate_rrf")}

    llm = None
    if a.llm:
        from komed_llm import LLM        # work/komed_llm.py
        import llm_query
        prov, _, model = a.llm.partition(":")
        llm = LLM(prov, model or None, cache_path=os.path.join(HERE, "e3_llm_cache.jsonl"),
                  mock_fn=llm_query.mock_fn if prov == "mock" else None)

    def run_system(fuse: bool, llm_mode: str = "off", llm_mesh: bool = True):
        r = retrieve.Retriever(index_path=os.path.join(work, "pubmed_index.pkl"), verbose=False, fuse=fuse,
                               llm=llm if llm_mode != "off" else None, llm_mode=llm_mode, llm_mesh=llm_mesh,
                               w_llm_claim=a.w_claim, w_llm_passage=a.w_passage)
        u2d = r.idx["ui2docs"]

        def concepts(claim):
            return [{"ui": c["ui"], "name": c["name"], "span": c["span"], "score": 100.0, "method": "exact", "track": "en"}
                    for c in extract(claim) if not a.min_ui_docs or len(u2d.get(c["ui"], ())) >= a.min_ui_docs]
        r.concepts = concepts
        run, modes, cmap = {}, collections.Counter(), {}
        for qid in qids:
            q = queries[qid]
            res = r.search(q, k=100)
            cmap[qid] = [c["name"] for c in res.get("concepts") or []]
            ids = [h["pmid"] for h in res["hits"]]
            mode = res.get("mode", "없음")
            if not ids:
                cons = res.get("concepts") or []
                qt = r._query_terms(q, cons)
                if llm_mode != "off":
                    lq = llm_query.terms(llm_query.expand(llm, q, cons, llm_mesh), a.w_claim, a.w_passage)
                    qt = llm_query.merge(qt, lq)
                ids = [corpus_ids[i] for i, _ in r._global_top(qt, 100)]
                mode = "폴백(전역)"
            run[qid] = ids
            modes[mode] += 1
        return run, modes, concepts, cmap

    def ci(vals):
        rnd = random.Random(0); n = len(vals); out = []
        for _ in range(2000):
            out.append(sum(vals[rnd.randrange(n)] for _ in range(n)) / n)
        out.sort()
        return out[50], out[1949]

    base_nd = [bench["bm25"][q]["ndcg10"] for q in qids]
    base_rc = [bench["bm25"][q]["recall100"] for q in qids]
    lines = [f"# 시스템 코드(retrieve.py)로 재현한 E3 — {name} (질의 {len(qids)} · 문서 {len(corpus):,} · MIN_UI_DOCS={a.min_ui_docs})", "",
             f"하네스 bm25 행: nDCG@10 {statistics.mean(base_nd):.3f} · Recall@100 {statistics.mean(base_rc):.3f}  (Δ 의 기준)", "",
             "| 조건 | nDCG@10 [95% CI] | Recall@100 | Δ vs bm25 [95% CI] | ΔRecall@100 [95% CI] | 하네스 같은 행 | 질의별 nDCG 평균차 | 상위 10 동일 |",
             "|---|---|---|---|---|---|---|---|"]
    configs = [(False, "off", True, "시스템 게이트 단독 (--no-fuse)", "gate", "gate"),
               (True, "off", True, "시스템 게이트+전역 RRF (기본)", "gate_rrf", "gate_rrf")]
    tag = (a.llm or "").split(":")[0]
    if llm:
        meshes = {"both": (True, False), "on": (True,), "off": (False,)}[a.llm_mesh]
        for m in meshes:
            for lm in [x.strip() for x in a.llm_modes.split(",") if x.strip()]:
                assert lm in retrieve.LLM_MODES[1:], lm
                configs.append((True, lm, m, f"LLM {lm} · MeSH {'있음' if m else '없음'}", "gate_rrf",
                                f"llm_{tag}_{lm}_{'mesh' if m else 'nomesh'}"))
        # LLM 을 질의마다 한 번(MeSH 있음/없음 각각) 미리 병렬로 불러 캐시에 넣는다. 모드끼리는 같은 출력을 쓴다.
        _, _, cfn, _ = run_system(False)
        items = [(queries[q], cfn(queries[q]), m) for q in qids for m in meshes]   # 개념 0건이면 cons=[] 로 폴백 자리와 같다
        t_pf = __import__("time").time()
        n_new = llm_query.prefetch(llm, items, a.workers)
        print(f"[llm] {llm.label} · 새 호출 {n_new}회 · 캐시 {len(llm.cache)}건 · {__import__('time').time()-t_pf:.0f}초", flush=True)

    per_q = {}
    for fuse, lm, m, label, ref, fname in configs:
        run, modes, _, cmap = run_system(fuse, lm, m)
        if fname == "gate_rrf":           # 11/8 재순위(rerank_nfcorpus.py)의 mesh 질의 변형이 쓴다
            with open(os.path.join(preds, f"ir_{name}_system_concepts.jsonl"), "w", encoding="utf-8") as f:
                for qid in qids:
                    f.write(json.dumps({"id": qid, "concepts": cmap[qid]}, ensure_ascii=False) + "\n")
        nd = [rn.ndcg(run[q], qrels[q], 10) for q in qids]
        rc = [rn.recall(run[q], qrels[q], 100) for q in qids]
        n3 = [rn.ndcg(run[q], qrels[q], 3) for q in qids]
        per_q[fname] = (nd, rc, n3)
        lo, hi = ci(nd); dlo, dhi = rn.boot_diff(base_nd, nd); rlo, rhi = rn.boot_diff(base_rc, rc)
        ref_nd = [bench[ref][q]["ndcg10"] for q in qids]
        mad = statistics.mean(abs(x - y) for x, y in zip(nd, ref_nd))
        same10 = statistics.mean(1.0 if run[q][:10] == bench[ref][q]["ranking"][:10] else 0.0 for q in qids)
        lines.append(f"| {label} | {statistics.mean(nd):.3f} [{lo:.3f}, {hi:.3f}] | {statistics.mean(rc):.3f} | "
                     f"{statistics.mean(nd)-statistics.mean(base_nd):+.3f} [{dlo:+.3f}, {dhi:+.3f}] | {statistics.mean(rc)-statistics.mean(base_rc):+.3f} [{rlo:+.3f}, {rhi:+.3f}] | "
                     f"{ref} {statistics.mean(ref_nd):.3f} | {mad:.4f} | {100*same10:.1f}% |")
        lines.append(f"| ↳ 모드 | {dict(modes)} | | | | | | |")
        with open(os.path.join(preds, f"ir_{name}_system_{fname}.trec"), "w", encoding="utf-8") as f:
            for qid, ranked in run.items():
                for rk, d in enumerate(ranked, 1):
                    f.write(f"{qid} Q0 {d} {rk} {1.0/rk:.6f} system_{fname}\n")

    if llm:
        # 고르는 데 쓰는 비교: 지금 시스템(게이트+RRF) 대비, 그리고 같은 모드에서 MeSH 있음 − 없음
        snd, src_, sn3 = per_q["gate_rrf"]
        lines += ["", f"## LLM 조건 비교 ({llm.label} · w_claim {a.w_claim} · w_passage {a.w_passage})", "",
                  "| 조건 | nDCG@3 | Δ nDCG@10 vs 시스템 게이트+RRF [95% CI] | Δ Recall@100 vs 시스템 게이트+RRF [95% CI] |",
                  "|---|---|---|---|",
                  f"| 시스템 게이트+RRF (기본) | {statistics.mean(sn3):.3f} | — | — |"]
        for fuse, lm, m, label, ref, fname in configs[2:]:
            nd, rc, n3 = per_q[fname]
            dlo, dhi = rn.boot_diff(snd, nd); rlo, rhi = rn.boot_diff(src_, rc)
            lines.append(f"| {label} | {statistics.mean(n3):.3f} | {statistics.mean(nd)-statistics.mean(snd):+.3f} [{dlo:+.3f}, {dhi:+.3f}] | "
                         f"{statistics.mean(rc)-statistics.mean(src_):+.3f} [{rlo:+.3f}, {rhi:+.3f}] |")
        if a.llm_mesh == "both":
            lines += ["", "| 모드 | MeSH 있음 − 없음, Δ nDCG@10 [95% CI] | Δ Recall@100 [95% CI] |", "|---|---|---|"]
            for lm in [x.strip() for x in a.llm_modes.split(",") if x.strip()]:
                on, off = per_q[f"llm_{tag}_{lm}_mesh"], per_q[f"llm_{tag}_{lm}_nomesh"]
                dlo, dhi = rn.boot_diff(off[0], on[0]); rlo, rhi = rn.boot_diff(off[1], on[1])
                lines.append(f"| {lm} | {statistics.mean(on[0])-statistics.mean(off[0]):+.3f} [{dlo:+.3f}, {dhi:+.3f}] | "
                             f"{statistics.mean(on[1])-statistics.mean(off[1]):+.3f} [{rlo:+.3f}, {rhi:+.3f}] |")
        lines += ["", "고르는 기준(개선안 §3, 돌리기 전에 고정): dev nDCG@10 최고. 단 dev Recall@100 이 시스템 게이트+RRF 보다 낮지 않을 것. "
                  "차이가 0.005 안이면 단순한 쪽(global < leg < all 순으로 단순). LLM 출력은 bench/e3_llm_cache.jsonl."]
    lines += ["", "시스템과 하네스의 남은 차이: 질의어(시스템은 표목 별칭 토큰 0.35 를 더하고 전역 검색에도 같은 질의어를 씀 — 하네스 전역은 원 질의 토큰만) · "
              "체크태그 16개(Humans·Adult 등)를 시스템 색인은 ui2docs 에서 뺌 · 폴백 자리에서 실제 시스템은 기권하지만 여기서는 전역 순위를 채점함. "
              "run 파일 preds_v2/ir_<name>_system_*.trec 는 trec_eval 로 재검증할 수 있다."]
    os.chdir(cwd0)
    rp = a.report_out or os.path.join(HERE, f"report_ir_{name}_system{'_llm_' + tag if llm else ''}.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("→", rp)


if __name__ == "__main__":
    main()
