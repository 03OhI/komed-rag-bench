"""Diagnostic for E3 gate: how many gold docs fall inside the MeSH candidate set (exact-only concept extraction, identical
to run_nfcorpus.py without --fuzzy because non-exact hits are discarded there anyway). Read-only; does not change the harness.

Usage: python3 e3_gate_diag.py <bench dir> <beir dir> <mesh.json> [split]
"""
import collections
import json
import statistics
import sys

BENCH, BEIR, MESHJ = sys.argv[1], sys.argv[2], sys.argv[3]
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "test"
sys.path.insert(0, BENCH)
sys.path.insert(0, BENCH + "/..")
import run_nfcorpus as rn  # noqa: E402
import mesh_resolve as mr  # noqa: E402

# make the resolver exact-only (fast): fuzzy candidates are discarded by the harness when --fuzzy is off
if getattr(mr, "_HAVE_RAPIDFUZZ", False):
    class _NoFuzzy:
        @staticmethod
        def extract(*a, **k):
            return []
    mr.process = _NoFuzzy()
mr._HAVE_RAPIDFUZZ = True

corpus, queries, qrels = rn.load_beir(BEIR, SPLIT)
mesh = json.load(open(MESHJ, encoding="utf-8"))
idx = rn.Index(corpus, mesh, with_mesh=True)
extract = rn.make_concept_extractor(BENCH + "/../mesh_lookup.pkl", 85.0, False, None)

rows = []
for qid, qtext in queries.items():
    cons = extract(qtext)
    gold = {d for d, r in qrels[qid].items() if r > 0}
    gold_idx = {idx.ids.index(d) for d in gold if d in corpus} if gold else set()
    cands, mode = idx.candidates([c["ui"] for c in cons]) if cons else (None, "없음")
    if cands:
        cs = set(cands)
        cov = len(gold_idx & cs) / len(gold_idx) if gold_idx else None
        rows.append({"id": qid, "q": qtext, "n_con": len(cons), "cons": [(c["span"], c["name"]) for c in cons], "mode": mode,
                     "n_cand": len(cands), "n_gold": len(gold_idx), "cov": cov})
    else:
        rows.append({"id": qid, "q": qtext, "n_con": len(cons), "cons": [(c["span"], c["name"]) for c in cons], "mode": "폴백" if cons else "없음",
                     "n_cand": None, "n_gold": len(gold_idx), "cov": None})

gated = [r for r in rows if r["n_cand"]]
print(f"질의 {len(rows)} · 게이트 걸림 {len(gated)} · 개념 0건 {sum(1 for r in rows if r['n_con']==0)} · 개념 있으나 후보 없음 {sum(1 for r in rows if r['n_con'] and not r['n_cand'])}")
covs = [r["cov"] for r in gated if r["cov"] is not None]
print(f"게이트 걸린 질의의 정답 문서 포함률(후보 안에 든 정답 비율): 평균 {statistics.mean(covs):.3f} · 중앙값 {statistics.median(covs):.3f}")
print(f"  포함률 0 인 질의 {sum(1 for c in covs if c == 0)} · 1.0 인 질의 {sum(1 for c in covs if c == 1)}")
print(f"  정답 수 평균 {statistics.mean(r['n_gold'] for r in gated):.1f} · 후보 수 중앙값 {statistics.median(r['n_cand'] for r in gated):.0f}")
tot_gold = sum(r["n_gold"] for r in gated)
tot_in = sum(round(r["cov"] * r["n_gold"]) for r in gated if r["cov"] is not None)
print(f"  정답 문서 합산: {tot_in}/{tot_gold} = {tot_in/tot_gold:.3f}")
mode_c = collections.Counter(r["mode"] for r in rows)
print("모드:", dict(mode_c))
# examples of low coverage
print("\n포함률이 낮은 예 (후보 수 · 정답 수 · 포함률 · 개념):")
for r in sorted(gated, key=lambda r: (r["cov"] if r["cov"] is not None else 9, -r["n_gold"]))[:12]:
    print(f"  {r['q'][:40]:40s} 후보 {r['n_cand']:5d} · 정답 {r['n_gold']:3d} · 포함 {r['cov']:.2f} · {r['cons']}")
print("\n포함률이 높은 예:")
for r in sorted(gated, key=lambda r: (-(r["cov"] if r["cov"] is not None else -1), r["n_cand"]))[:8]:
    print(f"  {r['q'][:40]:40s} 후보 {r['n_cand']:5d} · 정답 {r['n_gold']:3d} · 포함 {r['cov']:.2f} · {r['cons']}")
json.dump(rows, open(sys.argv[1] + f"/../../e3_gate_diag_{SPLIT}.json", "w"), ensure_ascii=False, indent=0)
