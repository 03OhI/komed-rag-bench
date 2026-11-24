"""E3 per-query summary: candidates visited, modes, concept counts, paired deltas (reads run_nfcorpus.py's *_queries.jsonl)."""
import collections
import json
import math
import random
import statistics
import sys

PRED = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "nfcorpus"
SYSTEMS = sys.argv[3:] or ["bm25", "bm25_mesh", "gate", "gate_rrf"]


def load(system):
    rows = {}
    with open(f"{PRED}/ir_{NAME}_{system}_queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def boot_diff(a, b, n_boot=2000, seed=0):
    rnd = random.Random(seed)
    n = len(a)
    out = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        out.append(sum(b[i] - a[i] for i in idx) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot) - 1]


data = {s: load(s) for s in SYSTEMS}
qids = sorted(data[SYSTEMS[0]])
print(f"질의 {len(qids)}")
for s in SYSTEMS:
    rows = [data[s][q] for q in qids]
    nd = [r["ndcg10"] for r in rows]
    rc = [r["recall100"] for r in rows]
    vis = [r["visited"] for r in rows]
    modes = collections.Counter(r["mode"] for r in rows)
    ncon = [r["n_concepts"] for r in rows]
    print(f"\n== {s} ==")
    print(f"  nDCG@10 {statistics.mean(nd):.3f}  Recall@100 {statistics.mean(rc):.3f}")
    print(f"  방문 후보: 평균 {statistics.mean(vis):.0f} · 중앙값 {statistics.median(vis):.0f} · p25 {sorted(vis)[len(vis)//4]} · p75 {sorted(vis)[3*len(vis)//4]} · 최소 {min(vis)} · 최대 {max(vis)}")
    print(f"  ms/질의: 중앙값 {statistics.median([r['ms'] for r in rows]):.1f}")
    print(f"  모드: {dict(modes)}")
    if any(ncon):
        print(f"  개념 수: 평균 {statistics.mean(ncon):.2f} · 0건 {sum(1 for n in ncon if n == 0)} ({100*sum(1 for n in ncon if n == 0)/len(ncon):.1f}%) · 1건 {sum(1 for n in ncon if n == 1)} · 2건+ {sum(1 for n in ncon if n >= 2)}")
        gated = [r for r in rows if r["mode"] not in ("폴백", "global", "없음")]
        if gated:
            gv = [r["visited"] for r in gated]
            print(f"  게이트가 실제로 걸린 질의 {len(gated)} ({100*len(gated)/len(rows):.1f}%): 방문 후보 평균 {statistics.mean(gv):.0f} · 중앙값 {statistics.median(gv):.0f}")
            gnd = statistics.mean(r["ndcg10"] for r in gated)
            base = statistics.mean(data[SYSTEMS[0]][r["id"]]["ndcg10"] for r in gated)
            print(f"  게이트 걸린 질의에서 nDCG@10: {s} {gnd:.3f} vs {SYSTEMS[0]} {base:.3f}")
            fb = [r for r in rows if r["mode"] == "폴백"]
            if fb:
                print(f"  폴백 질의 {len(fb)}: nDCG@10 {statistics.mean(r['ndcg10'] for r in fb):.3f} vs {SYSTEMS[0]} {statistics.mean(data[SYSTEMS[0]][r['id']]['ndcg10'] for r in fb):.3f}")

print("\n== 짝지은 차이 (기준 = %s) ==" % SYSTEMS[0])
base_nd = [data[SYSTEMS[0]][q]["ndcg10"] for q in qids]
base_rc = [data[SYSTEMS[0]][q]["recall100"] for q in qids]
for s in SYSTEMS[1:]:
    nd = [data[s][q]["ndcg10"] for q in qids]
    rc = [data[s][q]["recall100"] for q in qids]
    lo, hi = boot_diff(base_nd, nd)
    rlo, rhi = boot_diff(base_rc, rc)
    wins = sum(1 for a, b in zip(base_nd, nd) if b > a + 1e-9)
    loss = sum(1 for a, b in zip(base_nd, nd) if b < a - 1e-9)
    print(f"  {s}: ΔnDCG@10 {statistics.mean(nd)-statistics.mean(base_nd):+.3f} [{lo:+.3f}, {hi:+.3f}] · ΔRecall@100 {statistics.mean(rc)-statistics.mean(base_rc):+.3f} [{rlo:+.3f}, {rhi:+.3f}] · 질의별 이김 {wins} 짐 {loss} 같음 {len(qids)-wins-loss}")
if "bm25_mesh" in data and "gate" in data:
    a = [data["bm25_mesh"][q]["ndcg10"] for q in qids]
    b = [data["gate"][q]["ndcg10"] for q in qids]
    lo, hi = boot_diff(a, b)
    print(f"  gate − bm25_mesh (같은 색인, 게이트만 다름): {statistics.mean(b)-statistics.mean(a):+.3f} [{lo:+.3f}, {hi:+.3f}]")
