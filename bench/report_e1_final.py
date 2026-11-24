"""실험 1 최종표 — 직접 판정 · 규칙을 뺀 조건 · KoMed-RAG (2026-11-16).

설계: claude_backup/실험1·2_재설계_확정_1116.md. 지표 묶음의 근거는 claude_backup/문헌비교_실험설계_1116.md §3
(CRAG 벤치마크의 정답 +1 · 기권 0 · 오답 −1, Feng 외의 답한 것 중 정답률, RGB 의 거부율, Wen 외의 "묶음으로 보고").

    python report_e1_final.py --out ../claude_backup/실험1_최종표_1116.md

정답(+1) = 예측이 정답과 같음(근거 불충분을 근거 불충분으로 답한 것 포함) · 불필요한 보류(0) = 정답이 지지·반박인데 근거 불충분으로 답함
· 틀린 확신(−1) = 지지·반박으로 답했는데 틀림. 진실성 = (정답 − 틀린 확신) / n.
신뢰구간: 군집 부트스트랩 2,000회 (HealthVer 는 질문 단위, Check-COVID 는 문헌 단위). 근거 문장이 여러 주장에 반복 쓰이기 때문이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = "gemini-gemini-3.1-pro-preview"
LAB = ("SUPPORT", "REFUTE", "NEI")
SR = ("SUPPORT", "REFUTE")
CC_DUP_PREFIX = "6026e71e6bc088b6d218dc11"      # Check-COVID test 중복 3행 (e1_pooled_1113.py 와 같은 규칙)


def label(rec: dict, exceed_to_nei: bool = False) -> str:
    return "NEI" if exceed_to_nei and rec.get("category") == "범위 초과" else rec["pred"]


def metrics(golds: list[str], preds: list[str]) -> dict:
    n = len(golds)
    correct = sum(g == p for g, p in zip(golds, preds))
    wrong = sum(p in SR and p != g for g, p in zip(golds, preds))
    missed = sum(g in SR and p == "NEI" for g, p in zip(golds, preds))
    answered = sum(p in SR for p in preds)
    n_sr = sum(g in SR for g in golds)
    flips = sum(g in SR and p in SR and g != p for g, p in zip(golds, preds))
    n_nei = n - n_sr
    f1 = []
    for c in LAB:
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"n": n, "correct": correct, "missed": missed, "wrong": wrong, "answered": answered,
            "truth": (correct - wrong) / n if n else float("nan"),
            "acc": 100 * correct / n if n else float("nan"), "mf1": 100 * sum(f1) / 3,
            "n_sr": n_sr, "flips": flips, "flip_rate": 100 * flips / n_sr if n_sr else float("nan"),
            "prec": 100 * (answered - wrong) / answered if answered else float("nan"),
            "coverage": 100 * answered / n if n else float("nan"),
            "nei_reject": 100 * sum(g == "NEI" and p == "NEI" for g, p in zip(golds, preds)) / n_nei if n_nei else float("nan")}


def breakeven(m_a: dict, m_b: dict):
    """정답 +1 · 오답 −c 로 채점할 때 a 가 b 를 앞서는 c 의 상한. 오답이 늘지 않으면 None(벌점 크기와 무관)."""
    dc, dw = m_a["correct"] - m_b["correct"], m_a["wrong"] - m_b["wrong"]
    return dc / dw if dw > 0 else None


def boot_diff(golds, preds_a, preds_b, clusters, fn, n_boot: int = 2000, seed: int = 0):
    groups = defaultdict(list)
    for k, c in enumerate(clusters):
        groups[c].append(k)
    G = list(groups.values())
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = [k for j in rng.integers(0, len(G), len(G)) for k in G[j]]
        g = [golds[k] for k in idx]
        d.append(fn(metrics(g, [preds_a[k] for k in idx])) - fn(metrics(g, [preds_b[k] for k in idx])))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def consistency(recs: list[dict]) -> float:
    return 100 * sum(bool(r.get("consistent")) for r in recs) / len(recs) if recs else float("nan")


# ───────────────────────────────────────────────────────────── 입출력
def read(paths: list[str], drop_prefix: str | None = None) -> dict:
    out = {}
    for p in paths:
        if not os.path.isfile(p):
            return {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if drop_prefix and r["id"].startswith(drop_prefix):
                    continue
                out[r["id"]] = r                 # id 별 마지막 기록 (오류 뒤 재시도 기록이 이긴다)
    bad = [i for i, r in out.items() if r.get("pred") not in LAB]
    if bad:
        raise ValueError(f"오류/비정상 예측이 마지막 기록으로 남음: {paths} {bad[:5]} (총 {len(bad)}건)")
    return out


def _hv_questions(csv_path):
    return {f"hv{r['id']}": r["question"].strip().lower() for r in csv.DictReader(open(csv_path, encoding="utf-8"))}


DATASETS = {
    "HealthVer test": {
        "free": [f"preds_e1_1113/healthver_free_{M}.jsonl"], "ours": [f"preds_e1_1113/healthver_ours_{M}.jsonl"],
        "norule": [f"preds_e1_norule_test/healthver_norule_{M}.jsonl"], "drop": None,
        "cluster": lambda r, q=None: q, "csv": "data/healthver/data/healthver_test.csv", "unit": "질문"},
    "Check-COVID test": {
        "free": [f"preds_e1_1113/checkcovid_free_{M}.jsonl", f"preds_e1_1113_ccext/checkcovid_free_{M}.jsonl"],
        "ours": [f"preds_e1_1113/checkcovid_ours_{M}.jsonl", f"preds_e1_1113_ccext/checkcovid_ours_{M}.jsonl"],
        "norule": [f"preds_e1_norule_test/checkcovid_norule_{M}.jsonl"], "drop": CC_DUP_PREFIX,
        "cluster": None, "csv": None, "unit": "문헌"},
}
NAMES = {"free": "직접 판정", "norule": "규칙을 뺀 조건", "ours": "KoMed-RAG"}


def _f(x, d=1):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def section(name: str, cfg: dict) -> str:
    conds = {c: read([os.path.join(HERE, p) for p in cfg[c]], cfg["drop"]) for c in ("free", "norule", "ours")}
    present = [c for c in ("free", "norule", "ours") if conds[c]]
    ids = list(conds["free"])
    for c in present:
        if set(conds[c]) != set(ids):
            raise ValueError(f"{name}: {c} 의 id 가 직접 판정과 다름 ({len(set(conds[c]) ^ set(ids))}건)")
    golds = [conds["free"][i]["gold"] for i in ids]
    if cfg["csv"]:
        q = _hv_questions(os.path.join(HERE, cfg["csv"]))
        clusters = [q[i] for i in ids]
    else:
        clusters = [conds["free"][i]["meta"]["doc_id"] for i in ids]
    P = {(c, e): [label(conds[c][i], e) for i in ids] for c in present for e in (False, True)}
    Mx = {k: metrics(golds, v) for k, v in P.items()}
    L = [f"## {name} — n={len(ids)} ({cfg['unit']} {len(set(clusters))}개 · 정답 지지·반박 {Mx[('free', False)]['n_sr']} · 근거 불충분 {len(ids) - Mx[('free', False)]['n_sr']})", "",
         "| 조건 | 진실성 | 정답 % | 불필요한 보류 % | 틀린 확신 % (건) | 답한 수 (응답률 %) | 답한 것 중 정답 % | 위험한 뒤집힘 % (건) | 근거 불충분 거부율 % | 정확도 % | macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(tag, m):
        n = m["n"]
        return (f"| {tag} | {m['truth']:+.3f} | {100 * m['correct'] / n:.1f} | {100 * m['missed'] / n:.1f} | {100 * m['wrong'] / n:.1f} ({m['wrong']}) | "
                f"{m['answered']} ({m['coverage']:.1f}) | {_f(m['prec'])} | {_f(m['flip_rate'])} ({m['flips']}) | {_f(m['nei_reject'])} | {m['acc']:.1f} | {m['mf1']:.1f} |")
    for c in present:
        L.append(row(NAMES[c], Mx[(c, False)]))
    for c in present:
        if c != "free":
            L.append(row(f"{NAMES[c]} · 범위 초과를 보류로 접음", Mx[(c, True)]))
    L += ["", "짝지은 차이 [95% 신뢰구간]", "",
          "| 비교 | Δ진실성 | Δ틀린 확신 (건) | Δ정답 (건) | Δ위험한 뒤집힘 (%p) | 오답 벌점 손익분기 |", "|---|---|---|---|---|---|"]
    pairs = [(a, b) for a, b in (("ours", "free"), ("ours", "norule"), ("norule", "free")) if a in present and b in present]
    for a, b in pairs:
        cells = []
        for key, fmt in (("truth", "{:+.3f}"), ("wrong", "{:+.0f}"), ("correct", "{:+.0f}"), ("flip_rate", "{:+.1f}")):
            pt = Mx[(a, False)][key] - Mx[(b, False)][key]
            lo, hi = boot_diff(golds, P[(a, False)], P[(b, False)], clusters, lambda m, k=key: m[k])
            cells.append(f"{fmt.format(pt)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        be = breakeven(Mx[(a, False)], Mx[(b, False)])
        cells.append("오답이 늘지 않음" if be is None else f"오답을 {be:.1f}배까지 무겁게 쳐도 앞섬" if be > 0 else "앞서지 못함")
        L.append(f"| {NAMES[a]} − {NAMES[b]} | " + " | ".join(cells) + " |")
    if "norule" in present:
        recs = [conds["norule"][i] for i in ids]
        inc = Counter((r["rule_category"], r["category"]) for r in recs if not r.get("consistent"))
        L += ["", f"판독–판정 일치율(규칙을 뺀 조건): {consistency(recs):.1f}% — LLM 이 낸 범주가 자기 판독에 고정 규칙을 적용한 범주와 같은 비율. "
              f"KoMed-RAG 는 구조상 100%다(저장된 판독 재적용으로 확인).",
              "어긋난 경우 (규칙이라면 → LLM 이 낸 범주): " + (", ".join(f"{a} → {b} {v}건" for (a, b), v in inc.most_common(8)) or "없음")]
    else:
        L += ["", "규칙을 뺀 조건: 아직 실행 전(예측 파일 없음)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = ["# 실험 1 최종표 — 직접 판정 · 규칙을 뺀 조건 · KoMed-RAG (자동 생성 — report_e1_final.py)", "",
             "판독 LLM Gemini 3.1 Pro. 진실성 = (정답 − 틀린 확신) / n. 범위 초과는 사전 등록 대응대로 지지로 채점하고, 보류로 접은 행을 함께 싣는다. "
             "신뢰구간은 군집 부트스트랩 2,000회(HealthVer 질문 단위, Check-COVID 문헌 단위).", ""]
    for name, cfg in DATASETS.items():
        parts.append(section(name, cfg))
    md = "\n".join(parts)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
