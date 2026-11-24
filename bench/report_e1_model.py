"""실험 1 — 판독 LLM 을 바꾼 HealthVer test 표 (2026-11-16, GPT-4o 추가 실행용).

지표·부트스트랩은 report_e1_final.py 의 함수를 그대로 쓴다. 사전 기록: claude_backup/실험1_GPT4o_사전기록_1116.md.
남은 형식 오류(pred 가 세 범주 밖)는 근거 불충분(보류)으로 채점하고 건수를 적는다(사전 기록 §5).

    python report_e1_model.py --dir preds_e1_gpt4o --model openai-gpt-4o-2024-08-06 --out ../claude_backup/실험1_GPT4o_결과_1116.md
"""
from __future__ import annotations

import argparse
import json
import os

import report_e1_final as R

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data/healthver/data/healthver_test.csv")
KEYS = (("truth", "{:+.3f}"), ("mf1", "{:+.1f}"), ("acc", "{:+.1f}"), ("wrong", "{:+.0f}"), ("correct", "{:+.0f}"), ("flip_rate", "{:+.1f}"))


def read(path: str):
    """id 별 마지막 기록. 세 범주 밖의 예측은 NEI 로 바꾸고 그 건수를 함께 돌려준다."""
    out, n_err = {}, 0
    if not os.path.isfile(path):
        return {}, 0
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    for r in out.values():
        if r.get("pred") not in R.LAB:
            r["pred"], r["category"] = "NEI", "근거 없음"
            n_err += 1
    return out, n_err


def render(pred_dir: str, model: str) -> str:
    conds, errs = {}, {}
    for c in ("free", "norule", "ours"):
        conds[c], errs[c] = read(os.path.join(HERE, pred_dir, f"healthver_{c}_{model}.jsonl"))
    present = [c for c in ("free", "norule", "ours") if conds[c]]
    ids = list(conds["free"])
    for c in present:
        if set(conds[c]) != set(ids):
            raise ValueError(f"{c} 의 id 가 직접 판정과 다름 ({len(set(conds[c]) ^ set(ids))}건)")
    golds = [conds["free"][i]["gold"] for i in ids]
    q = R._hv_questions(CSV)
    clusters = [q[i] for i in ids]
    P = {(c, e): [R.label(conds[c][i], e) for i in ids] for c in present for e in (False, True)}
    Mx = {k: R.metrics(golds, v) for k, v in P.items()}
    L = [f"## HealthVer test · {model} — n={len(ids)} (질문 {len(set(clusters))}개)", "",
         "남은 형식 오류(보류로 채점): " + ", ".join(f"{R.NAMES[c]} {errs[c]}건" for c in present), "",
         "| 조건 | 진실성 | 정답 % | 불필요한 보류 % | 틀린 확신 % (건) | 답한 수 (응답률 %) | 답한 것 중 정답 % | 위험한 뒤집힘 % (건) | 근거 불충분 거부율 % | 정확도 % | macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(tag, m):
        n = m["n"]
        return (f"| {tag} | {m['truth']:+.3f} | {100 * m['correct'] / n:.1f} | {100 * m['missed'] / n:.1f} | {100 * m['wrong'] / n:.1f} ({m['wrong']}) | "
                f"{m['answered']} ({m['coverage']:.1f}) | {R._f(m['prec'])} | {R._f(m['flip_rate'])} ({m['flips']}) | {R._f(m['nei_reject'])} | {m['acc']:.1f} | {m['mf1']:.1f} |")
    for c in present:
        L.append(row(R.NAMES[c], Mx[(c, False)]))
    for c in present:
        if c != "free":
            L.append(row(f"{R.NAMES[c]} · 범위 초과를 보류로 접음", Mx[(c, True)]))
    L += ["", "짝지은 차이 [95% 신뢰구간] — 첫 줄이 사전에 고정한 주 비교", "",
          "| 비교 | Δ진실성 | Δmacro-F1 | Δ정확도 | Δ틀린 확신 (건) | Δ정답 (건) | Δ위험한 뒤집힘 (%p) |", "|---|---|---|---|---|---|---|"]
    pairs = [(a, b, e) for a, b, e in (("ours", "free", False), ("ours", "norule", False), ("norule", "free", False), ("ours", "free", True))
             if a in present and b in present]
    for a, b, e in pairs:
        cells = []
        for key, fmt in KEYS:
            pt = Mx[(a, e)][key] - Mx[(b, False)][key]
            lo, hi = R.boot_diff(golds, P[(a, e)], P[(b, False)], clusters, lambda m, k=key: m[k])
            cells.append(f"{fmt.format(pt)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        L.append(f"| {R.NAMES[a]}{'(범위 초과를 보류로 접음)' if e else ''} − {R.NAMES[b]} | " + " | ".join(cells) + " |")
    if "norule" in present:
        recs = [conds["norule"][i] for i in ids]
        L += ["", f"판독–판정 일치율(규칙을 뺀 조건): {R.consistency(recs):.1f}%"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    text = render(a.dir, a.model)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
