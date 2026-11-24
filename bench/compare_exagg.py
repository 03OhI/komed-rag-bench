"""실험 3(과장 탐지) — 방식별 점수와 방식 간 짝지은 차이. 저장된 예측만 읽는다(호출 없음).

주 비교는 실행 전에 고정한 ours4 − direct 다(claude_backup/실험3_주조건_기록_1116.md).
쌍 단위 부트스트랩 2,000회, 시드 0. 같은 재표집을 모든 방식에 쓴다(짝지은 비교).

사용: python compare_exagg.py --dir preds_exagg_test [--out 파일.md]
"""
import argparse
import json
import os

import numpy as np

LAB = ("same", "exaggerates", "downplays")
MODEL = "gemini-gemini-3.1-pro-preview"
MODES = ("direct", "directreview", "ours4", "ours7")
PAIRS = (("ours4", "direct"), ("ours7", "direct"), ("ours4", "directreview"), ("ours7", "ours4"))
STRENGTH = (("pr_strength", "s1_4"), ("abs_strength", "s2_4"))   # (정답, 판독) — 보도자료 · 초록


def f1_class(gold, pred, c):
    tp = int(((gold == c) & (pred == c)).sum())
    fp = int(((gold != c) & (pred == c)).sum())
    fn = int(((gold == c) & (pred != c)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def exagg_f1(gold, pred):
    return 100 * f1_class(gold, pred, LAB.index("exaggerates"))


def macro_f1(gold, pred):
    return 100 * float(np.mean([f1_class(gold, pred, c) for c in range(len(LAB))]))


def draws(n, n_boot=2000, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def point_ci(gold, pred, metric, d):
    vals = [metric(gold[i], pred[i]) for i in d]
    return metric(gold, pred), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def diff_ci(gold, pa, pb, metric, d):
    vals = [metric(gold[i], pa[i]) - metric(gold[i], pb[i]) for i in d]
    return metric(gold, pa) - metric(gold, pb), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load(pred_dir, mode, model=MODEL):
    out = {}
    with open(os.path.join(pred_dir, f"exagg_{mode}_{model}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r                                   # 같은 id 가 여러 번이면 마지막 기록
    return out


def _encode(values, what):
    bad = sorted({str(v) for v in values if v not in LAB})
    if bad:
        raise ValueError(f"{what}: 세 범주 밖의 값 {bad}")
    return np.array([LAB.index(v) for v in values])


def aligned(preds):
    """방식별 {id: 기록} 을 같은 id 순서의 배열로 맞춘다. 어긋나면 멈춘다."""
    modes = list(preds)
    ids = sorted(preds[modes[0]])
    gold_txt = [preds[modes[0]][i]["gold"] for i in ids]
    for m in modes:
        if set(preds[m]) != set(ids):
            raise ValueError(f"{m}: 쌍 id 가 다른 방식과 다르다")
        if [preds[m][i]["gold"] for i in ids] != gold_txt:
            raise ValueError(f"{m}: 정답 라벨이 다른 방식과 다르다")
    gold = _encode(gold_txt, "gold")
    pred = {m: _encode([preds[m][i]["pred"] for i in ids], f"{m} pred") for m in modes}
    return ids, gold, pred


def strength_scores(records, gold_key, pred_key, n_level=4):
    """인과 강도 판독의 (macro-F1, 정확도). 정답에도 판독에도 없는 단계는 평균에서 뺀다."""
    g = np.array([r[gold_key] for r in records])
    p = np.array([r[pred_key] for r in records])
    f1s = [f1_class(g, p, c) for c in range(n_level) if int((g == c).sum() + (p == c).sum()) > 0]
    return 100 * float(np.mean(f1s)), 100 * float((g == p).mean())


def rule_label(pr, ab):
    """판정 규칙 — 보도자료 강도가 초록보다 크면 과장, 작으면 축소, 같으면 같음."""
    return "exaggerates" if pr > ab else "downplays" if pr < ab else "same"


def _has_reading(records):
    return all(k in r for r in records for pair in STRENGTH for k in pair)


def attribution(records):
    """규칙 방식의 오답을 어느 문장의 강도 오독에서 나왔는지로 나눈다(4단계 판독 기준)."""
    a = dict.fromkeys(("n", "errors", "err_pr_only", "err_abs_only", "err_both", "err_neither",
                       "correct_despite_misreading", "both_read_correctly", "replay_match", "gold_from_strengths",
                       "pr_read_stronger", "pr_read_weaker", "abs_read_stronger", "abs_read_weaker"), 0)
    for r in records:
        pg, ag, ps, as_ = r["pr_strength"], r["abs_strength"], r["s1_4"], r["s2_4"]
        pr_bad, ab_bad = ps != pg, as_ != ag
        wrong = r["pred"] != r["gold"]
        a["n"] += 1
        a["errors"] += wrong
        a["replay_match"] += rule_label(ps, as_) == r["pred"]            # 저장된 판독에 규칙을 다시 적용하면 예측이 나오는가
        a["gold_from_strengths"] += rule_label(pg, ag) == r["gold"]      # 정답 라벨이 강도 주석을 견준 값인가
        a["both_read_correctly"] += not (pr_bad or ab_bad)
        a["pr_read_stronger"] += ps > pg
        a["pr_read_weaker"] += ps < pg
        a["abs_read_stronger"] += as_ > ag
        a["abs_read_weaker"] += as_ < ag
        if wrong:
            a["err_both" if pr_bad and ab_bad else "err_pr_only" if pr_bad else "err_abs_only" if ab_bad else "err_neither"] += 1
        elif pr_bad or ab_bad:
            a["correct_despite_misreading"] += 1
    return {k: int(v) for k, v in a.items()}


def render(pred_dir, model=MODEL, n_boot=2000, seed=0):
    preds = {m: load(pred_dir, m, model) for m in MODES}
    ids, gold, pred = aligned(preds)
    d = draws(len(ids), n_boot, seed)
    counts = " · ".join(f"{LAB[c]} {int((gold == c).sum())}" for c in range(len(LAB)))
    out = [f"n={len(ids)} (정답 {counts}). 쌍 단위 부트스트랩 {n_boot:,}회, 시드 {seed}.", "",
           "| 방식 | 과장 F1 [95% CI] | 세 범주 macro-F1 [95% CI] | 정확도 % | 강도 4단계 macro-F1 (보도자료 / 초록) |",
           "|---|---|---|---|---|"]
    for m in MODES:
        e = point_ci(gold, pred[m], exagg_f1, d)
        f = point_ci(gold, pred[m], macro_f1, d)
        recs = [preds[m][i] for i in ids]
        s = " / ".join("%.1f" % strength_scores(recs, gk, pk)[0] for gk, pk in STRENGTH) if _has_reading(recs) else "-"
        out.append(f"| {m} | {e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}] | {f[0]:.1f} [{f[1]:.1f}, {f[2]:.1f}] "
                   f"| {100 * float((gold == pred[m]).mean()):.1f} | {s} |")
    out += ["", "| 비교 | Δ과장 F1 [95% CI] | Δ세 범주 macro-F1 [95% CI] |", "|---|---|---|"]
    for a, b in PAIRS:
        e = diff_ci(gold, pred[a], pred[b], exagg_f1, d)
        f = diff_ci(gold, pred[a], pred[b], macro_f1, d)
        out.append(f"| {a} − {b} | {e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}] | {f[0]:+.1f} [{f[1]:+.1f}, {f[2]:+.1f}] |")
    out += ["", "오답의 판독 귀속 — 강도 판독을 남기는 방식만 (4단계 판독 기준, 건)", "",
            "| 방식 | 오답 | 보도자료만 오독 | 초록만 오독 | 둘 다 오독 | 둘 다 맞게 읽고 틀림 | 오독했지만 정답 | 둘 다 맞게 읽은 쌍 "
            "| 판독으로 예측 재현 | 정답 = 강도 주석 비교 | 보도자료 세게 / 약하게 | 초록 세게 / 약하게 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        recs = [preds[m][i] for i in ids]
        if not _has_reading(recs):
            continue
        a = attribution(recs)
        out.append(f"| {m} | {a['errors']} | {a['err_pr_only']} | {a['err_abs_only']} | {a['err_both']} | {a['err_neither']} "
                   f"| {a['correct_despite_misreading']} | {a['both_read_correctly']} | {a['replay_match']}/{a['n']} "
                   f"| {a['gold_from_strengths']}/{a['n']} | {a['pr_read_stronger']} / {a['pr_read_weaker']} "
                   f"| {a['abs_read_stronger']} / {a['abs_read_weaker']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="preds_exagg_test")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out")
    a = ap.parse_args()
    text = render(a.dir, a.model)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
