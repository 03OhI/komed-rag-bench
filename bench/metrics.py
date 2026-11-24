#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — 벤치마크 표준 지표와 위험·응답률 분해. 외부 의존 없음.

    from metrics import classification_report3, bootstrap_ci, paired_bootstrap_diff, mcnemar_exact
    rep = classification_report3(golds, preds, labels=("SUPPORT","REFUTE","NEI"))

지표 정의 (논문 5.4절에 그대로 적을 것)
    macro-F1        클래스별 F1 의 단순 평균 (SciFact·HealthVer·CARE-XAI 관행)
    accuracy        전체 정답률
    dangerous flip  gold 가 SUPPORT/REFUTE 인 항목 중 예측이 그 반대(SUPPORT↔REFUTE)인 비율 — '위험한 오류율'
    abstention P/R  NEI 를 '기권'으로 보고 precision = NEI 예측 중 gold 도 NEI, recall = gold NEI 중 NEI 예측 (AbstentionBench 식)
    truthfulness    CRAG-Meta 채점: 정답 +1 · 기권(NEI 예측, gold≠NEI) 0 · 오답(gold≠NEI, 비NEI 오답) −1 · gold=NEI 인데 비NEI 예측 −1 · 평균
    agreement       같은 항목을 k 회 실행했을 때 예측이 전부 같은 항목 비율 (재현성)
    bootstrap CI    항목 재표집(기본 2,000회, seed 0) 백분위 95%
    McNemar         두 조건의 짝지은 정오 비교, 정확 이항 양측

주 검정 (2026-09-26 사전 등록 — 같은 판독 LLM 의 ours vs free, 데이터셋별)
    ① truthfulness  항목별 점수(truth_scores: +1 정답 · 0 기권 · −1 잘못된 확신)의 짝지은 차이
                    → 부호 뒤집기 순열 검정(paired_signflip, 양측) + 짝지은 부트스트랩 95% CI
    ② 위험한 뒤집힘  항목별 발생 여부(flip_flags)의 짝지은 비교 → 정확 McNemar(mcnemar_bool)
    macro-F1 · 정확도(mcnemar_exact)는 표준 지표로 그대로 병기한다. 데이터셋 간 다중 비교는 holm.
negative rejection  gold 가 NEI(근거 없음)인 항목 중 NEI 로 거부한 비율 (RGB) — E6 주 지표
wilson              비율의 Wilson 95% 구간 (표본이 작은 E6 용)
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

LABELS3 = ("SUPPORT", "REFUTE", "NEI")
LABELS4 = ("SUPPORT", "EXCEEDS", "REFUTE", "NEI")


def outcome_summary(golds, preds, labels=LABELS3):
    """ERROR/None are operational failures, never correct NEI or deliberate abstention.

    Rates use successful predictions; utility_attempted assigns failures zero credit.
    NEI is both a task label and withholding a substantive answer. Report both views.
    """
    if len(golds) != len(preds) or any(g not in labels for g in golds):
        raise ValueError("Aligned arrays and valid gold labels are required")
    pairs = [(g, p) for g, p in zip(golds, preds) if p in labels]
    n, attempted = len(pairs), len(golds)
    correct = sum(g == p for g, p in pairs)
    missing = sum(p == "NEI" and g != "NEI" for g, p in pairs)
    wrong = sum(p != "NEI" and g != p for g, p in pairs)
    answered = sum(p != "NEI" for _, p in pairs)
    answer_correct = sum(p != "NEI" and g == p for g, p in pairs)
    answerable = [(g, p) for g, p in pairs if g != "NEI"]
    ratio = lambda k, d: k / d if d else None
    return {"attempted": attempted, "n": n, "errors": attempted - n,
            "correct": correct, "missing": missing, "wrong": wrong,
            "correct_rate": ratio(correct, n), "missing_rate": ratio(missing, n),
            "wrong_rate": ratio(wrong, n), "coverage": ratio(answered, n),
            "answered": answered, "selective_accuracy": ratio(answer_correct, answered),
            "selective_risk": ratio(answered - answer_correct, answered),
            "answerable_n": len(answerable),
            "answerable_correct": ratio(sum(g == p for g, p in answerable), len(answerable)),
            "over_abstention": ratio(sum(p == "NEI" for _, p in answerable), len(answerable)),
            "utility_attempted": ratio(correct - wrong, attempted)}


def aligned_success(a, b, gold_key="gold", pred_key="pred", labels=LABELS3):
    """Return common successful items; validate labels and expose attrition separately."""
    aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Duplicate IDs in a comparison")
    pairs = []
    for key in sorted(aa.keys() & bb.keys()):
        x, y = aa[key], bb[key]
        if x.get(gold_key) != y.get(gold_key):
            raise ValueError(f"Gold mismatch for {key}")
        if (x.get(gold_key) in labels and not x.get("error") and not y.get("error")
                and x.get(pred_key) in labels and y.get(pred_key) in labels):
            pairs.append((x, y))
    return pairs


def paired_outcome_changes(a, b, gold_key="gold", pred_key="pred", labels=LABELS3):
    """a = baseline, b = ours. Distinguish avoided errors from lost correct answers."""
    pairs = aligned_success(a, b, gold_key, pred_key, labels)
    result = {"n": len(pairs), "union_n": len({r['id'] for r in a + b}),
              "wrong_to_abstain": 0, "correct_to_abstain": 0, "wrong_to_correct": 0,
              "correct_to_wrong": 0, "base_correct_non_nei": 0, "retained_correct_non_nei": 0}
    for x, y in pairs:
        g, p, q = x[gold_key], x[pred_key], y[pred_key]
        wrong_p, wrong_q = p != "NEI" and p != g, q != "NEI" and q != g
        result["wrong_to_abstain"] += wrong_p and q == "NEI"
        result["correct_to_abstain"] += p == g and g != "NEI" and q == "NEI"
        result["wrong_to_correct"] += wrong_p and q == g
        result["correct_to_wrong"] += p == g and wrong_q
        result["base_correct_non_nei"] += p == g and g != "NEI"
        result["retained_correct_non_nei"] += p == g == q and g != "NEI"
    return result


def confusion(golds, preds, labels=LABELS3):
    m = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(golds, preds):
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def prf(golds, preds, label):
    tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
    fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
    fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp + fn


def macro_f1(golds, preds, labels=LABELS3):
    fs = [prf(golds, preds, l)[2] for l in labels]
    return sum(fs) / len(fs)


def accuracy(golds, preds):
    return sum(1 for g, p in zip(golds, preds) if g == p) / max(1, len(golds))


def dangerous_flip(golds, preds):
    """gold ∈ {SUPPORT, REFUTE} 중 예측이 반대 극인 비율."""
    opp = {"SUPPORT": "REFUTE", "REFUTE": "SUPPORT"}
    den = [(g, p) for g, p in zip(golds, preds) if g in opp]
    if not den:
        return 0.0, 0
    return sum(1 for g, p in den if p == opp[g]) / len(den), len(den)


def abstention(golds, preds):
    P, R, F, n = prf(golds, preds, "NEI")
    rate = sum(1 for p in preds if p == "NEI") / max(1, len(preds))
    return {"abstain_precision": P, "abstain_recall": R, "abstain_f1": F, "abstain_rate": rate, "n_gold_nei": n}


def truthfulness(golds, preds):
    s = 0.0
    for g, p in zip(golds, preds):
        if g == p:
            s += 1.0
        elif p == "NEI":
            s += 0.0            # 기권 (missing)
        else:
            s -= 1.0            # 잘못된 확신 (hallucination 에 해당)
    return s / max(1, len(golds))


def classification_report3(golds, preds, labels=LABELS3, n_boot=2000, seed=0):
    out = {"n": len(golds), "accuracy": accuracy(golds, preds), "macro_f1": macro_f1(golds, preds, labels),
           "truthfulness": truthfulness(golds, preds), "confusion": confusion(golds, preds, labels)}
    for l in labels:
        P, R, F, n = prf(golds, preds, l)
        out[f"{l}_P"], out[f"{l}_R"], out[f"{l}_F1"], out[f"{l}_n"] = P, R, F, n
    fl, nfl = dangerous_flip(golds, preds)
    out["dangerous_flip"], out["dangerous_flip_n"] = fl, nfl
    out.update(abstention(golds, preds))
    if n_boot:
        out["macro_f1_ci"] = bootstrap_ci(golds, preds, lambda g, p: macro_f1(g, p, labels), n_boot, seed)
        out["accuracy_ci"] = bootstrap_ci(golds, preds, accuracy, n_boot, seed)
    return out


def bootstrap_ci(golds, preds, fn, n_boot=2000, seed=0, alpha=0.05):
    rnd = random.Random(seed)
    n = len(golds)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        vals.append(fn([golds[i] for i in idx], [preds[i] for i in idx]))
    vals.sort()
    return (vals[int(alpha / 2 * n_boot)], vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def paired_bootstrap_diff(golds, preds_a, preds_b, fn, n_boot=2000, seed=0, alpha=0.05):
    """fn(b) − fn(a) 의 항목 재표집 CI (같은 항목을 같이 뽑는다)."""
    rnd = random.Random(seed)
    n = len(golds)
    vals = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        g = [golds[i] for i in idx]
        vals.append(fn(g, [preds_b[i] for i in idx]) - fn(g, [preds_a[i] for i in idx]))
    vals.sort()
    return (vals[int(alpha / 2 * n_boot)], vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def mcnemar_exact(golds, preds_a, preds_b):
    """조건 a 만 정답 (b) vs 조건 b 만 정답 (c) → 정확 이항 양측 p."""
    b = sum(1 for g, x, y in zip(golds, preds_a, preds_b) if x == g and y != g)
    c = sum(1 for g, x, y in zip(golds, preds_a, preds_b) if y == g and x != g)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}



def truth_scores(golds, preds):
    """항목별 truthfulness 점수 (+1 정답 · 0 기권(NEI 예측) · −1 잘못된 확신) — 짝지은 검정용."""
    return [1.0 if g == p else (0.0 if p == "NEI" else -1.0) for g, p in zip(golds, preds)]


def paired_signflip(scores_a, scores_b, n_perm=20000, seed=0):
    """짝지은 차이 d = b − a 의 평균에 대한 부호 뒤집기 순열 검정(양측).
    차이가 0 이 아닌 항목이 16개 이하면 전수 열거(정확), 넘으면 몬테카를로 n_perm 회.
    반환: (평균 차이 b−a, p)."""
    d = [y - x for x, y in zip(scores_a, scores_b)]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    obs = sum(d) / n
    nz = [v for v in d if v != 0]
    if not nz:
        return obs, 1.0
    tot = abs(sum(nz))
    m = len(nz)
    if m <= 16:
        cnt = 0
        for mask in range(1 << m):
            s = sum(v if (mask >> i) & 1 else -v for i, v in enumerate(nz))
            if abs(s) >= tot - 1e-9:
                cnt += 1
        return obs, cnt / (1 << m)
    rnd = random.Random(seed)
    cnt = 0
    for _ in range(n_perm):
        s = sum(v if rnd.random() < 0.5 else -v for v in nz)
        if abs(s) >= tot - 1e-9:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def flip_flags(golds, preds):
    """항목별 위험한 뒤집힘 발생 여부 (gold SUPPORT↔예측 REFUTE 또는 그 반대)."""
    opp = {"SUPPORT": "REFUTE", "REFUTE": "SUPPORT"}
    return [g in opp and p == opp[g] for g, p in zip(golds, preds)]


def mcnemar_bool(flags_a, flags_b):
    """짝지은 이진 사건 비교 — a 만 발생(a_only) vs b 만 발생(b_only), 정확 이항 양측 p."""
    b = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    c = sum(1 for x, y in zip(flags_a, flags_b) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}


def negative_rejection(golds, preds):
    """gold 가 NEI(근거 없음)인 항목 중 NEI 로 거부한 비율 (RGB negative rejection). 반환 (비율 또는 None, 거부 수, 분모)."""
    den = [p for g, p in zip(golds, preds) if g == "NEI"]
    if not den:
        return None, 0, 0
    k = sum(1 for p in den if p == "NEI")
    return k / len(den), k, len(den)


def holm(pvals):
    """Holm–Bonferroni 보정 p (입력 순서 그대로)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i]))
        adj[i] = run
    return adj


def wilson(k, n, z=1.96):
    """비율 k/n 의 Wilson 95% 구간."""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

def agreement(runs_by_id: dict[str, list[str]]):
    """{id: [pred_run1, pred_run2, ...]} → 전부 같은 항목 비율, 다수결 예측."""
    ids = [i for i, r in runs_by_id.items() if len(r) >= 2]
    same = sum(1 for i in ids if len(set(runs_by_id[i])) == 1)
    maj = {i: Counter(r).most_common(1)[0][0] for i, r in runs_by_id.items()}
    return {"n_items": len(ids), "all_same_rate": same / len(ids) if ids else None, "majority": maj}


def cohen_kappa(a, b, weights: str | None = None, levels=None):
    """두 코더의 κ. weights='linear'|'quadratic' 이면 가중 κ (levels 는 정렬된 범주 목록)."""
    if levels is None:
        levels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(levels)}
    k = len(levels)
    n = len(a)
    if n == 0:
        return None
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1
    ra = [sum(O[i]) / n for i in range(k)]
    rb = [sum(O[i][j] for i in range(k)) / n for j in range(k)]
    if weights is None:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        w = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    po = sum(w[i][j] * O[i][j] / n for i in range(k) for j in range(k))
    pe = sum(w[i][j] * ra[i] * rb[j] for i in range(k) for j in range(k))
    return 1 - po / pe if pe else None


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, tuple):
        return f"({x[0]:.{nd}f}, {x[1]:.{nd}f})"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


if __name__ == "__main__":
    g = ["SUPPORT", "REFUTE", "NEI", "SUPPORT", "REFUTE", "NEI", "SUPPORT"]
    p = ["SUPPORT", "SUPPORT", "NEI", "NEI", "REFUTE", "SUPPORT", "SUPPORT"]
    r = classification_report3(g, p, n_boot=200)
    for k, v in r.items():
        if k != "confusion":
            print(k, fmt(v))
    print(r["confusion"])
    print(mcnemar_exact(g, p, ["SUPPORT", "REFUTE", "NEI", "SUPPORT", "REFUTE", "NEI", "REFUTE"]))
    print(cohen_kappa([0, 1, 2, 3, 1], [0, 1, 3, 3, 2], weights="linear", levels=[0, 1, 2, 3]))
