"""실험 3 자리 바꾸기 검사 — 자리를 바꿔도 판정이 뒤집히는가. 저장된 예측만 읽는다(호출 없음).

지표 정의와 서술 규칙은 돌리기 전에 고정했다: claude_backup/실험3_자리바꾸기_사전기록_1116.md
  뒤집기 일치율(주 지표)   자리를 바꾼 뒤의 예측이 원래 예측을 뒤집은 것과 같은 쌍의 비율
  방향 판정 뒤집기 일치율  원래 예측이 과장·축소인 쌍만 놓고 본 같은 비율
  자리 기울기             두 방향을 합친 예측에서 (과장 수 − 축소 수) / (2 × 쌍 수). 내용만 따르면 0
쌍 단위 부트스트랩 2,000회, 시드 0. 합친 지표는 한 쌍의 두 방향을 함께 재표집한다.

사용: python swap_exagg.py --orig preds_exagg_test --swap preds_exagg_swap [--out 파일.md]
"""
import argparse

import numpy as np

import compare_exagg as C

SAME, EXAG, DOWN = (C.LAB.index(x) for x in ("same", "exaggerates", "downplays"))
_MIRROR = np.zeros(len(C.LAB), dtype=int)
_MIRROR[[SAME, EXAG, DOWN]] = [SAME, DOWN, EXAG]


def mirror(codes):
    return _MIRROR[np.asarray(codes)]


def consistency(o, s):
    """뒤집기 일치율(%) — 자리를 바꾼 뒤의 예측 s 가 원래 예측 o 를 뒤집은 것과 같은 쌍."""
    return 100 * float((s == mirror(o)).mean())


def consistency_directional(o, s):
    """원래 예측이 과장·축소인 쌍만 놓고 본 뒤집기 일치율(%). 그런 쌍이 없으면 nan."""
    keep = o != SAME
    return 100 * float((s[keep] == mirror(o[keep])).mean()) if keep.any() else float("nan")


def role_bias(o, s):
    """자리 기울기(%p) — 두 방향을 합쳐 '보도자료 칸이 더 세다'(과장)가 '약하다'(축소)보다 얼마나 많은가."""
    both = np.concatenate([o, s])
    return 100 * float((both == EXAG).sum() - (both == DOWN).sum()) / len(both)


def _ci(vals):
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def pair_ci(stat, o, s, d):
    return (stat(o, s),) + _ci([stat(o[i], s[i]) for i in d])


def pair_diff(stat, a, b, d):
    (ao, as_), (bo, bs) = a, b
    return (stat(ao, as_) - stat(bo, bs),) + _ci([stat(ao[i], as_[i]) - stat(bo[i], bs[i]) for i in d])


def _both(x_or, x_sw, i=None):
    return np.concatenate([x_or, x_sw]) if i is None else np.concatenate([x_or[i], x_sw[i]])


def combined_ci(metric, gold_or, pred_or, gold_sw, pred_sw, d):
    """두 방향을 합친 점수. 재표집은 쌍 단위 — 한 쌍의 두 방향이 함께 뽑힌다."""
    point = metric(_both(gold_or, gold_sw), _both(pred_or, pred_sw))
    return (point,) + _ci([metric(_both(gold_or, gold_sw, i), _both(pred_or, pred_sw, i)) for i in d])


def combined_diff(metric, gold_or, gold_sw, a, b, d):
    (ao, as_), (bo, bs) = a, b
    g = _both(gold_or, gold_sw)
    point = metric(g, _both(ao, as_)) - metric(g, _both(bo, bs))
    vals = [metric(_both(gold_or, gold_sw, i), _both(ao, as_, i)) - metric(_both(gold_or, gold_sw, i), _both(bo, bs, i)) for i in d]
    return (point,) + _ci(vals)


def load_both(dir_or, dir_sw, model=C.MODEL):
    """두 방향의 예측을 같은 id 순서로 맞춘다. 바꾼 쪽 정답이 원래 정답을 뒤집은 것이 아니면 멈춘다."""
    ids_o, gold_o, pred_o = C.aligned({m: C.load(dir_or, m, model) for m in C.MODES})
    ids_s, gold_s, pred_s = C.aligned({m: C.load(dir_sw, m, model) for m in C.MODES})
    if ids_o != ids_s:
        raise ValueError("두 방향의 쌍 id 가 다르다")
    if not (gold_s == mirror(gold_o)).all():
        raise ValueError("자리를 바꾼 쪽의 정답이 원래 정답을 뒤집은 것과 다르다")
    return ids_o, gold_o, pred_o, gold_s, pred_s


def _f(t, signed=False):
    fmt = "{:+.1f}" if signed else "{:.1f}"
    return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"


def render(dir_or, dir_sw, model=C.MODEL, n_boot=2000, seed=0):
    ids, gold_o, pred_o, gold_s, pred_s = load_both(dir_or, dir_sw, model)
    n = len(ids)
    d = C.draws(n, n_boot, seed)
    out = [f"n={n}쌍 (원래 방향 {n} + 자리를 바꾼 방향 {n}). 쌍 단위 부트스트랩 {n_boot:,}회, 시드 {seed}.", "",
           "정답 라벨을 쓰지 않는 지표 [95% CI]", "",
           "| 방식 | 뒤집기 일치율 % | 방향 판정 뒤집기 일치율 % (해당 쌍 수) | 자리 기울기 %p | 예측 분포 같음/과장/축소 (원래 → 바꿈) |",
           "|---|---|---|---|---|"]
    for m in C.MODES:
        o, s = pred_o[m], pred_s[m]
        dist = "/".join(str(int((o == c).sum())) for c in (SAME, EXAG, DOWN)) + " → " + "/".join(str(int((s == c).sum())) for c in (SAME, EXAG, DOWN))
        out.append(f"| {m} | {_f(pair_ci(consistency, o, s, d))} | {_f(pair_ci(consistency_directional, o, s, d))} ({int((o != SAME).sum())}) "
                   f"| {_f(pair_ci(role_bias, o, s, d), signed=True)} | {dist} |")
    out += ["", "정답 라벨을 쓰는 지표 [95% CI] — 바꾼 방향은 뒤집은 정답 대비", "",
            "| 방식 | 과장 F1 원래 | 과장 F1 바꿈 | 과장 F1 합침 | macro-F1 원래 | macro-F1 바꿈 | macro-F1 합침 | 정확도 % 원래 / 바꿈 |",
            "|---|---|---|---|---|---|---|---|"]
    for m in C.MODES:
        o, s = pred_o[m], pred_s[m]
        out.append(f"| {m} | {C.exagg_f1(gold_o, o):.1f} | {C.exagg_f1(gold_s, s):.1f} | {_f(combined_ci(C.exagg_f1, gold_o, o, gold_s, s, d))} "
                   f"| {C.macro_f1(gold_o, o):.1f} | {C.macro_f1(gold_s, s):.1f} | {_f(combined_ci(C.macro_f1, gold_o, o, gold_s, s, d))} "
                   f"| {100 * float((gold_o == o).mean()):.1f} / {100 * float((gold_s == s).mean()):.1f} |")
    out += ["", "짝지은 차이 [95% CI] — 첫 줄이 사전에 고정한 주 비교", "",
            "| 비교 | Δ뒤집기 일치율 %p | Δ방향 판정 뒤집기 일치율 %p | Δ자리 기울기 %p | Δ과장 F1 합침 | Δmacro-F1 합침 |",
            "|---|---|---|---|---|---|"]
    for a, b in C.PAIRS:
        pa, pb = (pred_o[a], pred_s[a]), (pred_o[b], pred_s[b])
        out.append(f"| {a} − {b} | {_f(pair_diff(consistency, pa, pb, d), True)} | {_f(pair_diff(consistency_directional, pa, pb, d), True)} "
                   f"| {_f(pair_diff(role_bias, pa, pb, d), True)} | {_f(combined_diff(C.exagg_f1, gold_o, gold_s, pa, pb, d), True)} "
                   f"| {_f(combined_diff(C.macro_f1, gold_o, gold_s, pa, pb, d), True)} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", required=True, help="원래 방향의 예측 폴더")
    ap.add_argument("--swap", required=True, help="자리를 바꾼 방향의 예측 폴더")
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--out")
    a = ap.parse_args()
    text = render(a.orig, a.swap, a.model)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
