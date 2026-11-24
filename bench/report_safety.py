"""Risk and usefulness must be visible together; no abstain-all winner designation."""
from collections import defaultdict
from metrics import (LABELS3, LABELS4, outcome_summary, paired_outcome_changes,
                     aligned_success, macro_f1, prf, wilson)


def percent(x):
    return "—" if x is None else f"{100*x:.1f}"


def rate_ci(k, n):
    if not n:
        return "— (0/0)"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f} [{100*lo:.1f}, {100*hi:.1f}] ({k}/{n})"


def table_safety(dataset, groups, lines):
    four = dataset == "e6"
    gk, pk, labels = ("gold4", "pred4", LABELS4) if four else ("gold", "pred", LABELS3)
    lines += [f"\n### {dataset} — 위험·응답·유용성 분해 (2026-10-23 v2{' · 4범주' if four else ''})\n",
              "정답(+1)·불필요한 보류(0)·틀린 확신(−1)을 함께 보고한다. API/형식 오류는 별도이며 NEI 정답으로 세지 않는다. "
              "보류만 하는 시스템은 응답률 0, 응답 정밀도 미정이다. 낮은 오류율만으로 우수하다고 결론 내리지 않는다. "
              "아래 비율은 유효 응답 기준; 전체 시도 utility는 오류에 0점을 부여한다. CI는 Wilson 95% 구간.\n",
              "| 모델 | 조건 | 유효/시도 · 오류 | 정답 / 불필요 보류 / 틀린 확신 건 | 틀린 확신 % [CI] | 응답률 % | 응답 중 정답 % [CI] | 답할 수 있는 항목 정답 % | 과잉 보류 % | utility 전체 시도 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    by_llm = defaultdict(dict)
    for (cond, llm), recs in sorted(groups.items()):
        selected = [r for r in recs if r.get(gk) in labels]
        if not selected:
            lines.append(f"| {llm} | {cond} | 0/{len(recs)} · gold{'4' if four else ''} 미등록 | — | — | — | — | — | — | — |")
            continue
        g = [r[gk] for r in selected]
        p = ["ERROR" if r.get("error") else r.get(pk) for r in selected]
        s = outcome_summary(g, p, labels)
        answer_correct = sum(x == y and y != "NEI" for x, y in zip(g, p))
        u = "—" if s["utility_attempted"] is None else f"{s['utility_attempted']:+.3f}"
        lines.append(f"| {llm} | {cond} | {s['n']}/{s['attempted']} · {s['errors']} | "
                     f"{s['correct']} / {s['missing']} / {s['wrong']} | {rate_ci(s['wrong'], s['n'])} | "
                     f"{percent(s['coverage'])} | {rate_ci(answer_correct, s['answered'])} | "
                     f"{percent(s['answerable_correct'])} | {percent(s['over_abstention'])} | {u} |")
        by_llm[llm][cond] = selected
    if four:
        lines += ["\n| 모델 | 조건 | 4범주 macro-F1 | EXCEEDS P / R / F1 | 실제 과장을 SUPPORT로 놓침 (k/n) |",
                  "|---|---|---|---|---|"]
        for llm, conds in sorted(by_llm.items()):
            for cond, recs in sorted(conds.items()):
                valid = [r for r in recs if not r.get("error") and r.get(pk) in labels]
                if not valid:
                    continue
                g, p = [r[gk] for r in valid], [r[pk] for r in valid]
                pp, rr, ff, n = prf(g, p, "EXCEEDS")
                missed = sum(x == "EXCEEDS" and y == "SUPPORT" for x, y in zip(g, p))
                lines.append(f"| {llm} | {cond} | {macro_f1(g,p,labels):.3f} | {pp:.3f} / {rr:.3f} / {ff:.3f} | {missed}/{n} |")
    lines += ["\n**같은 항목의 변화 — 개선과 잃은 정답을 모두 표시**\n",
              "| 모델 | 비교 (후자−전자) | 공통 유효/합집합 | 틀린 확신→보류 | 정답→불필요 보류 | 틀린 확신→정답 | 정답→틀린 확신 | 기준선 실질 정답 보존 (k/n) |",
              "|---|---|---|---|---|---|---|---|"]
    comparisons = [("free", "ours"), ("stdrag", "ours"), ("stdrag", "bm25rules"),
                   ("gatefree", "gaterules"), ("gaterules", "ours"), ("stdrag", "gatefree"),
                   ("bm25rules", "gaterules"), ('ours', 'oursv3'), ('stdragv3', 'oursv3'),
                   ('gatefreev3', 'oursv3')]
    for llm, conds in sorted(by_llm.items()):
        for base, ours in comparisons:
            if base not in conds or ours not in conds:
                continue
            s = paired_outcome_changes(conds[base], conds[ours], gk, pk, labels)
            lines.append(f"| {llm} | {base} → {ours} | {s['n']}/{s['union_n']} | {s['wrong_to_abstain']} | "
                         f"{s['correct_to_abstain']} | {s['wrong_to_correct']} | {s['correct_to_wrong']} | "
                         f"{s['retained_correct_non_nei']}/{s['base_correct_non_nei']} |")
    lines += ["\n**인용 감사 — 원문 존재와 의미적 뒷받침은 다른 지표**\n",
              "| 모델 | 조건 | 인용 있는 응답/유효 응답 | 원문 존재 (k/n 인용) | 의미적 인용 지지·재현율 |",
              "|---|---|---|---|---|"]
    for (cond, llm), recs in sorted(groups.items()):
        if cond == "closed":
            continue
        ok = [r for r in recs if not r.get("error") and r.get(pk) in labels]
        quoted = [r for r in ok if any(str(q or "").strip() for q in r.get("key_sentences", []))]
        vb = [bool(v) for r in quoted for q, v in zip(r.get("key_sentences", []), r.get("verbatim", [])) if str(q or "").strip()]
        lines.append(f"| {llm} | {cond} | {len(quoted)}/{len(ok)} | {sum(vb)}/{len(vb)} | 별도 사람/NLI 평가 필요 |")
    if four:
        lines.append("\n4범주 gold 없는 과거 실행은 재해석해 채점하지 않는다. 3범주 표는 호환용 보조표다. "
                     "E6 sec는 검색 사전계산 후 판정 시간이며 전체 파이프라인 지연이 아니다.")
