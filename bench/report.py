#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — preds/*.jsonl 을 모아 논문용 표(markdown)로. 문헌 공개 수치를 참조 행으로 같이 싣는다.

    python report.py                 # preds/ 전체 → report_bench.md (+ stdout)
    python report.py --preds preds --out report_bench.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from protocol import DEFAULT_PREDS
from report_safety import table_safety
from report_candidate import table_candidate

from metrics import (classification_report3, macro_f1, accuracy, paired_bootstrap_diff, mcnemar_exact,
                     agreement, cohen_kappa, prf, bootstrap_ci, fmt,
                     truthfulness, truth_scores, paired_signflip, flip_flags, mcnemar_bool, dangerous_flip,
                     negative_rejection, holm, wilson)

PRIMARY = []    # (dataset, llm, 대조, p_truthfulness, p_flip) — 주 검정 Holm 요약용 (2026-09-26 사전 등록)

HERE = os.path.dirname(os.path.abspath(__file__))

# 문헌 공개 수치 (근거: 벤치마크_베이스라인_조사_0915.md · RAG평가_벤치마크_베이스라인_조사_0915.md)
REFERENCE = {
    "scifact": [("MultiVerS (Longformer-large, test, abstract label+rationale F1)", "72.5"),
                ("GPT-4o zero-shot (쌍 단위 dev, macro-F1; arXiv 2606.12854)", "85.6"),
                ("GPT-5 zero-shot (동일)", "77.9"), ("Mistral-7B QLoRA (동일)", "88.4"), ("BioLinkBERT (동일)", "87.5")],
    "healthver": [("T5-base (test F1)", "79.6"), ("SciBERT (test F1)", "77.2"), ("BioLinkBERT (macro-F1; 2606.12854)", "81.9"),
                  ("GPT-4o zero-shot (macro-F1)", "53.2"), ("GPT-5 zero-shot (macro-F1)", "42.4")],
    "checkcovid": [("파이프라인(Vespa 검색, composed/extracted macro-F1)", "76.99 / 63.14"),
                   ("GPT-3.5 (composed/extracted macro-F1)", "69.55 / 60.43")],
    "exagg": [("MT-PET T1 직접 분류 (macro-F1, test 553)", "47.35"), ("MT-PET T2 강도 예측→규칙 (200-shot)", "57.44"),
              ("MT-PET T2 (4,500)", "61.11"), ("강도 예측 F1 (보도자료 / 초록)", "64.72 / 63.27")],
    "causal": [("BioBERT 미세조정 (macro-F1, Yu 2019)", "0.881"), ("BERT (Yu 2020, 5-fold)", "0.895"),
               ("ChatGPT zero-shot (+CoT, WASSA 2023)", "0.63–0.77")],
}
EXAGG_LABELS = ("same", "exaggerates", "downplays")
CAUSAL_LABELS = ("no_relationship", "correlational", "conditional_causal", "direct_causal")


def read_preds(path):
    by_id = {}
    protocols = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            protocols.add(r.get("run_fingerprint", "legacy"))
            if len(protocols) > 1:
                raise ValueError(f"Mixed protocols in {path}")
            if r.get("error"):
                r["pred"] = "ERROR"
                if "pred4" in r:
                    r["pred4"] = "ERROR"
            by_id[r["id"]] = r          # id 별 마지막 기록
    return list(by_id.values())


def parse_name(fn):
    m = re.match(r"(scifact|healthver|checkcovid|exagg|causal|e6|negtest|negpilot)_(\w+?)_(.+?)(?:_r(\d+))?\.jsonl$", os.path.basename(fn))
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4) or 0)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def table_verdict(dataset, groups, lines):
    lines.append(f"\n### {dataset} — 판정 계층 (E1)\n")
    lines.append("| 시스템 | 조건 | n | macro-F1 [95% CI] | Acc | F1 S / R / NEI | 위험한 뒤집힘 % | 기권 P / R | truthfulness | 인용 원문 존재 % | 인용≈정답 근거 % | 범위 초과 건 | 오류 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    preds_by = {}
    for (cond, llm), recs in sorted(groups.items()):
        err = sum(1 for r in recs if r.get("error"))
        recs = [r for r in recs if not r.get("error") and r.get("pred") in ("SUPPORT", "REFUTE", "NEI")]
        if not recs:
            lines.append(f"| {llm} | {cond} | 0 | — | | | | | | | | | {err} |")
            continue
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        rep = classification_report3(g, p)
        vb = [v for r in recs for v in (r.get("verbatim") or [])]
        rat = [r for r in recs if r.get("gold_rationale_available") and r["gold"] != "NEI"]
        ch = sum(1 for r in rat if r.get("cite_hit")) / len(rat) if rat else None
        ov = sum(1 for r in recs if r.get("category") == "범위 초과")
        lines.append(f"| {llm} | {cond} | {rep['n']} | {100*rep['macro_f1']:.1f} [{100*rep['macro_f1_ci'][0]:.1f}, {100*rep['macro_f1_ci'][1]:.1f}] | "
                     f"{100*rep['accuracy']:.1f} | {100*rep['SUPPORT_F1']:.1f} / {100*rep['REFUTE_F1']:.1f} / {100*rep['NEI_F1']:.1f} | "
                     f"{100*rep['dangerous_flip']:.1f} | {100*rep['abstain_precision']:.1f} / {100*rep['abstain_recall']:.1f} | "
                     f"{rep['truthfulness']:+.3f} | {pct(sum(vb)/len(vb) if vb else None)} | {pct(ch)} | {ov if cond=='ours' else '-'} | {err} |")
        preds_by[(cond, llm)] = (recs, g, p)
    for name, val in REFERENCE.get(dataset, []):
        lines.append(f"| (문헌) {name} | — | — | {val} | | | | | | | | |")
    # 예행 점검 (2026-09-26) — ours 가 '비교도 못할 만큼' 낮아지는 원인 네 가지를 먼저 본다
    diag = [(llm, recs) for (cond, llm), recs in sorted(groups.items()) if cond == "ours" and recs]
    if diag:
        lines.append("\n**점검 — ours (오류 · 정답 근거를 무관으로 버림 · 과잉 기권)**\n")
        lines.append("| LLM | n | 오류 % | 정답 지지·반박인데 관련 문헌 0건 % | NEI 예측 % / 정답 NEI % | 근거 없음 사유 (not_found · inconclusive · tied · error) |")
        lines.append("|---|---|---|---|---|---|")
        for llm, recs in diag:
            n_ = len(recs)
            sr = [r for r in recs if r["gold"] in ("SUPPORT", "REFUTE")]
            dropped = sum(1 for r in sr if not r.get("error") and (r.get("n_relevant") or 0) == 0)
            reasons = {k: sum(1 for r in recs if r.get("nei_reason") == k) for k in ("not_found", "inconclusive", "tied", "error")}
            lines.append(f"| {llm} | {n_} | {pct(sum(1 for r in recs if r.get('error')) / n_)} | "
                         f"{pct(dropped / len(sr) if sr else None)} ({dropped}/{len(sr)}) | "
                         f"{pct(sum(1 for r in recs if r['pred'] == 'NEI') / n_)} / {pct(sum(1 for r in recs if r['gold'] == 'NEI') / n_)} | "
                         f"{reasons['not_found']} · {reasons['inconclusive']} · {reasons['tied']} · {reasons['error']} |")
    # 짝지은 비교: 같은 LLM 의 ours vs free / closed
    pairs = []
    for (cond, llm) in preds_by:
        if cond == "ours":
            for other in ("free", "closed"):
                if (other, llm) in preds_by:
                    pairs.append((llm, other))
    if pairs:
        lines.append(f"\n**같은 LLM 안에서의 짝지은 비교 (ours − {'/'.join(sorted({o for _, o in pairs}))})** — "
                     "기존 검정(2026-09-26 계획; 항목 독립 가정의 탐색 분석): ① truthfulness 짝지은 차이(부호 뒤집기 순열 검정) "
                     "② 위험한 뒤집힘 발생의 짝지은 McNemar. macro-F1·정확도는 표준 지표로 병기.\n")
        lines.append("| LLM | 대조 | n | ① Δtruthfulness [95% CI] · p | ② Δ위험한 뒤집힘 %p (ours만 : 대조만) · p | "
                     "Δmacro-F1 [95% CI] | ΔAcc · McNemar 정오(ours만 : 대조만) p |")
        lines.append("|---|---|---|---|---|---|---|")
        for llm, other in pairs:
            ra, ga, pa = preds_by[("ours", llm)]
            rb, gb, pb = preds_by[(other, llm)]
            ida = {r["id"]: r["pred"] for r in ra}
            gold_a = {r["id"]: r["gold"] for r in ra}
            common = [r for r in rb if r["id"] in ida]
            if any(gold_a[r['id']] != r['gold'] for r in common):
                raise ValueError('Gold differs across paired E1 conditions')
            if not common:
                continue
            g = [r["gold"] for r in common]
            po = [ida[r["id"]] for r in common]
            pc = [r["pred"] for r in common]
            dt, pt = paired_signflip(truth_scores(g, pc), truth_scores(g, po))
            dci = paired_bootstrap_diff(g, pc, po, truthfulness)
            mf = mcnemar_bool(flip_flags(g, po), flip_flags(g, pc))
            dfl = dangerous_flip(g, po)[0] - dangerous_flip(g, pc)[0]
            d = paired_bootstrap_diff(g, pc, po, macro_f1)
            mc = mcnemar_exact(g, pc, po)
            PRIMARY.append((dataset, llm, other, pt, mf["p"]))
            lines.append(f"| {llm} | {other} | {len(g)} | {dt:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] · p={pt:.4f} | "
                         f"{100*dfl:+.1f} ({mf['a_only']} : {mf['b_only']}) · p={mf['p']:.4f} | "
                         f"{100*(macro_f1(g, po)-macro_f1(g, pc)):+.1f} [{100*d[0]:+.1f}, {100*d[1]:+.1f}] | "
                         f"{100*(accuracy(g, po)-accuracy(g, pc)):+.1f} · {mc['b_only']} : {mc['a_only']} → p={mc['p']:.4f} |")


def table_primary_holm(lines):
    """주 검정 p 를 같은 LLM·같은 대조 안에서 데이터셋 간 Holm 보정 (검정마다 따로)."""
    if not PRIMARY:
        return
    lines.append("\n**기존 탐색 검정의 다중 비교 보정 — Holm (같은 LLM·같은 대조 안에서 데이터셋 간, 검정별)**\n")
    lines.append("| LLM | 대조 | 데이터셋 | ① truthfulness p → Holm | ② 위험한 뒤집힘 p → Holm |")
    lines.append("|---|---|---|---|---|")
    groups = defaultdict(list)
    for ds, llm, other, pt, pf in PRIMARY:
        groups[(llm, other)].append((ds, pt, pf))
    for (llm, other), rows in sorted(groups.items()):
        ht, hf = holm([r[1] for r in rows]), holm([r[2] for r in rows])
        for (ds, pt, pf), a1, a2 in zip(rows, ht, hf):
            lines.append(f"| {llm} | {other} | {ds} | {pt:.4f} → {a1:.4f} | {pf:.4f} → {a2:.4f} |")


E6_COND = {"closed": "LLM-only (문헌 없음)", "stdrag": "Standard RAG (게이트 없는 BM25 상위 8)",
           "ours": "우리 (게이트 → 판독 → 규칙)", "bm25rules": "BM25 + 판독·규칙",
           "gatefree": "MeSH 게이트 + 자유 판정", "gaterules": "MeSH 게이트 + 규칙 (평가기 없음)",
           "stdragv3": "BM25 + 개선 초록·직접 판정", "gatefreev3": "게이트 + 개선 초록·직접 판정",
           "oursv3": "우리 v3 (근거 검증·단일 연구 범위)"}


def table_e6(groups, lines):
    lines.append("\n### E6 — 3범주 호환 보조표 (EXCEEDS→SUPPORT; 주 결과는 위 4범주 표)\n")
    lines.append("이전 프로토콜과의 호환 지표: truthfulness · 근거 없음 거부율. 정답이 부분·논란·빈칸인 주장은 채점에서 빼고 "
                 "주장별 표에만 둔다. 표본이 작아 p 값 없이 95% CI 만(truthfulness 는 부트스트랩, 비율은 Wilson).\n")
    lines.append("| LLM | 조건 | 채점 n | truthfulness [95% CI] | 정확도 [95% CI] | 근거 없음 거부율 (k/n) | 위험한 뒤집힘 % | 오류 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    by_llm = defaultdict(dict)
    for (cond, llm), recs in groups.items():
        by_llm[llm][cond] = recs
    scored = lambda recs: [r for r in recs if r.get("gold") in ("SUPPORT", "REFUTE", "NEI") and not r.get("error") and r.get("pred") in ("SUPPORT", "REFUTE", "NEI")]
    for llm, conds in sorted(by_llm.items()):
        for cond in E6_COND:
            recs = conds.get(cond)
            if not recs:
                continue
            sc = scored(recs)
            g, p = [r["gold"] for r in sc], [r["pred"] for r in sc]
            err = sum(1 for r in recs if r.get("error"))
            if not g:
                lines.append(f"| {llm} | {E6_COND.get(cond, cond)} | 0 | - | - | - | - | {err} |")
                continue
            tci = bootstrap_ci(g, p, truthfulness)
            k = sum(1 for a_, b_ in zip(g, p) if a_ == b_)
            aci = wilson(k, len(g))
            nr, nk, nn = negative_rejection(g, p)
            nr_txt = "-" if nr is None else f"{100 * nr:.0f}% ({nk}/{nn})"
            lines.append(f"| {llm} | {E6_COND.get(cond, cond)} | {len(g)} | {truthfulness(g, p):+.3f} [{tci[0]:+.3f}, {tci[1]:+.3f}] | "
                         f"{100 * k / len(g):.1f} [{100 * aci[0]:.1f}, {100 * aci[1]:.1f}] | {nr_txt} | "
                         f"{100 * dangerous_flip(g, p)[0]:.1f} | {err} |")
        if "ours" in conds:
            ido = {r["id"]: r for r in scored(conds["ours"])}
            for other in ("closed", "stdrag"):
                if other not in conds:
                    continue
                common = [r for r in scored(conds[other]) if r["id"] in ido]
                if any(ido[r['id']]['gold'] != r['gold'] for r in common):
                    raise ValueError('Gold differs across paired E6 conditions')
                if not common:
                    continue
                g = [r["gold"] for r in common]
                pc = [r["pred"] for r in common]
                po = [ido[r["id"]]["pred"] for r in common]
                d = paired_bootstrap_diff(g, pc, po, truthfulness)
                lines.append(f"| {llm} | ours − {E6_COND[other].split(' (')[0]} | {len(g)} | "
                             f"Δ {truthfulness(g, po) - truthfulness(g, pc):+.3f} [{d[0]:+.3f}, {d[1]:+.3f}] | | | | |")
    for llm, conds in sorted(by_llm.items()):
        ids = sorted({r["id"] for recs in conds.values() for r in recs})
        byc = {c: {r["id"]: r for r in recs} for c, recs in conds.items()}
        cols = [c for c in E6_COND if c in byc]
        lines.append(f"\n주장별 판정 — {llm} (기대 범주 → 조건별 예측; ours 는 범주·보류 사유)\n")
        lines.append("| id | 기대 | 정답 라벨 | " + " | ".join(E6_COND[c].split(" (")[0] for c in cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(cols))
        for i_ in ids:
            any_r = next(byc[c][i_] for c in cols if i_ in byc[c])
            cells = []
            for c in cols:
                r = byc[c].get(i_)
                if not r:
                    cells.append("")
                elif c == "ours":
                    cells.append(f"{r.get('category') or r['pred']}" + (f"({r['nei_reason']})" if r.get("nei_reason") else ""))
                else:
                    cells.append(r["pred"] + (" ⚠" if r.get("error") else ""))
            lines.append(f"| {i_} | {any_r.get('expected_raw', '')} | {any_r.get('gold') or '제외'} | " + " | ".join(cells) + " |")

def table_exagg(groups, lines):
    lines.append("\n### 과장 탐지 — Wright & Augenstein 2021 (E2)\n")
    lines += ["| 모델 | 방식 | 과장 P / R / F1 | 비과장→과장 오탐 (k/n) | 실제 과장 놓침 (k/n) |",
              "|---|---|---|---|---|"]
    for (mode, llm), rows in sorted(groups.items()):
        rows = [r for r in rows if not r.get('error') and r.get('pred') in EXAGG_LABELS]
        g, p = [r['gold'] for r in rows], [r['pred'] for r in rows]
        pp, rr, ff, n = prf(g, p, 'exaggerates')
        fp = sum(x != 'exaggerates' and y == 'exaggerates' for x, y in zip(g, p))
        fn = sum(x == 'exaggerates' and y != 'exaggerates' for x, y in zip(g, p))
        lines.append(f"| {llm} | {mode} | {pp:.3f} / {rr:.3f} / {ff:.3f} | {fp}/{len(g)-n} | {fn}/{n} |")
    lines.append("\n기존 3분류와 7단계 문턱 민감도를 모두 유지한다. test 결과로 유리한 문턱을 선택하지 않는다.\n")
    lines.append("| 시스템 | 방식 | n | macro-F1 3분류 [95% CI] | Acc | F1 same / exagg / down | 강도 코딩 Acc (보도자료 / 초록) | 가중 κ (보도자료 / 초록) | 7단계·2단계 초과 규칙 macro-F1 | 오류 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (mode, llm), recs in sorted(groups.items()):
        err = sum(bool(r.get("error")) for r in recs)
        recs = [r for r in recs if not r.get("error") and r.get("pred") in EXAGG_LABELS]
        if not recs:
            lines.append(f"| {llm} | {mode} | 0 | — | | | | | | {err} |")
            continue
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        mf = macro_f1(g, p, EXAGG_LABELS)
        ci = bootstrap_ci(g, p, lambda a, b: macro_f1(a, b, EXAGG_LABELS))
        f = [prf(g, p, l)[2] for l in EXAGG_LABELS]
        sa = ka = "-"
        if all("s1_4" in r for r in recs):
            s1 = [r["s1_4"] for r in recs]; s2 = [r["s2_4"] for r in recs]
            g1 = [r["pr_strength"] for r in recs]; g2 = [r["abs_strength"] for r in recs]
            sa = f"{100*accuracy(g1, s1):.1f} / {100*accuracy(g2, s2):.1f}"
            k1 = cohen_kappa(g1, s1, "linear", [0, 1, 2, 3]); k2 = cohen_kappa(g2, s2, "linear", [0, 1, 2, 3])
            ka = f"{fmt(k1)} / {fmt(k2)}"
        th2 = "-"
        if all("pred_th2_on7" in r for r in recs):
            th2 = f"{100*macro_f1(g, [r['pred_th2_on7'] for r in recs], EXAGG_LABELS):.1f}"
        lines.append(f"| {llm} | {mode} | {len(recs)} | {100*mf:.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}] | {100*accuracy(g, p):.1f} | "
                     f"{100*f[0]:.1f} / {100*f[1]:.1f} / {100*f[2]:.1f} | {sa} | {ka} | {th2} | {err} |")
    for name, val in REFERENCE["exagg"]:
        lines.append(f"| (문헌) {name} | — | — | {val} | | | | | | |")


def table_causal(groups, lines):
    lines.append("\n### 인과 언어 4분류 — Yu et al. 2019 (E2)\n")
    lines.append("| 시스템 | 방식 | n | macro-F1 [95% CI] | Acc | F1 none / corr / cond / direct | 오류 |")
    lines.append("|---|---|---|---|---|---|---|")
    for (mode, llm), recs in sorted(groups.items()):
        err = sum(bool(r.get("error")) for r in recs)
        recs = [r for r in recs if not r.get("error") and r.get("pred") in CAUSAL_LABELS]
        if not recs:
            lines.append(f"| {llm} | {mode} | 0 | — | | | {err} |")
            continue
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        mf = macro_f1(g, p, CAUSAL_LABELS)
        ci = bootstrap_ci(g, p, lambda a, b: macro_f1(a, b, CAUSAL_LABELS))
        f = [prf(g, p, l)[2] for l in CAUSAL_LABELS]
        lines.append(f"| {llm} | {mode} | {len(recs)} | {mf:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {accuracy(g, p):.3f} | "
                     f"{f[0]:.3f} / {f[1]:.3f} / {f[2]:.3f} / {f[3]:.3f} | {err} |")
    for name, val in REFERENCE["causal"]:
        lines.append(f"| (문헌) {name} | — | — | {val} | | | |")



def table_negtest(ds, groups, lines):
    """근거 없음 스트레스 테스트 — 오류 기록은 채점에서 뺀다(오류가 NEI 로 기록되어 거부율을 부풀리지 않게)."""
    title = "파일럿 — 캘리브레이션 판정" if ds == "negpilot" else "본 — 본판정·전문가 판정"
    lines.append(f"\n### 근거 없음 스트레스 테스트 ({title}; 주장 + 사람이 전원 '근거 아님'으로 판정한 문헌 1편, 정답 NEI)\n")
    lines.append("기존 계획의 탐색 검정: 근거 없음 거부율 ours − free, 거부 여부의 짝지은 정확 McNemar. 동일 주장 문헌쌍의 의존성은 별도 고려해야 한다. 보조: 인접(주제는 같고 결과지표가 다름). "
                 "오류 기록은 채점에서 제외.\n")
    lines.append("| LLM | 조건 | 채점 n | 거부율 [95% CI] | 무관 (k/n) | 인접 (k/n) | 잘못된 확신 % | 오류 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    by_llm = defaultdict(dict)
    for (cond, llm), recs in groups.items():
        by_llm[llm][cond] = recs
    ok = lambda recs: [r for r in recs if not r.get("error")]
    for llm, conds in sorted(by_llm.items()):
        for cond in ("ours", "free"):
            recs = conds.get(cond)
            if not recs:
                continue
            sc = ok(recs)
            n_ = len(sc)
            k = sum(1 for r in sc if r["pred"] == "NEI")
            ci = wilson(k, n_)
            sub = {}
            for kind in ("무관", "인접"):
                rs = [r for r in sc if (r.get("meta") or {}).get("kind") == kind]
                sub[kind] = f"{sum(1 for r in rs if r['pred'] == 'NEI')}/{len(rs)}"
            wrong = sum(1 for r in sc if r["pred"] in ("SUPPORT", "REFUTE"))
            lines.append(f"| {llm} | {cond} | {n_} | {pct(k / n_ if n_ else None)} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
                         f"{sub['무관']} | {sub['인접']} | {pct(wrong / n_ if n_ else None)} | {len(recs) - n_} |")
        if "ours" in conds and "free" in conds:
            fo = {r["id"]: r for r in ok(conds["free"])}
            common = [r for r in ok(conds["ours"]) if r["id"] in fo]
            if common:
                ro = [r["pred"] == "NEI" for r in common]
                rf = [fo[r["id"]]["pred"] == "NEI" for r in common]
                mc = mcnemar_bool(ro, rf)
                d = (sum(ro) - sum(rf)) / len(common)
                adj = [i for i, r in enumerate(common) if (r.get("meta") or {}).get("kind") == "인접"]
                adj_txt = (f"{100 * (sum(ro[i] for i in adj) - sum(rf[i] for i in adj)) / len(adj):+.1f}%p (n={len(adj)})" if adj else "-")
                lines.append(f"| {llm} | **ours − free** | {len(common)} | **Δ {100 * d:+.1f}%p** · ours만 거부 {mc['a_only']} : free만 거부 {mc['b_only']} · "
                             f"p={mc['p']:.4f} | | 인접 Δ {adj_txt} | | |")
            reasons = Counter(r.get("nei_reason") for r in ok(conds["ours"]) if r["pred"] == "NEI")
            lines.append(f"\nours 의 보류 사유 — " + " · ".join(f"{k or '?'} {v}" for k, v in reasons.most_common()))

def table_agreement(repeat_groups, lines):
    if not repeat_groups:
        return
    lines.append("\n### 재현성 — 같은 항목 반복 실행 판정 일치율\n")
    lines.append("| 데이터셋 | 조건 | LLM | 실행 수 | 항목 | 전부 같은 비율 |")
    lines.append("|---|---|---|---|---|---|")
    for (ds, cond, llm), runs in sorted(repeat_groups.items()):
        by_id = defaultdict(list)
        for recs in runs.values():
            for r in recs:
                if not r.get("error") and r.get("pred") != "ERROR":
                    by_id[r["id"]].append(r.get("pred4", r["pred"]))
        by_id = {i: values for i, values in by_id.items() if len(values) == len(runs)}
        ag = agreement(by_id)
        lines.append(f"| {ds} | {cond} | {llm} | {len(runs)} | {ag['n_items']} | {pct(ag['all_same_rate'])} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=DEFAULT_PREDS)
    ap.add_argument("--out", default=os.path.join(HERE, "report_bench.md"))
    a = ap.parse_args()
    PRIMARY.clear()
    has_mock = False
    verdict_groups = defaultdict(dict)      # dataset → {(cond, llm): recs}
    exagg_groups, causal_groups, e6_groups = {}, {}, {}
    neg_groups = defaultdict(dict)          # negtest/negpilot → {(cond, llm): recs}
    repeat_groups = defaultdict(dict)       # (ds, cond, llm) → {run: recs}
    for fn in sorted(glob.glob(os.path.join(a.preds, "*.jsonl"))):
        pn = parse_name(fn)
        if not pn:
            continue
        ds, cond, llm, run = pn
        recs = read_preds(fn)
        if not recs:
            continue
        has_mock = has_mock or llm.startswith('mock') or any(str(r.get('llm', '')).startswith('mock') for r in recs)
        if run:
            repeat_groups[(ds, cond, llm)][run] = recs
            if run != 1:
                continue
        if ds in ("scifact", "healthver", "checkcovid"):
            verdict_groups[ds][(cond, llm)] = recs
        elif ds == "exagg":
            exagg_groups[(cond, llm)] = recs
        elif ds == "causal":
            causal_groups[(cond, llm)] = recs
        elif ds == "e6":
            e6_groups[(cond, llm)] = recs
        elif ds in ("negtest", "negpilot"):
            neg_groups[ds][(cond, llm)] = recs
    lines = ["# 벤치마크 결과 (자동 생성 — report.py)", "",
             "지표 정의는 metrics.py. 오류는 NEI 정답으로 세지 않으며 성공 조건부 표와 오류 수를 함께 보고한다. "
             "문헌 행은 과거 문서의 참조값으로 이번 실행의 직접 비교가 아니다. 아래 기존 p 값은 항목 독립 가정의 탐색 분석이다. "
             "같은 주장/주제의 여러 문헌쌍은 독립 표본이 아니므로 확증 검정에는 주장 단위 재표집 또는 독립 주장 표본이 필요하다."]
    if has_mock:
        lines[2:2] = ["**모의 실행 포함: mock 행은 코드 점검용이며 실제 모델 성능이나 논문 결과로 사용할 수 없다.**", ""]
    for ds in ("scifact", "healthver", "checkcovid"):
        if verdict_groups.get(ds):
            table_safety(ds, verdict_groups[ds], lines)
            table_verdict(ds, verdict_groups[ds], lines)
    table_primary_holm(lines)
    if exagg_groups:
        table_exagg(exagg_groups, lines)
        table_candidate('exagg', exagg_groups, lines)
    if causal_groups:
        table_causal(causal_groups, lines)
    if e6_groups:
        table_safety("e6", e6_groups, lines)
        table_candidate('e6', e6_groups, lines)
        table_e6(e6_groups, lines)
    for ds in ("negpilot", "negtest"):
        if neg_groups.get(ds):
            table_negtest(ds, neg_groups[ds], lines)
    table_agreement(repeat_groups, lines)
    txt = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n→", a.out)


if __name__ == "__main__":
    main()
