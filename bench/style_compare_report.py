#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_compare_report.py — E5 문체 분류기 PLM 비교표 (2026-09-20 판).
train_kpf_bert_auto.py 가 남긴 kpf_train_report_<tag>.md 와 threshold_<tag>.json 을 모아 모델별 시드 평균±SD 표를 만든다.

태그 규약  <모델별칭>_s<시드>            예) cb_kpf_s1 · cb_klueroberta_s2         ← 학습 보고서(내부 테스트 6,001 · 홀드아웃 8,000)
          <모델별칭>_dedup_s<시드>      예) cb_kpf_dedup_s1                         ← --eval-only --sweep --eval-set dup_낚시성/test_dedup.jsonl
                                                                                     (AP = 내부 테스트, 외부 채점 = 근접중복 제거 홀드아웃 7,881)
          cb_tfidf_s<시드>              run_e5_colab.py 의 TF-IDF+LR 기준선 (같은 형식의 보고서)

    python style_compare_report.py                      # 현재 폴더의 kpf_train_report_*.md
    python style_compare_report.py --dir <보고서 폴더> --out e5_table       # → e5_table.md · e5_table.csv
    python style_compare_report.py --per-seed           # 시드별 행도 같이

판정 열: 홀드아웃(AI Hub 검증부 8,000 = 주 결과) · 근접중복 제거 홀드아웃(7,881) · 내부 테스트(7:1.5:1.5 의 1.5 = 6,001) · AP(내부 테스트).
문헌 참조 행은 분할·표본이 달라 직접 비교가 아니라 자리 확인용이다 — 표 각주에 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

HEAD = re.compile(r"^(?P<name>.*?)\s*—\s*정확도\s*(?P<acc>[0-9.]+)\s*·\s*macro-F1\s*(?P<f1>[0-9.]+)\s*$")
ROW = re.compile(r"^\|\s*(?P<cls>[^|]+?)\s*\|\s*(?P<p>[0-9.]+)\s*\|\s*(?P<r>[0-9.]+)\s*\|\s*(?P<f1>[0-9.]+)\s*\|\s*(?P<n>[0-9,]+)\s*\|")
AP = re.compile(r"test AP\(평균 정밀도\)\s*\*\*(?P<ap>[0-9.]+)\*\*")
EPOCH = re.compile(r"^에폭\s*(?P<ep>\d+):.*?val 정확도\s*(?P<acc>[0-9.]+)\s*·\s*macro-F1\s*(?P<f1>[0-9.]+)")
TIME = re.compile(r"^소요\s*(?P<min>[0-9.]+)분")
SEED = re.compile(r"seed\s+(?P<seed>\d+)")

NAMES = {"cb_kpf": "KPF-BERT (kpfbert)", "cb_klueroberta": "KLUE-RoBERTa-base", "cb_koelectra": "KoELECTRA-v3-base",
         "cb_kluebert": "KLUE-BERT-base", "cb_tfidf": "TF-IDF(char 2–4)+LR"}

LIT = [  # 문헌 참조 행 (분할·표본이 다르다 — 각주 필수)
    ("(문헌) AI Hub 공식 기준선 HAND", "정확도 87.1 (AI Hub 낚시성 기사 데이터셋 소개 페이지, 검증부)"),
    ("(문헌) AI Hub 공식 기준선 BERT", "정확도 83.88 · F1 90.80 (같은 출처)"),
    ("(문헌) 조예현 외 2025 KPF-BERT", "F1 88.6 (같은 AI Hub 데이터, 다른 분할)"),
]


def section_kind(name: str) -> str:
    n = name.replace("*", "")
    if "홀드아웃" in n:
        return "holdout"
    if "외부 채점" in n:
        return "dedup" if "dedup" in n else "external"
    if "테스트" in n or "test" in n.lower():
        return "test"
    return "other"


def parse(path: str) -> dict:
    """보고서 하나 → {"holdout": {...}, "test": {...}, "dedup": {...}, "ap": float|None, "epochs": [...], "minutes": float|None, "seed": int|None}"""
    lines = open(path, encoding="utf-8").read().splitlines()
    out = {"epochs": [], "ap": None, "minutes": None, "seed": None}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        m = HEAD.match(ln)
        if m:
            kind = section_kind(m.group("name"))
            rows = []
            j = i + 1
            while j < len(lines) and j < i + 12:
                r = ROW.match(lines[j].strip())
                if r and not r.group("cls").startswith("-"):
                    rows.append(r.groupdict())
                j += 1
            pos = rows[1] if len(rows) >= 2 else (rows[0] if rows else None)
            out[kind] = {"acc": float(m.group("acc")), "macro_f1": float(m.group("f1")),
                         "pos_p": float(pos["p"]) if pos else None, "pos_r": float(pos["r"]) if pos else None,
                         "pos_f1": float(pos["f1"]) if pos else None, "n": sum(int(r["n"].replace(",", "")) for r in rows) if rows else None,
                         "name": m.group("name").replace("*", "").strip()}
        m = AP.search(ln)
        if m:
            out["ap"] = float(m.group("ap"))
        m = EPOCH.match(ln)
        if m:
            out["epochs"].append((int(m.group("ep")), float(m.group("acc")), float(m.group("f1"))))
        m = TIME.match(ln)
        if m:
            out["minutes"] = float(m.group("min"))
        if ln.startswith("설정:") or ln.startswith("평가 전용:"):
            s = SEED.search(ln)
            if s:
                out["seed"] = int(s.group("seed"))
        i += 1
    return out


def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return sum(v) / len(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def fmt(v, digits=4):
    m, sd, n = mean_sd(v)
    if m is None:
        return "—"
    return f"{m:.{digits}f} ± {sd:.{digits}f}" if n > 1 else f"{m:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default=None, help="이 이름으로 .md 와 .csv 저장 (예: e5_table)")
    ap.add_argument("--per-seed", action="store_true")
    ap.add_argument("--include-rehearsal", action="store_true", help="리허설(_rehearsal_) 태그도 표에 넣음 (기본 제외)")
    ap.add_argument("--expected-seeds", default="1,2,3", help="Predeclared seed IDs, not the best-performing subset")
    a = ap.parse_args()

    train, dedup = defaultdict(dict), defaultdict(dict)        # model → seed → parsed
    for fn in sorted(glob.glob(os.path.join(a.dir, "kpf_train_report_*.md"))):
        tag = os.path.basename(fn)[len("kpf_train_report_"):-3]
        m = re.match(r"^(?P<model>.+?)(?P<dd>_dedup)?_s(?P<seed>\d+)$", tag)
        if not m:
            continue
        if "_rehearsal" in tag and not a.include_rehearsal:
            continue
        rec = parse(fn)
        rec["tag"] = tag
        tj = os.path.join(a.dir, f"threshold_{tag}.json")
        if os.path.isfile(tj):
            try:
                rec["ap"] = rec["ap"] if rec["ap"] is not None else json.load(open(tj, encoding="utf-8")).get("test_AP")
            except Exception:
                pass
        (dedup if m.group("dd") else train)[m.group("model")][m.group("seed")] = rec

    order = {"cb_kpf": 0, "cb_klueroberta": 1, "cb_koelectra": 2, "cb_kluebert": 3, "cb_tfidf": 9}
    models = sorted(set(train) | set(dedup), key=lambda m: (order.get(m, 5), m))
    hdr = ["모델(태그)", "시드", "홀드아웃 8,000 정확도", "홀드아웃 macro-F1", "홀드아웃 낚시성 P / R / F1",
           "중복제거 홀드아웃 정확도", "중복제거 홀드아웃 macro-F1", "내부 테스트 정확도", "내부 테스트 macro-F1", "AP(내부 테스트)", "학습 분"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    csv_rows = [hdr]

    def row_for(model, seeds):
        tr = [train[model][s] for s in seeds if s in train[model]]
        dd = [dedup[model][s] for s in seeds if s in dedup[model]]
        g = lambda recs, k, f: [r.get(k, {}).get(f) for r in recs if r.get(k)]  # noqa: E731
        hp, hr, hf = g(tr, "holdout", "pos_p"), g(tr, "holdout", "pos_r"), g(tr, "holdout", "pos_f1")
        ap_v = [r["ap"] for r in dd if r.get("ap") is not None] or [r["ap"] for r in tr if r.get("ap") is not None]
        cells = [NAMES.get(model, model), str(len(tr) or len(dd)),
                 fmt(g(tr, "holdout", "acc")), fmt(g(tr, "holdout", "macro_f1")),
                 (f"{mean_sd(hp)[0]:.4f} / {mean_sd(hr)[0]:.4f} / {mean_sd(hf)[0]:.4f}" if hf and hf[0] is not None else "—"),
                 fmt(g(dd, "dedup", "acc")), fmt(g(dd, "dedup", "macro_f1")),
                 fmt(g(tr, "test", "acc")), fmt(g(tr, "test", "macro_f1")),
                 fmt(ap_v), fmt([r["minutes"] for r in tr if r.get("minutes") is not None], 1)]
        return cells

    for model in models:
        seeds = sorted(set(train[model]) | set(dedup[model]), key=lambda s: int(s))
        cells = row_for(model, seeds)
        lines.append("| " + " | ".join(cells) + " |"); csv_rows.append(cells)
        if a.per_seed:
            for s in seeds:
                c = row_for(model, [s]); c[0] = f"  └ {model}_s{s}"
                lines.append("| " + " | ".join(c) + " |"); csv_rows.append(c)
    for name, note in LIT:
        lines.append(f"| {name} | — | {note} | | | | | | | | |"); csv_rows.append([name, "—", note])

    text = "\n".join(lines)
    required = set(a.expected_seeds.split(','))
    audit = ["\n\n**비교 완결성 — 결과를 보고 시드를 골라 평균내지 않는다**", "",
             "| 모델 | 학습 시드 | 중복 제거 시드 | 누락 |", "|---|---|---|---|"]
    for model in models:
        tr, dd = set(train[model]), set(dedup[model])
        missing = ["학습 " + ','.join(sorted(required-tr))] if required-tr else []
        if required-dd:
            missing.append("중복 제거 " + ','.join(sorted(required-dd)))
        audit.append(f"| {model} | {','.join(sorted(tr))} | {','.join(sorted(dd))} | {'; '.join(missing) or '없음'} |")
    text += '\n'.join(audit)
    text += ("\n\n동급 여부는 평균이 비슷하거나 유의하지 않다는 이유만으로 선언하지 않는다. "
             "원 홀드아웃·중복 제거 결과를 함께 보고, 같은 장비·분할·3시드에서 비교한다. "
             "이 집계는 예측 원본이 없어 항목별 짝지은 검정이나 비열등성 검정을 대신하지 않는다. "
             "광고성·자극성의 매체 외부 검증은 별도 실험이며 낚시성 점수로 일반화하지 않는다.")
    text += ("\n\n주: 홀드아웃 = AI Hub 낚시성 데이터셋 검증부에서 뽑은 8,000건(낚시성 4,000 · 정상 4,000, `clickbait_valid.jsonl`) — 주 결과. "
             "중복제거 홀드아웃 = 학습 데이터와 어절 3-gram 자카드 ≥ 0.5 인 119건을 뺀 7,881건(`dup_낚시성/test_dedup.jsonl`). "
             "내부 테스트 = 학습 표본 40,000 의 7:1.5:1.5 분할 중 1.5(6,001건, `kpf_split_낚시성.json`). "
             "AP = 내부 테스트의 평균 정밀도(임계값 무관). 시드 3개 평균 ± 표준편차. "
             "문헌 행은 AI Hub 데이터셋 소개 페이지·해당 논문의 수치로 분할·표본이 달라 직접 비교가 아니다.")
    print(text)
    if a.out:
        open(a.out + ".md", "w", encoding="utf-8").write(text + "\n")
        with open(a.out + ".csv", "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(csv_rows)
        print(f"\n→ {a.out}.md · {a.out}.csv")
    if not models:
        print("(kpf_train_report_<모델>_s<시드>.md 가 없습니다 — --dir 확인)")


if __name__ == "__main__":
    main()
