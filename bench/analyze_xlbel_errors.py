#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_xlbel_errors.py — XL-BEL KO 예측(preds/el_xlbel_ko_<system>.jsonl)의 오류 분해 (2026-09-22)

엄격 P@1 만 보면 안 보이는 것을 나눈다:
  · 계층 관계  : 예측이 gold 의 MeSH 트리에서 부모/자식/형제(같은 부모)인가 — XL-BEL gold 는 '앵커가 가리키는 위키 문서'의 개념이라
                 표면형보다 상위(심장질환→관상동맥질환)이거나 하위(직장암→Neoplasms)인 경우가 많다
  · 무응답 원인 : 스톱리스트(일반어 차단) / 라틴 문자·원소기호 / 조사·부가어 붙은 구 / 사전에 없음
  · 보조 지표   : 계층 관대 P@1 (gold 또는 부모·자식) — 엄격값과 나란히 보고, 대체하지 않는다

    python analyze_xlbel_errors.py --system kob           # → xlbel_error_analysis_kob.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)


def load_stop(path):
    stop = set()
    if os.path.isfile(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.split("#", 1)[0].strip()
            if ln:
                stop.add(ln)
    return stop


def trees(mesh_ko, ui):
    return (mesh_ko.get(ui) or {}).get("tree_numbers") or []


def relation(pred, golds, mesh_ko):
    """'same' / 'child' (예측이 gold 의 하위) / 'parent' (예측이 gold 의 상위) / 'sibling' / 'ancestor2+' / 'descendant2+' / 'none'"""
    if pred in golds:
        return "same"
    pt = trees(mesh_ko, pred)
    best = "none"
    rank = {"none": 0, "sibling": 1, "ancestor2+": 2, "descendant2+": 2, "parent": 3, "child": 3}
    for g in golds:
        for gt in trees(mesh_ko, g):
            for p in pt:
                rel = None
                if p.startswith(gt + "."):
                    rel = "child" if p.count(".") == gt.count(".") + 1 else "descendant2+"
                elif gt.startswith(p + "."):
                    rel = "parent" if gt.count(".") == p.count(".") + 1 else "ancestor2+"
                elif "." in p and "." in gt and p.rsplit(".", 1)[0] == gt.rsplit(".", 1)[0]:
                    rel = "sibling"
                if rel and rank[rel] > rank[best]:
                    best = rel
    return best


LATIN = re.compile(r"[A-Za-z]")
PARTICLE_TAIL = re.compile(r"(을|를|이|가|은|는|의|에|에서|으로|로|와|과|도|만)$")


def noans_reason(mention, stop, mesh_ko_terms):
    m = mention.strip()
    if m in stop:
        return "스톱리스트(일반어 차단)"
    if LATIN.search(m) and not re.search(r"[가-힣]", m):
        return "라틴 문자·원소기호(한국어 리졸버 범위 밖)"
    if LATIN.search(m):
        return "한글+라틴 혼용"
    if len(m.split()) >= 3 or re.search(r"\d{4}년", m) or PARTICLE_TAIL.search(m.replace(" ", "")) and len(m) > 4:
        return "구(句)·부가어·조사 포함"
    if m.replace(" ", "") in mesh_ko_terms:
        return "사전에 있으나 무응답(공백·규칙 문제)"
    return "사전에 없음(용어·이형태)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="kob")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--mesh-ko", default=os.path.join(MESH, "mesh_ko_full.json"))
    ap.add_argument("--stop", default=os.path.join(MESH, "ko_stoplist.txt"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    preds = a.preds or os.path.join(HERE, "preds", f"el_xlbel_ko_{a.system}.jsonl")
    out = a.out or os.path.join(HERE, f"xlbel_error_analysis_{a.system}.md")
    rows = [json.loads(l) for l in open(preds, encoding="utf-8")]
    mesh_ko = json.load(open(a.mesh_ko, encoding="utf-8"))
    terms = set()
    for d in mesh_ko.values():
        for t in [d.get("ko_main") or ""] + (d.get("ko_synonyms") or []):
            if t:
                terms.add(t.replace(" ", ""))
    stop = load_stop(a.stop)

    L = [f"# XL-BEL KO 오류 분해 — {a.system} ({len(rows)}건, MeSH 대응 가능분)", ""]
    cats = ["C", "D", "A", "B"]
    L += ["## 1. 엄격 P@1 vs 계층 관대 P@1 (gold 또는 그 부모·자식)", "",
          "| 트리 | n | 엄격 P@1 | +부모/자식 | +형제 | 무응답 | 진짜 오답 |", "|---|---|---|---|---|---|---|"]
    summary = {}
    for cat in cats + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        if not rs:
            continue
        n = len(rs); c = collections.Counter()
        for r in rs:
            if r["noans"]:
                c["noans"] += 1; continue
            rel = relation(r["top"][0]["ui"], set(r["gold"]), mesh_ko)
            c[rel] += 1
        strict = c["same"]; hier = strict + c["parent"] + c["child"]; sib = hier + c["sibling"]
        wrong = n - c["noans"] - sib
        summary[cat] = (n, strict, hier, sib, c["noans"], wrong)
        L.append(f"| {cat} | {n} | {100*strict/n:.1f} | {100*hier/n:.1f} | {100*sib/n:.1f} | {100*c['noans']/n:.1f} | {100*wrong/n:.1f} |")
    L += ["", "관계 정의: 부모/자식 = MeSH 트리 번호가 한 단계 위/아래 · 형제 = 같은 부모. 예측이 gold 보다 **구체적**(자식)인 경우가 많다 — 앵커 표면형은 구체어인데 gold 는 링크된 문서의 상위 개념.", ""]

    L += ["## 2. 오답의 관계 분포 (응답했으나 엄격 오답)", "", "| 트리 | 자식(더 구체적) | 부모(더 일반적) | 형제 | 2단계 이상 | 무관 |", "|---|---|---|---|---|---|"]
    for cat in cats:
        rs = [r for r in rows if r["cat"] == cat and not r["noans"] and not r["hit1"]]
        c = collections.Counter(relation(r["top"][0]["ui"], set(r["gold"]), mesh_ko) for r in rs)
        L.append(f"| {cat} | {c['child']} | {c['parent']} | {c['sibling']} | {c['ancestor2+'] + c['descendant2+']} | {c['none']} |")

    L += ["", "## 3. 무응답 원인", "", "| 트리 | 원인 | n | 예 |", "|---|---|---|---|"]
    for cat in cats:
        rs = [r for r in rows if r["cat"] == cat and r["noans"]]
        by = collections.defaultdict(list)
        for r in rs:
            by[noans_reason(r["mention"], stop, terms)].append(r["mention"])
        for reason, ms in sorted(by.items(), key=lambda x: -len(x[1])):
            L.append(f"| {cat} | {reason} | {len(ms)} | {' · '.join(ms[:6])} |")

    L += ["", "## 4. '무관' 오답 전부 (사람이 볼 것)", ""]
    for cat in cats:
        rs = [r for r in rows if r["cat"] == cat and not r["noans"] and not r["hit1"]]
        bad = [r for r in rs if relation(r["top"][0]["ui"], set(r["gold"]), mesh_ko) == "none"]
        if bad:
            L.append(f"**{cat}** ({len(bad)}건)")
            L += [f"- {r['mention']} → {r['top'][0]['name']} [{r['top'][0]['method']} {r['top'][0]['score']}] · gold {', '.join(r['gold_names'])}" for r in bad]
            L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
