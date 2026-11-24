#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loaders.py — 공개 벤치마크를 하나의 항목 형식으로 읽는다.

항목 형식 (run_verdict.py 가 쓰는 것)
    {"id": str, "dataset": str, "claim": str,
     "docs": [{"doc_id": str, "title": str, "abstract": str, "sentences": [str], "pubtypes": [], "year": ""}],
     "gold": "SUPPORT" | "REFUTE" | "NEI",
     "gold_rationale": [str],          # 정답 근거 문장 (있으면) — 인용 재현율 계산용
     "meta": {...}}

지원
    scifact     SciFact 원본 (github.com/allenai/scifact → data.tar.gz): claims_dev.jsonl + corpus.jsonl.
                쌍 단위(주장 × cited_doc) 로 만든다. 문헌에 evidence 가 있으면 그 라벨(SUPPORT / CONTRADICT→REFUTE), 없으면 NEI.
                Check-COVID(github.com/posuer/Check-COVID) 도 같은 형식이라 --claims/--corpus 만 바꾸면 된다.
    healthver   github.com/sarrouti/healthver → data/healthver_test.csv (id,evidence,claim,label,topic_ip,question)
    exagg       Wright & Augenstein 2021 (HF copenlu/scientific-exaggeration-detection 의 test 분할을 csv/jsonl/parquet 로 저장)
                열: press_release_conclusion, press_release_strength, abstract_conclusion, abstract_strength, exaggeration_label
    causal      Yu et al. 2019 (github.com/junwang4/causal-language-use-in-science → data/…csv). 열 이름은 자동 추정.
                라벨 번호의 뜻은 저장소 README 기준 0 no relationship · 1 direct causal · 2 conditional causal · 3 correlational
                (분포 1,356 / 494 / 213 / 998 로 확인할 것 — 다르면 CAUSAL_CODE 를 고친다)
"""
from __future__ import annotations

import csv
import json
import os
import sys

csv.field_size_limit(10 ** 8)

GOLD3 = ("SUPPORT", "REFUTE", "NEI")


def _read_json_any(path: str):
    """jsonl 또는 json(list/dict) 를 list 로."""
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    if txt[0] == "{" and "\n" not in txt.strip():
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _norm_label(s: str) -> str:
    s = (s or "").strip().upper()
    if s in ("SUPPORT", "SUPPORTS", "SUPPORTED", "ENTAILMENT", "TRUE"):
        return "SUPPORT"
    if s in ("CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "REFUTED", "CONTRADICTION", "FALSE"):
        return "REFUTE"
    if s in ("NOINFO", "NEI", "NEUTRAL", "NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NOTENOUGHINFO", "UNADDRESSED"):
        return "NEI"
    raise ValueError(f"알 수 없는 라벨: {s!r}")


# ───────────────────────────────────────────────────────────── SciFact 형식
def load_scifact(claims_path: str, corpus_path: str, dataset: str = "scifact", limit: int = 0) -> list[dict]:
    corpus = {}
    for d in _read_json_any(corpus_path):
        sents = d.get("abstract") or []
        if isinstance(sents, str):
            sents = [sents]
        did = d.get("doc_id", d.get("cord_id"))          # Check-COVID 원본은 cord_id (2026-11-10)
        corpus[str(did)] = {"doc_id": str(did), "title": d.get("title") or "",
                                    "abstract": " ".join(sents), "sentences": list(sents),
                                    "pubtypes": [], "year": str(d.get("year") or "")}
    items = []
    # 2026-11-10: 주장 파일을 쉼표로 여러 개 줄 수 있다 (Check-COVID test = composed + extracted)
    all_claims = [c for p in str(claims_path).split(",") if p.strip() for c in _read_json_any(p.strip())]
    for c in all_claims:
        if "cord_id" in c and "label" in c:
            # Check-COVID 원본 형식: 주장 1건 = 문헌 1건(cord_id) · label · evidence_set[{sent_index, type}]
            # 이 벤치마크는 **문장 단위**다 — 라벨은 evidence_set 문장에 대한 것이고, NEI 항목은 같은 주장을 같은 초록의
            # 다른(근거가 아닌) 문장과 짝지은 것이다. 초록 전체를 주면 NEI 정답이 틀려지므로 evidence_set 문장만 문헌으로 준다
            # (HealthVer 와 같은 형식 · 제목은 주지 않는다).
            full = corpus.get(str(c["cord_id"]))
            if not full:
                continue
            gold = _norm_label(c["label"])
            idx = sorted({int(e["sent_index"]) for e in (c.get("evidence_set") or [])
                          if 0 <= int(e.get("sent_index", -1)) < len(full["sentences"])})
            if not idx:
                continue
            sents = [full["sentences"][i] for i in idx]
            doc = {"doc_id": full["doc_id"], "title": "", "abstract": " ".join(sents), "sentences": sents,
                   "pubtypes": [], "year": full.get("year", "")}
            rat = sents if gold != "NEI" else []
            items.append({"id": f"{c.get('id')}__{c['cord_id']}", "dataset": dataset, "claim": c["claim"],
                          "docs": [doc], "gold": gold, "gold_rationale": rat,
                          "meta": {"claim_id": c.get("id"), "doc_id": str(c["cord_id"]),
                                   "source": "auth" if c.get("is_auth") else "nonauth"}})
            if limit and len(items) >= limit:
                return items
            continue
        ev = c.get("evidence") or {}
        cited = [str(x) for x in (c.get("cited_doc_ids") or [])]
        if not cited:
            cited = list(ev.keys())
        for doc_id in cited:
            doc = corpus.get(str(doc_id))
            if not doc:
                continue
            ev_list = ev.get(str(doc_id)) or ev.get(int(doc_id) if doc_id.isdigit() else doc_id) or []
            if ev_list:
                labels = {_norm_label(e.get("label")) for e in ev_list}
                gold = "REFUTE" if "REFUTE" in labels else "SUPPORT"
                rat = []
                for e in ev_list:
                    for si in e.get("sentences") or []:
                        if 0 <= int(si) < len(doc["sentences"]):
                            rat.append(doc["sentences"][int(si)])
            else:
                gold, rat = "NEI", []
            items.append({"id": f"{c.get('id')}__{doc_id}", "dataset": dataset, "claim": c["claim"],
                          "docs": [doc], "gold": gold, "gold_rationale": rat,
                          "meta": {"claim_id": c.get("id"), "doc_id": doc_id,
                                   "source": c.get("source") or c.get("auth") or c.get("authorship") or ""}})
            if limit and len(items) >= limit:
                return items
    return items


# ───────────────────────────────────────────────────────────── HealthVer
def load_healthver(csv_path: str, limit: int = 0) -> list[dict]:
    items = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ev = (r.get("evidence") or "").strip()
            items.append({"id": f"hv{r.get('id') or i}", "dataset": "healthver", "claim": (r.get("claim") or "").strip(),
                          "docs": [{"doc_id": f"hv{r.get('id') or i}", "title": "", "abstract": ev, "sentences": [ev],
                                    "pubtypes": [], "year": ""}],
                          "gold": _norm_label(r.get("label")), "gold_rationale": [ev],
                          "meta": {"topic": r.get("topic_ip"), "question": r.get("question")}})
            if limit and len(items) >= limit:
                break
    return items


# ───────────────────────────────────────────────────────────── 과장 (W&A 2021)
def _read_table(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        return _read_json_any(path)
    if ext == ".parquet":
        import pandas as pd  # noqa
        return pd.read_parquet(path).to_dict("records")
    delim = "\t" if ext == ".tsv" else ","
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def load_exagg(path: str, limit: int = 0) -> list[dict]:
    items = []
    for i, r in enumerate(_read_table(path)):
        items.append({"id": str(r.get("original_file_id") or i), "dataset": "exagg",
                      "pr": (r.get("press_release_conclusion") or "").strip(),
                      "abs": (r.get("abstract_conclusion") or "").strip(),
                      "pr_strength": int(r["press_release_strength"]), "abs_strength": int(r["abstract_strength"]),
                      "gold": (r.get("exaggeration_label") or "").strip().lower()})
        if limit and len(items) >= limit:
            break
    return items


# ───────────────────────────────────────────────────────────── 인과 언어 (Yu 2019)
CAUSAL_CODE = {0: "no_relationship", 1: "direct_causal", 2: "conditional_causal", 3: "correlational"}   # 저장소 README 기준


def load_causal(path: str, limit: int = 0, code: dict | None = None) -> list[dict]:
    code = code or CAUSAL_CODE
    rows = _read_table(path)
    if not rows:
        return []
    keys = list(rows[0].keys())
    sk = next((k for k in keys if k.lower() in ("sentence", "text", "conclusion", "sent")), keys[0])
    lk = next((k for k in keys if "label" in k.lower() or k.lower() in ("class", "y")), keys[-1])
    items = []
    for i, r in enumerate(rows):
        try:
            lab = int(float(r[lk]))
        except (TypeError, ValueError):
            continue
        items.append({"id": str(r.get("id") or i), "dataset": "causal", "sentence": (r[sk] or "").strip(),
                      "gold_code": lab, "gold": code.get(lab, str(lab))})
        if limit and len(items) >= limit:
            break
    return items


def label_counts(items: list[dict], key: str = "gold") -> dict:
    c = {}
    for it in items:
        c[it.get(key)] = c.get(it.get(key), 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    # 사용법:  python data_loaders.py scifact claims_dev.jsonl corpus.jsonl
    #         python data_loaders.py healthver healthver_test.csv | exagg test.csv | causal pubmed.csv
    kind = sys.argv[1]
    if kind == "scifact":
        its = load_scifact(sys.argv[2], sys.argv[3])
    elif kind == "healthver":
        its = load_healthver(sys.argv[2])
    elif kind == "exagg":
        its = load_exagg(sys.argv[2])
    elif kind == "causal":
        its = load_causal(sys.argv[2])
    else:
        raise SystemExit("scifact | healthver | exagg | causal")
    print(kind, len(its), "items; labels:", label_counts(its))
    print(json.dumps(its[0], ensure_ascii=False)[:600])
