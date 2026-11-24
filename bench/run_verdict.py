#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_verdict.py — E1 판정 계층 벤치마크 러너 (SciFact · HealthVer · Check-COVID)

    # SciFact dev (쌍 단위) — 우리 시스템(판독 + 고정 규칙) · 같은 LLM 자유 판정 · closed-book
    python run_verdict.py --dataset scifact --claims data/scifact/claims_dev.jsonl --corpus data/scifact/corpus.jsonl \
        --llm gemini:gemini-3.1-pro-preview --cond ours free closed
    # HealthVer test
    python run_verdict.py --dataset healthver --csv data/healthver/healthver_test.csv --llm upstage:solar-pro4 --cond ours free
    # 재현성(3회 반복, 캐시 끔)
    python run_verdict.py --dataset scifact ... --llm gemini --cond ours --repeat 3 --no-cache --limit 150

조건
    ours    verdict.py 의 SYSTEM·TOOL 로 판독 → verdict.decide() 고정 규칙 → {지지,범위 초과}→SUPPORT · 반박→REFUTE · 근거 없음→NEI
            (범위 초과 여부·초과 차원·S/R 가중·인용 문장 존재 여부(verbatim)를 함께 기록)
    free    같은 문헌을 주고 LLM 이 범주를 **직접** 출력 (규칙 없음 ablation)
    closed  문헌 없이 LLM 지식만으로 출력 (검색 효과 ablation)

출력  preds/<dataset>_<cond>_<provider-model>[_r<k>].jsonl  — 항목마다 gold · pred · 세부. 이미 있는 id 는 건너뛴다(재개).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import verdict  # noqa: E402  (mesh/verdict.py — 논문의 판독 프롬프트·고정 규칙 그대로)
from data_loaders import load_scifact, load_healthver, label_counts  # noqa: E402
from llm_client import make_llm  # noqa: E402
from protocol import DEFAULT_PREDS, VERSION, fingerprint, resume_ids, stamp, exclude_items

CAT2LABEL = {"지지": "SUPPORT", "범위 초과": "SUPPORT", "반박": "REFUTE", "근거 없음": "NEI"}

FREE_SYSTEM = """당신은 주장(claim)을 주어진 문헌과 대조해 판정하는 판정기다. **주어진 문헌만** 근거로 삼는다(자기 지식으로 진위를 정하지 않는다).
SUPPORT = 문헌이 주장을 뒷받침한다 / REFUTE = 주장의 관계를 직접 조사한 문헌이 반대 결과나 효과·연관 없음의 결과를 보고한다 / NEI = 문헌만으로는 판단할 근거가 부족하다.
주제만 겹치는 무관·인접 문헌은 REFUTE가 아니라 NEI다. 근거 미확인은 효과 없음의 증거가 아니다.
판정 근거가 된 문헌 문장 한 개를 영어 원문 그대로 key_sentence 에 복사하고, reason 에 한 문장으로 이유를 쓴다."""

CLOSED_SYSTEM = """당신은 주장(claim)의 사실성을 판정하는 판정기다. 문헌은 주어지지 않는다 — 당신이 아는 의학 지식으로 판정한다.
SUPPORT = 확립된 근거가 주장을 뒷받침한다 / REFUTE = 근거가 주장과 배치된다 / NEI = 근거가 불충분하거나 모른다. reason 에 한 문장으로 이유를 쓴다."""

FREE_TOOL = {"name": "submit_verdict", "description": "주장에 대한 판정을 제출한다.",
             "input_schema": {"type": "object",
                              "properties": {"label": {"type": "string", "enum": ["SUPPORT", "REFUTE", "NEI"]},
                                             "key_sentence": {"type": "string"},
                                             "reason": {"type": "string"}},
                              "required": ["label"]}}

# ── 규칙을 뺀 조건 (norule, 2026-11-16) ─────────────────────────────────────────
# 같은 구조화 판독(v1)을 시키되 최종 범주를 고정 규칙이 아니라 LLM 이 직접 정한다. 결정적 단계(규칙)의 몫을 따로 보기 위한
# 절제 조건이다 — Faithful CoT(Lyu 외 2023)의 "No solver" 와 같은 설계. 네 범주의 정의만 주고 임계·집계 방식은 주지 않는다.
# 사전 기록: claude_backup/실험1·2_재설계_확정_1116.md §2 (결과가 어떻든 실험 1 표에 싣는다).
_RULE_SENT = "최종 범주(지지/범위 초과/반박/근거 없음)는\n당신이 아니라 고정 규칙이 정한다. 당신은 재료만 낸다."
assert _RULE_SENT in verdict.SYSTEM
NORULE_SYSTEM = verdict.SYSTEM.replace(
    _RULE_SENT, "그리고 그 판독을 바탕으로\n최종 범주(지지/범위 초과/반박/근거 없음)도 당신이 직접 정한다.") + """

최종 범주 — category (위 판독을 바탕으로 주장에 대해 하나를 고른다)
   지지       관련 문헌의 결론이 주장을 뒷받침하고, 주장이 문헌이 진술한 수준을 넘지 않는다.
   범위 초과  문헌의 결론은 주장과 같은 방향이지만, 주장이 문헌보다 강하게 말한다(인과 강도 · 조언 수준 · 대상 종 · 개입의 특정성).
   반박       관련 문헌이 주장과 반대되는 결과나 효과 없음을 보고한다.
   근거 없음  주장의 개입·노출과 결과를 함께 다룬 문헌이 없거나, 관련 문헌의 결론이 유보적이어서 판단할 수 없다."""
NORULE_TOOL = copy.deepcopy(verdict.TOOL)
NORULE_TOOL["name"] = "submit_reading_with_category"
NORULE_TOOL["description"] = "주장과 문헌 대조 결과와 최종 범주를 제출한다."
NORULE_TOOL["input_schema"]["properties"]["category"] = {"type": "string", "enum": ["지지", "범위 초과", "반박", "근거 없음"]}
NORULE_TOOL["input_schema"]["required"] = list(NORULE_TOOL["input_schema"]["required"]) + ["category"]
_READ_LAST = "각 문헌에 대해 relevant/stance/key_sentence/coding 을, 주장에 대해 claim_coding 을, 그리고 explanation_ko 를 submit_reading 으로 제출하라."
_NORULE_LAST = ("각 문헌에 대해 relevant/stance/key_sentence/coding 을, 주장에 대해 claim_coding 을, 그리고 explanation_ko 와 "
                "최종 범주 category 를 submit_reading_with_category 로 제출하라.")


def _norm(s: str) -> str:
    return re.sub(r"[\s\W]+", " ", (s or "").lower()).strip()


def _in_rationale(sentence: str, rationale: list[str]) -> bool:
    s = _norm(sentence)
    if not s or not rationale:
        return False
    return any(s in _norm(r) or _norm(r) in s for r in rationale)


def to_hits(docs: list[dict]) -> list[dict]:
    return [{"pmid": d.get("doc_id"), "year": d.get("year") or "-", "pubtypes": d.get("pubtypes") or [],
             "title": d.get("title") or "", "abstract": d.get("abstract") or "", "rank": i + 1}
            for i, d in enumerate(docs)]


def citation_checks(sentences, hits, rules, numbers=None):
    """Same check for both conditions, limited to the abstracts actually shown.

    Whitespace normalization only: this is quote presence, NOT entailment/citation recall.
    Blank citations receive no success credit; cross-document concatenation is disallowed.
    """
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
    abstracts = [norm(h.get("abstract"))[:rules["abstract_chars"]] for h in hits[:rules["k_evidence"]]]
    result = []
    for i, sentence in enumerate(sentences):
        quote = norm(sentence)
        if numbers is None:
            candidates = abstracts
        else:
            number = numbers[i]
            candidates = [abstracts[number - 1]] if isinstance(number, int) and 1 <= number <= len(abstracts) else []
        result.append(bool(quote) and any(quote in abstract for abstract in candidates))
    return result


def run_item(item: dict, cond: str, llm, rules: dict) -> dict:
    t0 = time.time()
    rec = {"id": item["id"], "dataset": item["dataset"], "cond": cond, "llm": llm.label, "gold": item["gold"],
           "claim": item["claim"], "meta": item.get("meta", {}), "protocol_version": VERSION,
           "gold_rationale_available": bool(item.get("gold_rationale"))}
    try:
        if cond == "ours":
            hits = to_hits(item["docs"])[:rules["k_evidence"]]
            reading = llm.call(verdict.system_for(rules), verdict._prompt(item["claim"], hits, rules), verdict.tool_for(rules))
            if not isinstance(reading.get("evidence"), list) or not isinstance(reading.get("claim_coding"), dict):
                raise ValueError("Malformed reading response")
            res = verdict.decide(item["claim"], hits, reading, rules)
            rec.update({"pred": CAT2LABEL[res["category"]], "category": res["category"],
                        "nei_reason": res.get("nei_reason"), "sub": res["sub"],
                        "mixed": res["mixed"], "support_w": res["support_w"], "refute_w": res["refute_w"],
                        "n_relevant": res["n_relevant"], "exceeded": [x["dim"] for x in res["exceeded"]],
                        "claim_coding": res["claim_coding"],
                        "stances": [e["stance"] for e in res["evidence"]],
                        "verbatim": [e["verbatim"] for e in res["evidence"] if e["relevant"]],
                        "key_sentences": [e["key_sentence"] for e in res["evidence"] if e["relevant"]],
                        "explanation_ko": res["explanation_ko"]})
            if rules.get("reader") == "suff":     # 판독기 S (2026-11-15) — v1 레코드 키는 그대로
                rec.update({"sufficiency": [e.get("sufficiency") for e in res["evidence"] if e["relevant"]],
                            "n_partial": res["n_partial"], "n_insufficient": res["n_insufficient"],
                            "n_suff_missing": res["n_suff_missing"]})
            ks = rec["key_sentences"]
            rec["verbatim"] = citation_checks(ks, hits, rules,
                                              [e["n"] for e in res["evidence"] if e["relevant"]])
            rec["cite_hit"] = any(_in_rationale(k, item.get("gold_rationale") or []) for k in ks) if ks else False
        elif cond == "norule":
            hits = to_hits(item["docs"])[:rules["k_evidence"]]
            prompt = verdict._prompt(item["claim"], hits, rules).replace(_READ_LAST, _NORULE_LAST)
            reading = llm.call(NORULE_SYSTEM, prompt, NORULE_TOOL)
            if not isinstance(reading.get("evidence"), list) or not isinstance(reading.get("claim_coding"), dict):
                raise ValueError("Malformed reading response")
            cat = str(reading.get("category") or "").strip()
            if cat not in CAT2LABEL:
                raise ValueError(f"Invalid category: {cat!r}")
            res = verdict.decide(item["claim"], hits, reading, rules)     # 같은 판독에 규칙을 적용했다면 나왔을 범주
            rec.update({"pred": CAT2LABEL[cat], "category": cat,
                        "rule_category": res["category"], "consistent": cat == res["category"],
                        "nei_reason": res.get("nei_reason"), "n_relevant": res["n_relevant"],
                        "exceeded": [x["dim"] for x in res["exceeded"]], "claim_coding": res["claim_coding"],
                        "stances": [e["stance"] for e in res["evidence"]],
                        "key_sentences": [e["key_sentence"] for e in res["evidence"] if e["relevant"]],
                        "explanation_ko": res["explanation_ko"]})
            ks = rec["key_sentences"]
            rec["verbatim"] = citation_checks(ks, hits, rules, [e["n"] for e in res["evidence"] if e["relevant"]])
            rec["cite_hit"] = any(_in_rationale(k, item.get("gold_rationale") or []) for k in ks) if ks else False
        elif cond == "free":
            hits = to_hits(item["docs"])[:rules["k_evidence"]]
            out = llm.call(FREE_SYSTEM, verdict._prompt(item["claim"], hits, rules).replace(
                "각 문헌에 대해 relevant/stance/key_sentence/coding 을, 주장에 대해 claim_coding 을, 그리고 explanation_ko 를 submit_reading 으로 제출하라.",
                "label / key_sentence / reason 을 submit_verdict 로 제출하라."), FREE_TOOL)
            lab = str(out.get("label", "")).upper()
            if lab not in ("SUPPORT", "REFUTE", "NEI"):
                raise ValueError(f"Invalid verdict label: {lab!r}")
            rec.update({"pred": lab,
                        "key_sentences": [out.get("key_sentence", "")], "reason": out.get("reason", "")})
            ks = rec["key_sentences"][0]
            rec["verbatim"] = citation_checks([ks], hits, rules)
            rec["cite_hit"] = _in_rationale(ks, item.get("gold_rationale") or [])
        elif cond == "closed":
            out = llm.call(CLOSED_SYSTEM, f"[주장] {item['claim']}\n\nlabel / reason 을 submit_verdict 로 제출하라.", FREE_TOOL)
            lab = str(out.get("label", "")).upper()
            if lab not in ("SUPPORT", "REFUTE", "NEI"):
                raise ValueError(f"Invalid verdict label: {lab!r}")
            rec.update({"pred": lab, "reason": out.get("reason", "")})
        else:
            raise ValueError(cond)
    except Exception as e:  # noqa: BLE001
        rec.update({"pred": "ERROR", "error": str(e)[:500]})
    rec["sec"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scifact", "healthver", "checkcovid"])
    ap.add_argument("--claims"); ap.add_argument("--corpus"); ap.add_argument("--csv")
    ap.add_argument("--llm", required=True, help="provider[:model]  예) gemini:gemini-3.1-pro-preview · upstage:solar-pro4")
    ap.add_argument("--cond", nargs="+", choices=["ours", "free", "closed", "norule"], default=["ours", "free", "closed"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1, help="같은 항목을 k 회 (재현성). 2 이상이면 --no-cache 권장")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=DEFAULT_PREDS)
    ap.add_argument("--exclude-items", nargs="*", default=[], help="Pilot JSONL or ID files excluded from this run")
    ap.add_argument("--rules", default=os.path.join(MESH, "verdict_rules.json"))
    ap.add_argument("--k", type=int, default=0, help="문헌 수 상한 (기본 verdict_rules 의 k_evidence)")
    a = ap.parse_args()
    if a.repeat > 1:
        a.no_cache = True

    if a.dataset in ("scifact", "checkcovid"):
        assert a.claims and a.corpus, "--claims --corpus 필요"
        items = load_scifact(a.claims, a.corpus, dataset=a.dataset, limit=a.limit)
    else:
        assert a.csv, "--csv 필요"
        items = load_healthver(a.csv, limit=a.limit)
    items = exclude_items(items, a.exclude_items)
    rules = verdict.load_rules(a.rules)
    if a.k:
        rules["k_evidence"] = a.k
    print(f"[{a.dataset}] {len(items)} items  labels={label_counts(items)}  llm={a.llm}  cond={a.cond}  rules={rules['exceed_threshold']}")
    os.makedirs(a.out, exist_ok=True)
    tag = a.llm.replace(":", "-").replace("/", "_")
    lock = threading.Lock()

    for cond in a.cond:
        for r in range(1, a.repeat + 1):
            suffix = f"_r{r}" if a.repeat > 1 else ""
            path = os.path.join(a.out, f"{a.dataset}_{cond}_{tag}{suffix}.jsonl")
            fp = fingerprint(items, {"llm": a.llm, "cond": cond, "rules": rules, "run": r},
                             [__file__, verdict.__file__, os.path.join(HERE, "llm_client.py"), os.path.join(HERE, "data_loaders.py")])
            done = resume_ids(path, fp)
            todo = [it for it in items if it["id"] not in done]
            print(f"  {cond} run{r}: {len(todo)} to do ({len(done)} done) → {os.path.basename(path)}")
            if not todo:
                continue
            llms = [make_llm(a.llm, cache=not a.no_cache) for _ in range(max(1, a.workers))]
            n_err = 0
            with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
                futs = {ex.submit(run_item, it, cond, llms[i % len(llms)], rules): it for i, it in enumerate(todo)}
                for j, fu in enumerate(as_completed(futs), 1):
                    rec = fu.result()
                    stamp(rec, fp)
                    rec["run"] = r
                    n_err += bool(rec.get("error"))
                    with lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                    if j % 25 == 0 or j == len(todo):
                        print(f"    {j}/{len(todo)}  err={n_err}", flush=True)
            if n_err:
                print(f"  ⚠ {cond}: 오류 {n_err}건 (pred=ERROR) — 재실행 시 오류분 재시도, 보류 정답으로 채점하지 않음.")
    print("done →", a.out)


if __name__ == "__main__":
    main()
