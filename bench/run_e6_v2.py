"""E6 v2: four labels and retrieval x decision ablations. Production rules are unchanged."""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path

from run_e6 import MESH, HERE, stdrag_hits, make_ours, READING_LINE
import verdict
from llm_client import make_llm
from run_verdict import run_item, citation_checks
from protocol import DEFAULT_PREDS, fingerprint, resume_ids, stamp, digest, exclude_items
from grounded_text import prepare_hits, quote_present, checked_call
import verdict_v3

CONDITIONS = ("closed", "stdrag", "bm25rules", "gatefree", "gaterules", "ours")
CANDIDATE_CONDITIONS = ('stdragv3', 'gatefreev3', 'oursv3')
GATE_CONDITIONS = ('gatefree', 'gaterules', 'ours', 'gatefreev3', 'oursv3')
STD_CONDITIONS = ('stdrag', 'bm25rules', 'stdragv3')
CAT4 = {"지지": "SUPPORT", "범위 초과": "EXCEEDS", "반박": "REFUTE", "근거 없음": "NEI"}
GOLD4 = {**CAT4, "범위초과": "EXCEEDS", "범위 밖": "NEI", "근거없음": "NEI",
         **{x: x for x in CAT4.values()}}
FREE4_TOOL = {"name": "submit_verdict", "description": "근거와 4범주 판정", "input_schema": {
    "type": "object", "properties": {"label": {"type": "string", "enum": list(CAT4.values())},
    "reason": {"type": "string"}, "citations": {"type": "array", "items": {"type": "object",
    "properties": {"n": {"type": "integer"}, "key_sentence": {"type": "string"}},
    "required": ["n", "key_sentence"]}}}, "required": ["label", "reason", "citations"]}}


def free4_system(rules, closed=False):
    definitions = verdict.SYSTEM.split("문헌마다 답할 것", 1)[1].split("explanation_ko  2~4문장", 1)[0]
    return ("주장을 SUPPORT(근거 지지), EXCEEDS(지지 방향 근거는 있으나 표현 범위 초과), "
            "REFUTE(직접 반박 또는 조사된 효과 없음), NEI(판단 보류) 중 하나로 직접 판정한다. "
            "주제만 같은 인접·무관 문헌은 REFUTE가 아니라 NEI다. 다음 관련성·차원 정의를 적용하되 "
            "코딩 재료 대신 최종 label을 직접 제출한다.\n" + definitions +
            "\n관련 지지 문헌 중 차원별 최고 수준과 주장을 비교한다. 초과 차이 문턱: " +
            json.dumps(rules["exceed_threshold"], ensure_ascii=False) + ". 근거 가중: " +
            json.dumps(rules["grade_weight"], ensure_ascii=False) +
            ". 동률·중립만·관련 없음은 NEI, 반박 우세는 REFUTE, 지지 우세는 초과 여부로 나눈다. "
            "reason은 한국어 2~4문장, citations는 문헌 번호와 초록 원문 문장. 인용 없음은 빈 배열. " +
            ("문헌이 없는 조건이다. 자체 지식으로 판단하되 불확실하면 NEI, citations는 빈 배열." if closed else
             "주어진 문헌만 사용한다. 외부 지식으로 빈 근거를 메우지 않는다."))


def load_claims(path=None, expected=None, confirm=False):
    with open(path or os.path.join(MESH, "eval_claims.tsv"), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    updates = {}
    if expected:
        with open(expected, encoding="utf-8-sig", newline="") as f:
            for e in csv.DictReader(f, delimiter="\t"):
                if e["id"] in updates:
                    raise ValueError("Duplicate expected ID: " + e["id"])
                updates[e["id"]] = e
    out, seen = [], set()
    for source in rows:
        r = {**source, **{k: v for k, v in updates.get(source["id"], {}).items() if v}}
        if r["id"] in seen:
            raise ValueError("Duplicate claim ID: " + r["id"])
        seen.add(r["id"])
        exp = (r.get("expected") or "").strip()
        if r["id"] == "C15" and exp == "범위초과":
            exp = "범위 밖"  # Historical C15 meant nonmedical, not exaggeration.
        if exp not in ("", "부분", "논란") and exp not in GOLD4:
            raise ValueError(f"Unknown expected label {exp!r}: {r['id']}")
        g4 = GOLD4.get(exp)
        if confirm and (g4 is None or r.get("split") != "test" or
                        not (r.get("gold_source") or "").strip() or r.get("gold_source") == "legacy_unverified"):
            raise ValueError(f"Confirm requires gold, split=test and independent gold_source: {r['id']}")
        out.append({"id": r["id"], "claim": r["claim"], "expected_raw": exp, "gold4": g4,
                    "gold": "SUPPORT" if g4 == "EXCEEDS" else g4,
                    "meta": {"split": r.get("split", "pilot"), "tags": r.get("tags", ""),
                             "gold_source": r.get("gold_source", "legacy_unverified")}})
    return out


class SnapshotRetriever:
    def __init__(self, snapshot):
        self.snapshot, self.boost = snapshot, 0.0

    def search(self, claim, k=10):
        return self.snapshot[claim]


class CountedLLM:
    def __init__(self, llm):
        self.llm, self.label, self.calls = llm, llm.label, 0

    def call(self, *args, **kwargs):
        self.calls += 1
        return self.llm.call(*args, **kwargs)


def run_one(c, cond, llm, rules, std, gate, sysm=None):
    start = time.perf_counter()
    rec = {**c, "dataset": "e6", "cond": cond, "llm": llm.label}
    try:
        hits = (gate if cond in GATE_CONDITIONS else std).get(c["id"], [])[:rules["k_evidence"]]
        if cond in CANDIDATE_CONDITIONS:
            hits = prepare_hits((gate if cond in GATE_CONDITIONS else std).get(c['id'], []), rules)
        if cond in ("ours", "oursv3"):
            result = sysm.verify_claim(c["claim"])
            fact = result["fact"]
            if fact.get("error") or fact.get("nei_reason") == "error":
                raise ValueError(fact.get("error") or "Pipeline reading failure")
            ev = [e for e in fact.get("evidence", []) if e.get("relevant")]
            rec.update(pred4=CAT4[fact["category"]], category=fact["category"], nei_reason=fact.get("nei_reason"),
                       evaluator=result["evaluator"]["verdict"], n_relevant=fact.get("n_relevant", 0),
                       key_sentences=[e["key_sentence"] for e in ev], explanation_ko=fact.get("explanation_ko", ""))
            if cond == 'oursv3':
                rec['verbatim'] = [quote_present(e['key_sentence'], hits[e['n']-1]['_segments']) for e in ev]
                rec.update(reader_version=verdict_v3.VERSION, reading_audit=fact.get('reading_audit'),
                           scope_profiles=fact.get('scope_profiles'), exceeded=fact.get('exceeded'),
                           shown_evidence=fact.get('shown_evidence'))
            else:
                rec["verbatim"] = citation_checks(rec["key_sentences"], hits, rules, [e["n"] for e in ev])
        elif cond in ("bm25rules", "gaterules"):
            item = {**c, "dataset": "e6", "docs": [{**h, "doc_id": h.get("pmid")} for h in hits]}
            result = run_item(item, "ours", llm, rules)
            if result.get("error"):
                raise ValueError(result["error"])
            rec.update({k: v for k, v in result.items() if k not in ("cond", "gold", "dataset", "sec")})
            rec["pred4"] = CAT4[result["category"]]
        elif cond in ("closed", "stdrag", "gatefree", "stdragv3", "gatefreev3"):
            if cond == "closed":
                hits = []
                prompt = "[주장] " + c["claim"] + "\nlabel / reason / citations를 제출하라."
            else:
                prompt = verdict._prompt(c["claim"], hits, rules).replace(READING_LINE, "label / reason / citations를 제출하라.")
            system = free4_system(rules, cond == "closed")
            if cond in CANDIDATE_CONDITIONS:
                system = system.replace('관련 지지 문헌 중 차원별 최고 수준과 주장을 비교한다.',
                                        '지지 문헌 중 하나라도 주장 수준을 모든 차원에서 함께 충족하는지 비교한다.')
                system += '\n' + verdict_v3.GUIDANCE + '\n이 조건은 코딩 대신 최종 label/reason/citations를 직접 제출한다.'
                def validate_direct(out):
                    if not isinstance(out, dict) or out.get('label') not in CAT4.values() or not isinstance(out.get('citations'), list):
                        raise ValueError('Invalid direct verdict')
                    if out['label'] != 'NEI' and not out['citations']:
                        raise ValueError('A substantive verdict requires a source quotation')
                    for cite in out['citations']:
                        if not isinstance(cite, dict):
                            raise ValueError('Each citation must be an object')
                        n = cite.get('n')
                        if type(n) is not int or not 1 <= n <= len(hits) or not quote_present(cite.get('key_sentence'), hits[n-1]['_segments']):
                            raise ValueError('Citation must match its shown document')
                out, audit = checked_call(llm, system, prompt, FREE4_TOOL, validate_direct)
                rec.update(reading_audit=audit, reader_version='direct-matched-v3',
                           shown_evidence=[{'pmid': h.get('pmid'), 'segments': h['_segments']} for h in hits])
            else:
                out = llm.call(system, prompt, FREE4_TOOL)
            label = str(out.get("label", "")).upper()
            if label not in CAT4.values() or not isinstance(out.get("citations"), list):
                raise ValueError("Invalid four-label verdict or citations")
            cites = out["citations"]
            rec.update(pred4=label, reason=out.get("reason", ""), key_sentences=[e.get("key_sentence", "") for e in cites])
            rec["verbatim"] = ([quote_present(e['key_sentence'], hits[e['n']-1]['_segments']) for e in cites]
                               if cond in CANDIDATE_CONDITIONS else
                               citation_checks(rec["key_sentences"], hits, rules, [e.get("n") for e in cites]))
        else:
            raise ValueError(cond)
        rec["pred"] = "SUPPORT" if rec["pred4"] == "EXCEEDS" else rec["pred4"]
        rec.update(n_hits=len(hits), input_pmids=[h.get("pmid") for h in hits])
    except Exception as exc:
        rec.update(pred="ERROR", pred4="ERROR", error=str(exc)[:500])
    rec.update(sec=round(time.perf_counter() - start, 4), timing_scope="decision_only_retrieval_precomputed")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm")
    ap.add_argument("--cond", nargs="+", choices=CONDITIONS+CANDIDATE_CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--claims"); ap.add_argument("--expected")
    ap.add_argument("--phase", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--exclude-items", nargs="*", default=[])
    ap.add_argument("--export-template")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k-evidence", type=int, default=8)
    ap.add_argument("--k-search", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=DEFAULT_PREDS)
    a = ap.parse_args()
    claims = exclude_items(load_claims(a.claims, a.expected, a.phase == "confirm"), a.exclude_items)
    if a.limit:
        claims = claims[:a.limit]
    if a.phase == "confirm" and {c['gold4'] for c in claims} != set(CAT4.values()):
        ap.error('Confirmatory E6 requires all four gold classes after exclusion/limit')
    if a.export_template:
        with open(a.export_template, "x", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "claim", "expected", "split", "tags", "gold_source"])
            for c in claims:
                writer.writerow([c["id"], c["claim"], c["expected_raw"], "pilot", "", "legacy_unverified"])
        return
    if not a.llm or not claims or a.k_search < a.k_evidence or a.k_evidence < 1:
        ap.error("Require --llm, nonempty claims and k-search >= k-evidence >= 1")
    if a.repeat > 1:
        a.no_cache = True
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    rules = verdict.load_rules()
    rules["k_evidence"] = a.k_evidence
    from retrieve import Retriever
    from retrieval_evaluator import load_thresholds
    evaluator_thresholds = load_thresholds()
    cwd = os.getcwd()
    os.chdir(MESH)
    try:
        retr = Retriever(verbose=False)
        t0 = time.perf_counter()
        inputs = fingerprint(claims, {"k": a.k_evidence, "k_search": a.k_search},
                             [__file__, os.path.join(HERE, "run_e6.py"), os.path.join(MESH, "retrieve.py"),
                              os.path.join(MESH, "bm25_util.py"), os.path.join(MESH, "pubmed_index.pkl")])
        cache_path = os.path.join(a.out, "e6_bm25_" + inputs[:16] + ".json")
        std = stdrag_hits(retr, claims, a.k_search, cache_path) if any(c in a.cond for c in STD_CONDITIONS) else {}
        snapshots = {c["claim"]: retr.search(c["claim"], k=a.k_search) for c in claims} if any(c in a.cond for c in GATE_CONDITIONS) else {}
        gate = {c["id"]: snapshots[c["claim"]]["hits"] for c in claims if c["claim"] in snapshots}
        prep = time.perf_counter() - t0
        tag = a.llm.replace(":", "-").replace("/", "_")
        for cond in a.cond:
            for repeat in range(1, a.repeat + 1):
                suffix = f"_r{repeat}" if a.repeat > 1 else ""
                path = os.path.join(a.out, f"e6_{cond}_{tag}{suffix}.jsonl")
                fp = fingerprint(claims, {"llm": a.llm, "cond": cond, "run": repeat, "rules": rules,
                                          "phase": a.phase, "inputs": inputs, "std": std, "gate": gate,
                                          "evaluator_thresholds": evaluator_thresholds},
                                 [__file__, verdict.__file__, os.path.join(HERE, "run_verdict.py"),
                                  os.path.join(MESH, "pipeline.py"), os.path.join(MESH, "retrieval_evaluator.py"),
                                  os.path.join(MESH, "verdict_v3.py"), os.path.join(MESH, "grounded_text.py"),
                                  os.path.join(HERE, "llm_client.py")])
                done = resume_ids(path, fp)
                llm = CountedLLM(make_llm(a.llm, cache=not a.no_cache))
                sysm = make_ours(SnapshotRetriever(snapshots), llm, a.k_search) if cond in ("ours", "oursv3") else None
                if sysm is not None:
                    sysm.rules = rules
                    sysm.thresholds = evaluator_thresholds
                    sysm.reading_version = 'v3' if cond == 'oursv3' else 'legacy'
                with open(path, "a", encoding="utf-8") as f:
                    for c in claims:
                        if c["id"] in done:
                            continue
                        before = llm.calls
                        rec = run_one(c, cond, llm, rules, std, gate, sysm)
                        rec.update(run=repeat, phase=a.phase, llm_calls=llm.calls - before,
                                   preparation_sec=round(prep, 4), k_evidence=a.k_evidence)
                        f.write(json.dumps(stamp(rec, fp), ensure_ascii=False) + "\n")
                        f.flush()
                print(cond, repeat, "→", path)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
