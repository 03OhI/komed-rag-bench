#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exagg.py — E2 범위 초과(인과 강도) 벤치마크 러너

  (1) Wright & Augenstein 2021 과장 탐지 — HF 공개 test 563 쌍 (보도자료 결론문 vs 초록 결론문; 논문 본문은 553 — 문헌 값은 553 기준, 각주)
      예행(프롬프트·접기·규칙 점검, 문헌이 few-shot 학습에 쓴 train 100 쌍 — test 는 건드리지 않는다):
      python run_exagg.py --task exagg --data data/exagg_train.csv --llm gemini:gemini-3.1-pro-preview --mode ours7 ours4 direct
      본실행(프롬프트 동결 뒤 한 번):
      python run_exagg.py --task exagg --data data/exagg_test.csv --llm gemini:gemini-3.1-pro-preview --mode ours7 ours4 direct
        ours7   논문 표 1 의 7단계 인과 강도로 두 문장을 코딩(판독 프롬프트의 정의 그대로) → 4단계로 접음 → 규칙(보도자료 > 초록 = exaggerates)
        ours4   벤치마크의 4단계 정의(0 관계 없음·1 상관·2 조건부 인과·3 인과)로 직접 코딩 → 같은 규칙
        direct  LLM 이 same / exaggerates / downplays 를 직접 출력 (T1 형 — 비교 대조군)
      → 보고: 3분류 macro-F1 (문헌: MT-PET T1 47.35 / T2 57.44 · 61.11), 강도 코딩 정확도·가중 κ
  (2) Yu et al. 2019 인과 언어 4분류 — PubMed 결론문 3,061
      python run_exagg.py --task causal --data data/causal_pubmed.csv --llm ... --mode ours7 bench4
      → 보고: 4분류 macro-F1 (문헌: BioBERT 0.881 · ChatGPT 0.63–0.77)

7 → 4 접기(사전 확정, 바꾸면 여기 기록):  1 진술 없음·2 관계 없음 → no_relationship(0) / 3 상관·4 모호 → correlational(1) /
                                         5 조건부 인과·6 인과 가능 → conditional_causal(2) / 7 명확한 인과 → direct_causal(3)
출력  preds/exagg_<mode>_<llm>.jsonl · preds/causal_<mode>_<llm>.jsonl
무료 크레딧 예행(라벨마다 50문장, 시드 0 — Kim 외 2023 의 개발셋과 같은 크기):
      python run_exagg.py --task causal --data data/causal_pubmed.csv --llm upstage:solar-pro4 --mode ours7 --per-class 50 --out preds_pilot
      예행 결과는 --out 을 따로 두어 본실행 preds/ 와 섞지 않는다(재개 로직이 id 만 보고 건너뛰므로).
변경 이력
    2026-09-27  SYS7 의 인과 강도 정의를 verdict.CAUSAL_SCALE 에서 가져오게 함 — 그 전에는 판독 프롬프트와 다른 예시를 썼다
                ('linked to' 3 · 'may' 4 · 'can' 5). 이제 둘 다 Sumner 2014 원문 예시(4 linked to · 5 might · 6 can). 7→4 접기는 그대로.
                --per-class 추가. 이전 판 claude_backup/run_exagg_0927_before_sumner.py.bak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loaders import load_exagg, load_causal, label_counts  # noqa: E402
from llm_client import make_llm  # noqa: E402
from protocol import DEFAULT_PREDS, fingerprint, resume_ids, stamp, exclude_items
from causal_pair import predict as aligned_predict, GUIDANCE as ALIGNMENT_GUIDANCE

MAP7TO4 = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3}
NAME4 = {0: "no_relationship", 1: "correlational", 2: "conditional_causal", 3: "direct_causal"}
CODE4 = {v: k for k, v in NAME4.items()}

# 논문 판독 프롬프트(verdict.SYSTEM)와 같은 인과 강도 정의 — verdict.CAUSAL_SCALE 을 그대로 쓴다(9/27, Sumner 2014 원문 예시)
from verdict import CAUSAL_SCALE  # noqa: E402
SYS7 = ("당신은 과학 문장의 **인과 강도**를 코딩하는 코더다. 문장이 진술하는 관계의 강도를 아래 7단계 중 하나의 정수로 답한다. "
        "자기 지식으로 사실 여부를 판단하지 말고, **문장이 무엇이라고 말하는지**만 본다.\n " + CAUSAL_SCALE)

SYS4 = """당신은 과학 문장의 인과 언어를 코딩하는 코더다. 문장이 진술하는 관계를 아래 4단계 중 하나로 답한다. 사실 여부가 아니라 **문장의 표현 강도**만 본다.
 0 no_relationship — 관계가 없다고 진술하거나 관계 진술이 없음 · 1 correlational — 상관·연관(associated, linked, correlated) ·
 2 conditional_causal — 조건부·완화된 인과(may/might/could cause, can, in some cases, suggests a causal role) ·
 3 direct_causal — 직접 인과(causes, reduces, prevents, improves, leads to 를 단정)"""

SYS_DIRECT = """당신은 보도자료가 원 논문을 과장했는지 판정한다. 두 문장(보도자료 결론, 논문 초록 결론)을 비교해
 exaggerates = 보도자료가 논문보다 강한 인과·효과를 말함 / downplays = 보도자료가 논문보다 약하게 말함 / same = 같은 강도
 중 하나를 답한다. 사실 여부가 아니라 **표현 강도의 비교**다."""
SYS_DIRECT += "\n판정할 강도 범주의 정의는 코딩 조건과 같다.\n" + SYS4

TOOL_PAIR7 = {"name": "code_strength", "description": "두 문장의 인과 강도(1~7)", "input_schema": {"type": "object",
              "properties": {"s1": {"type": "integer", "minimum": 1, "maximum": 7}, "s2": {"type": "integer", "minimum": 1, "maximum": 7}},
              "required": ["s1", "s2"]}}
TOOL_PAIR4 = {"name": "code_strength", "description": "두 문장의 인과 강도(0~3)", "input_schema": {"type": "object",
              "properties": {"s1": {"type": "integer", "minimum": 0, "maximum": 3}, "s2": {"type": "integer", "minimum": 0, "maximum": 3}},
              "required": ["s1", "s2"]}}
TOOL_ONE7 = {"name": "code_strength", "description": "문장의 인과 강도(1~7)", "input_schema": {"type": "object",
             "properties": {"s": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["s"]}}
TOOL_ONE4 = {"name": "code_strength", "description": "문장의 인과 강도(0~3)", "input_schema": {"type": "object",
             "properties": {"s": {"type": "integer", "minimum": 0, "maximum": 3}}, "required": ["s"]}}
TOOL_DIRECT = {"name": "judge_exaggeration", "description": "과장 판정", "input_schema": {"type": "object",
               "properties": {"label": {"type": "string", "enum": ["same", "exaggerates", "downplays"]}, "reason": {"type": "string"}},
               "required": ["label"]}}


def rule(pr: int, ab: int, th: int = 1) -> str:
    if pr - ab >= th:
        return "exaggerates"
    if ab - pr >= th:
        return "downplays"
    return "same"


def run_exagg_item(it: dict, mode: str, llm) -> dict:
    start = time.perf_counter()
    class CounterLLM:
        def __init__(self, client):
            self.client, self.label, self.calls = client, client.label, 0
        def call(self, *args, **kwargs):
            self.calls += 1
            return self.client.call(*args, **kwargs)
    llm = CounterLLM(llm)
    rec = {"id": it["id"], "task": "exagg", "mode": mode, "llm": llm.label, "gold": it["gold"],
           "pr_strength": it["pr_strength"], "abs_strength": it["abs_strength"]}
    prompt = f"[문장 1 — 보도자료 결론] {it['pr']}\n[문장 2 — 논문 초록 결론] {it['abs']}\n\n"
    try:
        if mode == "aligned4":
            rec.update(aligned_predict(it['pr'], it['abs'], llm, SYS4))
        elif mode == "ours7":
            out = llm.call(SYS7, prompt + "두 문장 각각의 인과 강도(1~7)를 code_strength 로 제출하라.", TOOL_PAIR7)
            s1, s2 = int(out["s1"]), int(out["s2"])
            if s1 not in range(1, 8) or s2 not in range(1, 8):
                raise ValueError("Strength outside 1..7")
            m1, m2 = MAP7TO4[max(1, min(7, s1))], MAP7TO4[max(1, min(7, s2))]
            rec.update({"s1_7": s1, "s2_7": s2, "s1_4": m1, "s2_4": m2,
                        "pred": rule(m1, m2, 1), "pred_th2_on7": rule(s1, s2, 2), "pred_th1_on7": rule(s1, s2, 1)})
        elif mode == "ours4":
            out = llm.call(SYS4, prompt + "두 문장 각각의 인과 강도(0~3)를 code_strength 로 제출하라.", TOOL_PAIR4)
            s1, s2 = int(out["s1"]), int(out["s2"])
            if s1 not in range(4) or s2 not in range(4):
                raise ValueError("Strength outside 0..3")
            rec.update({"s1_4": s1, "s2_4": s2, "pred": rule(s1, s2, 1)})
        elif mode in ("direct", "directreview"):
            out = llm.call(SYS_DIRECT, prompt + "label 을 judge_exaggeration 으로 제출하라.", TOOL_DIRECT)
            if mode == 'directreview':
                rec['draft'] = out
                out = llm.call(SYS_DIRECT + '\n' + ALIGNMENT_GUIDANCE +
                               '\n이번 조건에서는 코딩 객체 대신 최종 label과 짧은 reason을 제출한다.',
                               prompt + '\n다음 초안을 원문과 다시 비교하라. 부정·완화가 다른 관계에 걸렸는지 확인하고 '
                               '오류가 있으면 고쳐 최종 label을 제출하라. 초안은 정답이 아니다.\n' +
                               json.dumps(out, ensure_ascii=False), TOOL_DIRECT)
            lab = str(out.get("label", "")).lower()
            if lab not in ("same", "exaggerates", "downplays"):
                raise ValueError(f"Invalid exaggeration label: {lab!r}")
            rec.update({"pred": lab, "reason": out.get("reason", "")})
        else:
            raise ValueError(mode)
    except Exception as e:  # noqa: BLE001
        rec.update({"pred": "ERROR", "error": str(e)[:500]})
    rec.update(llm_calls=llm.calls, sec=round(time.perf_counter()-start, 4))
    return rec


def run_causal_item(it: dict, mode: str, llm) -> dict:
    rec = {"id": it["id"], "task": "causal", "mode": mode, "llm": llm.label, "gold": it["gold"], "gold_code": it["gold_code"]}
    prompt = f"[문장] {it['sentence']}\n\n"
    try:
        if mode == "ours7":
            out = llm.call(SYS7, prompt + "이 문장의 인과 강도(1~7)를 code_strength 로 제출하라.", TOOL_ONE7)
            s = int(out["s"])
            if s not in range(1, 8):
                raise ValueError("Strength outside 1..7")
            rec.update({"s_7": s, "pred": NAME4[MAP7TO4[max(1, min(7, s))]]})
        elif mode == "bench4":
            out = llm.call(SYS4, prompt + "이 문장의 인과 강도(0~3)를 code_strength 로 제출하라.", TOOL_ONE4)
            s = int(out["s"])
            if s not in range(4):
                raise ValueError("Strength outside 0..3")
            rec.update({"s_4": s, "pred": NAME4[max(0, min(3, s))]})
        else:
            raise ValueError(mode)
    except Exception as e:  # noqa: BLE001
        rec.update({"pred": "ERROR", "error": str(e)[:500]})
    return rec


TASK_MODES = {"exagg": {"ours7", "ours4", "direct", "directreview", "aligned4"}, "causal": {"ours7", "bench4"}}
DEFAULT_MODES = {"exagg": ["ours7", "ours4", "direct"], "causal": ["ours7", "bench4"]}


def select_items(task: str, data: str, limit: int = 0, exclude=(), per_class: int = 0) -> list[dict]:
    """항목 고르기 — main 과 batch_gemini.py collect 가 같은 항목을 쓰도록 한 곳에 둔다 (2026-11-14)."""
    items = load_exagg(data, limit=limit) if task == "exagg" else load_causal(data, limit=limit)
    items = exclude_items(items, list(exclude or []))
    if per_class:
        import random
        rng, by = random.Random(0), {}
        for it in items:
            by.setdefault(it["gold"], []).append(it)
        items = [it for lab in sorted(by) for it in rng.sample(by[lab], min(per_class, len(by[lab])))]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["exagg", "causal"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--mode", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=0,
                    help="라벨마다 N개 고정 표본(시드 0) — 무료 크레딧 예행용. 예행은 --out 을 따로 둘 것")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=DEFAULT_PREDS)
    ap.add_argument("--exclude-items", nargs="*", default=[], help="Exclude pilot JSONL/ID lists from final evaluation")
    a = ap.parse_args()
    if a.per_class and os.path.abspath(a.out) == os.path.abspath(DEFAULT_PREDS):
        ap.error("--per-class is a pilot: supply a separate --out directory")

    items = select_items(a.task, a.data, a.limit, a.exclude_items, a.per_class)
    modes = a.mode or DEFAULT_MODES[a.task]
    fn = run_exagg_item if a.task == "exagg" else run_causal_item
    if not set(modes) <= TASK_MODES[a.task]:
        ap.error("Mode does not belong to this task")
    print(f"[{a.task}] {len(items)} items  labels={label_counts(items)}  llm={a.llm}  modes={modes}")
    os.makedirs(a.out, exist_ok=True)
    tag = a.llm.replace(":", "-").replace("/", "_")
    lock = threading.Lock()
    for mode in modes:
        path = os.path.join(a.out, f"{a.task}_{mode}_{tag}.jsonl")
        fp = fingerprint(items, {"mode": mode, "llm": a.llm},
                         [__file__, os.path.join(MESH, "verdict.py"), os.path.join(MESH, "causal_pair.py"),
                          os.path.join(MESH, "grounded_text.py"), os.path.join(HERE, "data_loaders.py"), os.path.join(HERE, "llm_client.py")])
        done = resume_ids(path, fp)
        todo = [it for it in items if it["id"] not in done]
        print(f"  {mode}: {len(todo)} to do ({len(done)} done) → {os.path.basename(path)}")
        if not todo:
            continue
        llms = [make_llm(a.llm, cache=not a.no_cache) for _ in range(max(1, a.workers))]
        n_err = 0
        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(fn, it, mode, llms[i % len(llms)]) for i, it in enumerate(todo)]
            for j, fu in enumerate(as_completed(futs), 1):
                rec = fu.result()
                stamp(rec, fp)
                n_err += bool(rec.get("error"))
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if j % 50 == 0 or j == len(todo):
                    print(f"    {j}/{len(todo)}  err={n_err}", flush=True)
    print("done →", a.out)


if __name__ == "__main__":
    main()
