#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cost.py — bench_usage.jsonl(토큰 사용량) → 모델별 비용 집계. 요금표는 2026-11-10 기준(USD / 1M tokens, 입력·출력).
가격이 바뀌면 PRICE 만 고친다. 표에 없는 모델은 --price in,out 으로 임시 지정.

    python cost.py                       # 집계
    python cost.py --estimate 10000      # 호출 10,000회 × (2,500 in + 600 out) 가정 시 모델별 예상 비용
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KRW = 1343.0

PRICE = {  # (input, output) USD per 1M
    "gemini-3.1-pro-preview": (2.00, 12.00), "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.8-flash": (0.75, 3.75), "gemini-3.5-flash": (1.50, 9.00), "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5": (5.00, 25.00),
    "gpt-6-luna": (0.10, 0.50), "gpt-5.6-terra": (2.00, 12.00), "gpt-5-mini": (0.25, 2.00), "gpt-5-nano": (0.05, 0.40), "gpt-4.1": (2.00, 8.00), "gpt-4o": (2.50, 10.00),
    "solar-pro4": (0.30, 1.20),   # solar-pro3·pro2·mini 는 2026-10-30 종료
    "deepseek-flash": (0.30, 1.20), "deepseek-v4-pro": (1.32, 3.96),
    "qwen3.8-flash": (0.15, 0.47), "qwen3.7-plus": (0.40, 1.60),
    "HCX-007": (3.72, 3.72),   # ⚠ 추정 — HCX-003 의 0.005원/토큰 앵커. CLOVA Studio 콘솔 요금표에서 확인 후 수정
    "HCX-DASH-002": (0.22, 0.22),
}


def price_for(model: str, override=None):
    if override:
        return override
    m = model.split(":", 1)[-1]
    for k, v in PRICE.items():
        if m.startswith(k) or k in m:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "bench_usage.jsonl"))
    ap.add_argument("--price", default=None, help="in,out USD/1M — 표에 없는 모델용")
    ap.add_argument("--estimate", type=int, default=0, help="호출 수 (2,500 in + 600 out 가정) 예상 비용표 — 상한")
    ap.add_argument("--plan", action="store_true", help="E1·E2 실제 프롬프트 크기로 모델별 비용 (실제 추정 + 상한)")
    a = ap.parse_args()

    if a.plan:
        plan_table()
        return
    override = tuple(float(x) for x in a.price.split(",")) if a.price else None

    if a.estimate:
        n = a.estimate
        print(f"예상 비용 — {n:,} calls × (2,500 in + 600 out)")
        print("| 모델 | USD | KRW |")
        print("|---|---|---|")
        for k, (pi, po) in PRICE.items():
            usd = n * (2500 * pi + 600 * po) / 1e6
            print(f"| {k} | ${usd:,.1f} | ₩{usd*KRW:,.0f} |")
        return
    if not os.path.isfile(a.log):
        print("사용량 로그 없음:", a.log)
        return
    tot = defaultdict(lambda: [0, 0, 0])
    for line in open(a.log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = tot[r["m"]]
        t[0] += 1
        t[1] += int(r.get("in") or 0)
        t[2] += int(r.get("out") or 0)
    print("| 모델 | 호출 | 입력 토큰 | 출력 토큰 | USD | KRW |")
    print("|---|---|---|---|---|---|")
    g = 0.0
    for m, (n, i, o) in sorted(tot.items()):
        p = price_for(m, override)
        usd = (i * p[0] + o * p[1]) / 1e6 if p else None
        g += usd or 0
        print(f"| {m} | {n:,} | {i:,} | {o:,} | {'$%.2f' % usd if usd is not None else '(요금 미등록)'} | {('₩%s' % format(int(usd*KRW), ',')) if usd is not None else ''} |")
    print(f"\n합계 ≈ ${g:.2f} ≈ ₩{g*KRW:,.0f}   (Gemini 는 usage 필드가 없을 수 있음 → AI Studio 대시보드로 확인)")


# ───────────────────────────────────────────── 실험 단위 비용 추정
# (건수, 입력토큰, 출력토큰) — 2026-09-15 실측: verdict.SYSTEM 770 + TOOL 317 = 고정 1,087 토큰,
# FREE 고정 204, CLOSED 고정 161, E2 SYS7 168. 초록 ≈ 330토큰, 문장 ≈ 55토큰.
E1_PROFILE = {"SciFact ours": (450, 1500, 350), "SciFact free": (450, 600, 120), "SciFact closed": (450, 200, 80),
              "HealthVer ours": (1823, 1230, 300), "HealthVer free": (1823, 450, 110), "HealthVer closed": (1823, 200, 80),
              "CheckCOVID ours": (226, 1400, 330), "CheckCOVID free": (226, 550, 120)}
E2_PROFILE = {"exagg ours7": (563, 360, 20), "exagg ours4": (563, 340, 20), "exagg direct": (563, 290, 40),
              "causal ours7": (3061, 300, 15), "causal bench4": (3061, 290, 15)}


def _sum(d, keys=None):
    it = d.items() if keys is None else [(k, v) for k, v in d.items() if any(x in k for x in keys)]
    n = sum(v[0] for _, v in it)
    return n, sum(v[0] * v[1] for _, v in it) / 1e6, sum(v[0] * v[2] for _, v in it) / 1e6


def plan_table():
    combos = {"E1 전체(ours+free+closed) + E2 전체": (_sum(E1_PROFILE), _sum(E2_PROFILE)),
              "E1 ours+free + E2 전체": (_sum(E1_PROFILE, ["ours", "free"]), _sum(E2_PROFILE)),
              "E1 ours+free": (_sum(E1_PROFILE, ["ours", "free"]), (0, 0, 0)),
              "E1 ours 만": (_sum(E1_PROFILE, ["ours"]), (0, 0, 0)),
              "SciFact ours+free": ((900, 450 * (1500 + 600) / 1e6, 450 * (350 + 120) / 1e6), (0, 0, 0))}
    print("모델별 · 범위별 비용 (실제 프롬프트 추정 / 상한 = 전 호출 2,500in+600out 가정)\n")
    for name, ((n1, i1, o1), (n2, i2, o2)) in combos.items():
        n, i, o = n1 + n2, i1 + i2, o1 + o2
        print(f"■ {name} — {n:,} 호출 · 입력 {i:.2f}M · 출력 {o:.2f}M")
        print("  | 모델 | 실제 추정 | 상한 |")
        print("  |---|---|---|")
        for m, (pi, po) in PRICE.items():
            real = (i * pi + o * po)
            cap = (n * 2500 * pi + n * 600 * po) / 1e6
            print(f"  | {m} | ₩{int(real*KRW):,} (${real:.1f}) | ₩{int(cap*KRW):,} |")
        print()
    print("배치 API(OpenAI·Anthropic·Gemini)는 위 값의 절반. 캐시 덕에 중단·재개는 무과금이지만 "
          "**프롬프트를 고치면 캐시가 무효가 되어 전량 재과금** — 예행 50건으로 프롬프트를 확정한 뒤 본실행할 것.")


if __name__ == "__main__":
    main()
