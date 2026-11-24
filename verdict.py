#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verdict.py — 사실성 축 생성 판정기: 주장 + 검색된 문헌 top-k → 지지 / 범위 초과 / 반박 / 근거 없음

설계 (논문 3.2절 · 3.4절 ⑦ · 1절 수정지시서 ⑥)
    범주 4개  지지 · 범위 초과 · 반박 · 근거 없음
    범위 초과 = 근거는 실재하되 주장이 그 범위를 넘어선 경우. 네 차원으로 조작화:
        인과 강도(7)  진술 없음 < 관계 없음 < 상관 < 모호 < 조건부 인과 < 인과 가능 < 명확한 인과   [Sumner 2014·2016]
        조언 수준(4)  조언 없음 < 암묵적 < 전문가 대상 < 일반인 대상                           [Sumner]
        종간 외삽(3)  명시적 비인간 < 암묵적 인간 < 명시적 인간                                  [Sumner]
        개입 특정성(4) 일반 개입 < 동일 계열 < 동일 성분·다른 제형 < 동일 제품                     [본 연구 정의 — 식약처 고시 2019-67 §3·§7, FTC 2022 §III.B.5]
    "어느 한 차원에서라도 기사의 진술이 문헌보다 상위 단계에 위치할 경우 범위 초과로 판정하고,
     초과가 발생한 차원과 대조에 사용된 문헌 문장을 함께 출력한다. 대조의 기준선은 검색된 문헌의 초록에 기술된 진술로 하며,
     차원별 초과 판정의 임계 단계 차이는 구현 단계에서 조정한다."  ← 임계 단계는 verdict_rules.json (기본값 아래 RULES)

역할 분담 — 감사 가능성
    LLM 이 하는 일: 문헌마다 (관련 여부, 입장 지지/반박/중립, 근거 문장 원문 인용, 네 차원 코딩) + 주장의 네 차원 코딩 + 한국어 설명
    규칙이 하는 일: 범주 결정. 근거등급 가중 합으로 지지/반박, 차원 대조로 범위 초과. **LLM 은 범주를 직접 정하지 않는다.**
    → "학습된 가중치가 아닌 고정 규칙으로 설계되어 최종 판정 근거를 사용자에게 그대로 노출할 수 있다" (3.2절)

    from verdict import judge, fact_label
    fact = judge(claim, hits, llm)      # hits = Retriever.search()["hits"]
    fact_label(fact)                    # 화면 라벨 — '근거 없음' 은 사유(nei_reason)별로 다르다

변경 이력
    2026-09-25  '근거 없음' 의 사유를 nei_reason(not_found · inconclusive · tied · out_of_scope · error)으로 분리하고
                화면 라벨·판정문 지시를 사유별로 바꿈 — 약학 전문가 자문(9/25) 2번: 못 찾음 ≠ 효과 없음 ≠ 반대 근거.
                범주 결정 로직(분기·가중·임계)은 그대로. 효과 없음·반대 방향 보고는 원래대로 refutes → 반박.
    2026-09-26  relevant 정의에 판정기준서 §1-1 규칙 3 세부를 추가(노출을 치료·예방한 연구 · 대상/배경으로만 등장 · 역방향 → false) —
                약학 전문가 자문 3번(직접/간접/무관) 대응: 시스템의 관련 정의를 검색 gold(2점) 정의와 같게 맞춤. 인접 근거는 판정에 쓰지 않는다는
                한계 문장이 참이 되도록. 실 LLM 실행 전이라 무효화할 캐시 없음. 이전 판 claude_backup/verdict_0926_before_rule3.py.bak
    2026-09-27  인과 강도 7단계의 예시 표현을 Sumner 외 2014(BMJ 349:g7015) Methods 원문과 맞춤 — 3 상관 = associated with,
                4 모호 = linked to, 5 조건부 인과 = may·might·could(+likely), 6 인과 가능 = can, 7 = 단정. 이전 판은 may·might 를 4,
                can 을 5 에 두어 한 칸씩 낮았다('연관' 문헌 → '~할 수도 있다' 주장이 2단계가 아닌 1단계 차로 계산돼 범위 초과를 놓침).
                bench/run_exagg.py(E2)의 SYS7 도 또 다른 예시를 쓰고 있었으므로 정의를 CAUSAL_SCALE 하나로 모으고 SYS7 이 가져다 쓰게 함.
                주장 쪽 한국어 예시 한 줄 추가. 단계 이름·임계(2)·분기·가중은 그대로. 실 LLM 판정 결과가 없어 무효화할 것 없음
                (프롬프트가 바뀌므로 캐시 키도 바뀜). 이전 판 claude_backup/verdict_0927_before_sumner.py.bak
"""
from __future__ import annotations

import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = ("지지", "범위 초과", "반박", "근거 없음")

# ── 화면·판정문 라벨 (2026-09-25) ─────────────────────────────────────────────
# '근거 없음' 은 범주 하나에 사유가 셋이다(원고 표 10). 못 찾은 것을 '근거 없음'이라고만 쓰면 '효과가 없다'로 읽히므로
# 화면에는 사유별 문구를 쓴다 — 약학 전문가 자문(9/25) 2번(못 찾음 ≠ 효과 없음 ≠ 반대 근거). 범주 결정 로직은 그대로다.
# 효과 없음·반대 방향을 보고한 문헌은 stance=refutes → 반박(R > S)이지 근거 없음이 아니다.
NEI_LABEL = {
    "not_found":    "검색 범위 내 근거 미확인 — 판단 보류",  # 관련 문헌 0건 (검색 상위 k건에 개입·결과를 함께 다룬 문헌 없음 / 검색 단계 후보 없음)
    "inconclusive": "근거 불충분 — 결론 유보",               # 관련 문헌은 있으나 전부 중립 (S = R = 0)
    "tied":         "근거 혼재 — 판단 보류",                 # 지지·반박 가중 합이 같음 (S = R > 0)
    "out_of_scope": "범위 밖 — 검증할 의학 개념 없음",       # 의학 개념이 리졸브되지 않음 (pipeline, 표 10 밖)
    "error":        "판독 실패 — 판단 보류",                 # LLM 호출 실패 (시스템 오류, 표 10 밖)
}
NEI_BADGE = {"not_found": "근거 미확인", "inconclusive": "결론 유보", "tied": "근거 혼재",
             "out_of_scope": "범위 밖", "error": "판독 실패"}
CAT_LABEL = {
    "지지": "지지 — 근거가 주장을 뒷받침",
    "범위 초과": "범위 초과 — 근거는 있으나 주장이 그 범위를 넘음",
    "반박": "반박 — 근거가 주장과 배치(효과 없음 또는 반대 방향 보고)",
}


def fact_label(fact: dict) -> str:
    """판정 → 화면 라벨. '근거 없음' 은 nei_reason 별 문구(사유가 없으면 not_found 로 본다)."""
    if fact.get("category") == "근거 없음":
        return NEI_LABEL.get(fact.get("nei_reason") or "not_found", NEI_LABEL["not_found"])
    return CAT_LABEL.get(fact.get("category"), fact.get("category") or "")


def fact_badge(fact: dict) -> str:
    """판정 → 배지(짧은 말). '근거 없음' 은 사유별(근거 미확인·결론 유보·근거 혼재·범위 밖·판독 실패)."""
    if fact.get("category") == "근거 없음":
        return NEI_BADGE.get(fact.get("nei_reason") or "not_found", "판단 보류")
    return fact.get("category") or ""


DIMS = {
    "causal": {"name": "인과 강도",
               "levels": ["진술 없음", "관계 없음", "상관", "모호", "조건부 인과", "인과 가능", "명확한 인과"]},
    "advice": {"name": "조언 수준",
               "levels": ["조언 없음", "암묵적 조언", "전문가 대상 조언", "일반인 대상 조언"]},
    "species": {"name": "종간 외삽",
                "levels": ["명시적 비인간", "암묵적 인간", "명시적 인간"]},
    "specificity": {"name": "개입 특정성",
                    "levels": ["일반 개입", "동일 계열 개입", "동일 성분·다른 제형", "동일 제품"]},
}

# 고정 규칙 기본값 — verdict_rules.json 이 있으면 덮어쓴다. 판정 뒤에 바꾸면 HARKing 이므로 여기 기록해 둔다.
RULES = {
    # 차원별 초과 임계(단계 차). 인과·조언은 인접 단계가 모호해 2, 종간·특정성은 한 단계도 실질적이라 1.
    "exceed_threshold": {"causal": 2, "advice": 2, "species": 1, "specificity": 1},
    # 근거등급 가중 — 지지/반박 집계에 쓴다 (retrieve.py 의 EVIDENCE_W 와 같은 유형 집합)
    "grade_weight": {"Meta-Analysis": 3.0, "Systematic Review": 3.0, "Practice Guideline": 3.0,
                     "Randomized Controlled Trial": 2.0, "default": 1.0},
    "k_evidence": 8,          # LLM 에 보여주는 문헌 수 (검색 상위)
    "mixed_ratio": 0.30,      # 소수 입장 가중 비율이 이 이상이면 '혼재' 표시. 가중 합이 같으면 범주를 정하지 않는다(근거 없음·판단 보류)
    "abstract_chars": 1800,   # 문헌당 초록 최대 길이 (비용)
}


def load_rules(path: str | None = None) -> dict:
    path = path or os.path.join(HERE, "verdict_rules.json")
    r = json.loads(json.dumps(RULES))
    if os.path.isfile(path):
        d = json.load(open(path, encoding="utf-8"))
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(r.get(k), dict):
                r[k].update(v)
            else:
                r[k] = v
    return r


# ─────────────────────────────────────────────────────────────── LLM 인터페이스
# 인과 강도 7단계의 정의와 예시 — Sumner 외 2014(BMJ 349:g7015) Methods 의 척도를 그대로 따른다(2026-09-27 정렬).
#   3 상관 'is associated with' · 4 모호 'linked to' · 5 조건부 인과 'might increase' · 6 인과 가능 'can increase' · 7 'increases'
# bench/run_exagg.py(E2 인과 언어·과장 벤치마크)도 이 문자열을 가져다 쓴다 — 판독 모델과 벤치마크가 같은 정의로 코딩하도록.
CAUSAL_SCALE = ("1 진술 없음 · 2 관계 없음(효과/연관이 없다고 진술) · 3 상관(associated with, correlated with) · "
                "4 모호(linked to, tied to) · 5 조건부 인과(may/might/could/likely + 인과 동사, 예: might increase) · "
                "6 인과 가능(can + 인과 동사, 예: can increase) · 7 명확한 인과(causes/increases/reduces/prevents 를 단정)")

SYSTEM = """당신은 한국어 건강뉴스 주장을 PubMed 문헌과 **대조**하는 판독기다. 자기 지식으로 주장의 진위를 정하는 것이 아니라,
주어진 문헌의 초록에 **무엇이 쓰여 있는지**를 주장과 대조해 구조화된 판독을 제출한다. 최종 범주(지지/범위 초과/반박/근거 없음)는
당신이 아니라 고정 규칙이 정한다. 당신은 재료만 낸다.

문헌마다 답할 것
 relevant   이 문헌이 주장의 개입·노출과 결과지표를 함께 다루는가 (주제만 겹치면 false). 대상 집단이 다르다고 false 로 만들지 말 것 —
            종간 외삽 차원에서 다룬다.
            주장의 개입·노출은 문헌에서 **처치·노출·위험요인**으로 다뤄져야 true 다. 그 노출을 치료·예방한 개입 연구, 노출이 연구 대상·배경으로만
            나오는 문헌(치주질환 환자에게 다른 약), 결과→노출의 역방향 연구는 false — 인접 근거이지 주장의 대조 근거가 아니다(관련성 판정기준서 §1-1 규칙 3).
 stance     relevant 인 문헌만. supports = 초록의 결론이 주장 방향과 같다 / refutes = 결론이 주장과 배치되거나 효과 없음·무관을 보고 /
            neutral = 관련은 있으나 결론이 불충분·혼재·판단 유보. 지지 여부를 결과의 통계적 유의성과 결론 문장으로 판단한다.
 key_sentence   판단의 근거가 된 초록 문장 **한 개를 영어 원문 그대로** 복사한다(바꿔 쓰지 말 것). 결론(Conclusion) 문장을 우선.
 key_sentence_ko  그 문장의 한국어 번역.
 coding     이 문헌의 초록이 진술하는 수준 — 네 차원:
   causal  {CAUSAL_SCALE}
   advice  1 조언 없음 · 2 암묵적(임상적 함의 언급) · 3 전문가 대상 권고 · 4 일반인 대상 권고
   species 1 명시적 비인간(동물·세포·시험관) · 2 암묵적 인간(인간 여부 불명확) · 3 명시적 인간(사람 대상 연구)
   specificity  이 문헌의 개입이 **주장의 개입**과 얼마나 같은가: 1 일반 개입(같은 범주의 넓은 행위·식이) · 2 동일 계열(같은 약물군·영양소군) ·
                3 동일 성분·다른 제형(같은 성분이지만 용량·제형·배합이 다르거나 불특정) · 4 동일 제품(주장이 말한 바로 그 제품·제형·용량)

주장에 대해 답할 것 — claim_coding (같은 네 차원, 주장 문장이 진술하는 수준):
   causal   주장 문장의 인과 표현 강도(위 7단계). 한국어 예: '~와 관련·연관이 있다' 3 · '~할 수도 있다·~일 가능성이 있다' 5 ·
            '~할 수 있다·~에 도움이 될 수 있다' 6 · '~한다·낮춘다·예방한다'(단정) 7.
   advice   주장이 독자에게 하는 조언의 수준(제품 복용 권유 = 4).
   species  주장이 말하는 대상(사람이면 3).  specificity  주장이 특정 제품(상표·제형·용량)을 말하면 4, 특정 성분이면 3, 계열이면 2, 일반이면 1.

explanation_ko  2~4문장. 문헌에 쓰인 것만으로 쓴다. 문헌 번호([1] 등)를 달아 어느 문헌에서 온 말인지 보이게 한다.
                주장이 옳다/틀리다를 단정하지 말고 "문헌은 ~라고 보고한다" 식으로 쓴다. 문헌에 없는 지식을 보태지 않는다.
                관련 문헌이 없거나 결론이 유보됐으면 "근거가 없다"고 쓰지 말고 "검색된 문헌에서는 확인되지 않는다"처럼 쓴다.
                찾지 못한 것은 효과가 없다는 뜻이 아니다. 문헌이 효과 없음을 보고했으면 그 결과를 그대로 쓴다("효과가 없었다고 보고한다").

주의: 주장을 반박하는 문헌도 relevant=true 다. 관련성은 지지가 아니다."""
SYSTEM = SYSTEM.replace("{CAUSAL_SCALE}", CAUSAL_SCALE)

# ── 판독기 v2 (2026-11-15) — 주장 유형별 관련성 기준 ─────────────────────────────
# v1 의 relevant 정의는 개입·노출 → 결과 주장에만 맞는다. 전파 경로·성질·기전 같은 사실 주장과
# "~라는 근거는 없다" 같은 부정·메타 주장은 개입·결과 쌍이 없어 관련 문헌이 있어도 전부 not_found 로 떨어졌다
# (HealthVer test 11/13: 정답 지지·반박 386건이 관련 0건). 개입·결과 주장의 기준(9/26 §1-1 규칙 3)은 그대로 두고,
# 그 밖의 주장에만 '대상과 속성을 직접 진술' 기준을 더한다. rules["reader"] == "v2" 일 때만 쓴다 — v1 재현용 SYSTEM 은 그대로.
_V1_REL = """ relevant   이 문헌이 주장의 개입·노출과 결과지표를 함께 다루는가 (주제만 겹치면 false)."""
_V2_REL = """ relevant   먼저 주장의 유형을 본다.
            (가) 개입·노출 → 결과 주장(~가 ~을 낮춘다·예방한다·위험을 높인다·치료한다): 이 문헌이 주장의 개입·노출과 결과지표를
                 함께 다루는가 (주제만 겹치면 false). 아래 세부 규칙을 그대로 적용한다.
            (나) 그 밖의 주장 — 전파 경로·발생·기원 같은 사실 주장, 병원체·질병의 성질·기전 주장, 권고·지침 주장,
                 "~라는 근거는 없다"·"~는 사실이 아니다" 같은 부정·메타 주장: 이 문헌이 주장이 말하는 **대상**과 **그 대상에 대한
                 진술 내용**(경로·성질·권고·근거 유무)을 직접 다루면 true. 주제(예: 같은 질병)만 겹치면 false.
                 부정·메타 주장의 stance: 문헌이 그 효과를 부정하거나 근거 부족·허위 정보로 보고하면 supports, 효과를 확인하면 refutes."""
assert _V1_REL in SYSTEM
SYSTEM_V2 = SYSTEM.replace(_V1_REL, _V2_REL)


# ── 판독기 S (2026-11-15) — 문헌별 성립 여부(sufficiency) ─────────────────────────
# v1 판독기는 문헌 결론의 '방향'(stance)만 묻는다. 직접 판정은 "이 문헌으로 주장이 성립하는가"를 묻는다.
# 방향만 맞으면 지지로 셈해져 틀린 확신이 직접 판정의 두 배였다(실험 1, 11/13). 판독기가 설명문에 이미 쓰는
# "~는 확인되지 않는다"를 칸으로 꺼내 규칙이 쓰게 한다. rules["reader"] == "suff" 일 때만 쓴다 — v1 문자열은 그대로.
# 설계: claude_backup/설계_판독기v3_성립여부_1115.md
_V1_STANCE_END = """            neutral = 관련은 있으나 결론이 불충분·혼재·판단 유보. 지지 여부를 결과의 통계적 유의성과 결론 문장으로 판단한다."""
_SUFF = """
 sufficiency  relevant 이고 stance 가 supports·refutes 인 문헌만. "이 문헌만으로 주장의 **핵심 진술**이 그 방향으로 판가름 나는가"를
            답한다. 핵심 진술 = 주장이 독자에게 주로 단언하는 내용(누가 · 무엇이 · 어떤 결과). 주장이 여러 문장이면 주된 단언 하나를 핵심으로
            잡는다. 배경 설명 · 출처 · 날짜 · 시험 개수 같은 부수 세부, 예시로 든 목록의 나머지 항목은 핵심이 아니다 — 문헌에 없어도
            partial 사유가 되지 않는다. 지지와 반박은 다르게 본다 — 겹친 주장은 핵심 부분이 모두 맞아야 참이고, 하나만 틀려도 거짓이다.
            stance = supports
              full     핵심 진술이 문헌에 있다.
              partial  핵심 진술을 이루는 부분 가운데 하나가 문헌에 없다 — 주장이 대등하게 내세운 두 결과 중 하나(착용자 보호와 타인
                       보호), 주장의 핵심 대상 일부(성인과 아이를 함께 단언했는데 아이 자료가 없음), 핵심으로 내세운 수치(두 배 이상)처럼.
              none     주제와 방향은 비슷하나 핵심 진술을 판가름하지 못한다 — 막연한 주장에 주제만 맞는 문헌, 다른 질환·다른 대상의
                       결과에서 추론을 거쳐야 이어지는 문헌.
            stance = refutes
              full     문헌이 핵심 진술 가운데 **하나라도** 직접 반대 결과로 보고한다. 주장이 단언한 효과·전파·위험이 없었다거나 그런
                       근거가 없다고 문헌이 보고하는 것도 반대 결과다(효과 없음 · 반대 방향 · 주장한 수치에 못 미침 · 근거 없음 보고).
              none     그렇지 않다. (반박에는 partial 을 쓰지 않는다.)
            강도만 다르면 full 이다 — 문헌이 상관·가능성·동물·같은 계열 수준으로 말하는데 주장이 인과·단정·사람·특정 제품으로 말하는
            경우처럼 인과 강도 · 조언 · 종 · 특정성 네 차원의 차이는 sufficiency 에서 따지지 말고 coding 으로만 답한다(규칙이 범위 초과로
            따로 비교한다). '완치한다'처럼 결과 자체가 다른 것(사망 감소 ≠ 완치)은 강도가 아니라 핵심 진술의 차이다."""
assert _V1_STANCE_END in SYSTEM
SYSTEM_SUFF = SYSTEM.replace(_V1_STANCE_END, _V1_STANCE_END + _SUFF)
SUFF_VALUES = ("full", "partial", "none")


def system_for(rules: dict) -> str:
    """rules["reader"] 에 맞는 판독기 프롬프트. 기본 v1(11/13 본실행과 같은 문자열 — 캐시 키 유지)."""
    reader = (rules or {}).get("reader")
    if reader == "suff":
        return SYSTEM_SUFF
    return SYSTEM_V2 if reader == "v2" else SYSTEM

TOOL = {
    "name": "submit_reading",
    "description": "주장과 문헌 대조 결과를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claim_coding": {"type": "object", "properties": {
                "causal": {"type": "integer", "minimum": 1, "maximum": 7},
                "advice": {"type": "integer", "minimum": 1, "maximum": 4},
                "species": {"type": "integer", "minimum": 1, "maximum": 3},
                "specificity": {"type": "integer", "minimum": 1, "maximum": 4}},
                "required": ["causal", "advice", "species", "specificity"]},
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "n": {"type": "integer", "description": "문헌 번호 [n]"},
                "relevant": {"type": "boolean"},
                "stance": {"type": "string", "enum": ["supports", "refutes", "neutral", ""]},
                "key_sentence": {"type": "string"},
                "key_sentence_ko": {"type": "string"},
                "coding": {"type": "object", "properties": {
                    "causal": {"type": "integer", "minimum": 1, "maximum": 7},
                    "advice": {"type": "integer", "minimum": 1, "maximum": 4},
                    "species": {"type": "integer", "minimum": 1, "maximum": 3},
                    "specificity": {"type": "integer", "minimum": 1, "maximum": 4}}},
                "note": {"type": "string"}},
                "required": ["n", "relevant"]}},
            "explanation_ko": {"type": "string"}},
        "required": ["claim_coding", "evidence", "explanation_ko"]}}

TOOL_SUFF = copy.deepcopy(TOOL)
TOOL_SUFF["input_schema"]["properties"]["evidence"]["items"]["properties"]["sufficiency"] = {
    "type": "string", "enum": list(SUFF_VALUES)}


def tool_for(rules: dict) -> dict:
    """rules["reader"] 에 맞는 판독 도구. v1·v2 는 TOOL 객체 그대로(캐시 키·요청 본문 유지)."""
    return TOOL_SUFF if (rules or {}).get("reader") == "suff" else TOOL


def _prompt(claim: str, hits: list[dict], rules: dict) -> str:
    lines = [f"[주장] {claim}", "", "[문헌]"]
    for i, h in enumerate(hits, 1):
        pt = ", ".join(h.get("pubtypes") or []) or "-"
        ab = re.sub(r"\s+", " ", h.get("abstract") or "")[:rules["abstract_chars"]]
        lines.append(f"[{i}] PMID {h.get('pmid')} · {h.get('year')} · {pt}\n제목: {h.get('title')}\n초록: {ab}\n")
    lines.append("각 문헌에 대해 relevant/stance/key_sentence/coding 을, 주장에 대해 claim_coding 을, 그리고 explanation_ko 를 submit_reading 으로 제출하라.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────── 고정 규칙
def _w(h: dict, rules: dict) -> float:
    gw = rules["grade_weight"]
    return max([gw.get(p, 0.0) for p in (h.get("pubtypes") or [])] + [gw["default"]])


def _norm_space(s: str) -> str:
    return re.sub(r"[\s\W]+", " ", (s or "").lower()).strip()


def decide(claim: str, hits: list[dict], reading: dict, rules: dict) -> dict:
    """LLM 판독(reading) + 문헌 메타(hits) → 범주. 전부 결정적."""
    by_n = {i + 1: h for i, h in enumerate(hits)}
    cc = {d: int(reading.get("claim_coding", {}).get(d) or 1) for d in DIMS}
    suff = (rules or {}).get("reader") == "suff"
    n_partial = n_insufficient = n_suff_missing = 0
    ev_out, S, R, N = [], 0.0, 0.0, 0
    for e in reading.get("evidence", []):
        n = int(e.get("n") or 0)
        h = by_n.get(n)
        if not h:
            continue
        rel = bool(e.get("relevant"))
        st = (e.get("stance") or "").strip() if rel else ""
        if rel and st not in ("supports", "refutes", "neutral"):
            st = "neutral"
        ks = (e.get("key_sentence") or "").strip()
        verbatim = bool(ks) and _norm_space(ks) in _norm_space((h.get("abstract") or "") + " " + (h.get("title") or ""))
        cod = {d: int((e.get("coding") or {}).get(d) or 1) for d in DIMS}
        w = _w(h, rules)
        sv = None                                   # 판독기 S 의 성립 여부 (supports·refutes 문헌만)
        if suff and rel and st in ("supports", "refutes"):
            sv = str(e.get("sufficiency") or "").strip().lower()
            if sv not in SUFF_VALUES:               # 누락·형식 오류 → v1 과 같은 동작(full)으로 두고 센다
                n_suff_missing += 1
                sv = "full"
            if st == "refutes" and sv == "partial":  # 반박에는 partial 이 없다 (설계 §4-1)
                sv = "none"
            if sv == "partial":
                n_partial += 1
            elif sv == "none":
                n_insufficient += 1
        counted = sv in (None, "full")
        if rel and st == "supports" and counted:
            S += w
        elif rel and st == "refutes" and counted:
            R += w
        elif rel:
            N += 1
        ev_out.append({"n": n, "pmid": h.get("pmid"), "title": h.get("title"), "year": h.get("year"),
                       "pubtypes": h.get("pubtypes") or [], "journal": h.get("journal"), "rank": h.get("rank"),
                       "relevant": rel, "stance": st, "weight": w if rel else 0.0,
                       "key_sentence": ks, "key_sentence_ko": (e.get("key_sentence_ko") or "").strip(),
                       "verbatim": verbatim, "coding": cod if rel else None, "note": (e.get("note") or "").strip()})
        if suff:
            ev_out[-1]["sufficiency"] = sv

    relevant = [e for e in ev_out if e["relevant"]]
    result = {"category": None, "nei_reason": None, "sub": "", "mixed": False, "confidence": None,
              "support_w": S, "refute_w": R, "n_relevant": len(relevant), "n_neutral": N,
              "claim_coding": cc, "exceeded": [], "evidence": ev_out,
              "explanation_ko": (reading.get("explanation_ko") or "").strip()}
    if suff:
        result.update({"n_partial": n_partial, "n_insufficient": n_insufficient, "n_suff_missing": n_suff_missing})
    if not relevant:
        result["category"] = "근거 없음"
        result["nei_reason"] = "not_found"
        result["sub"] = f"검색 상위 {len(hits)}건 중 주장의 개입·결과를 함께 다룬 문헌 없음 — 효과가 없다는 뜻이 아님"
        return result
    if S + R == 0:
        result["category"] = "근거 없음"
        if suff and n_partial:
            result["nei_reason"] = "partial"
            result["sub"] = "주장의 일부만 문헌에서 확인됨 — 효과가 없다는 뜻이 아님"
            return result
        result["nei_reason"] = "inconclusive"
        result["sub"] = f"관련 문헌 {len(relevant)}건이 있으나 모두 결론 불충분·유보(중립)"
        return result
    result["confidence"] = round(abs(S - R) / (S + R), 3)
    result["mixed"] = min(S, R) / (S + R) >= rules["mixed_ratio"]
    if S == R:
        result["category"] = "근거 없음"
        result["nei_reason"] = "tied"
        result["sub"] = "지지·반박 근거의 가중 합이 같음 — 판단 보류(혼재)"
        result["mixed"] = True
        return result
    if R > S:
        result["category"] = "반박"
        result["sub"] = "근거등급 가중 합에서 반박(효과 없음·반대 방향 보고)이 우세" + (" (지지 근거도 있음 — 혼재)" if result["mixed"] else "")
        return result
    # 지지 우세 → 범위 초과 검사: 지지 문헌이 진술하는 최고 수준을 기준선으로
    sup = [e for e in relevant if e["stance"] == "supports" and e.get("sufficiency", "full") == "full"]
    th = rules["exceed_threshold"]
    for d, meta in DIMS.items():
        best = max(sup, key=lambda e: e["coding"][d])
        ev_level = best["coding"][d]
        if cc[d] - ev_level >= th[d]:
            result["exceeded"].append({
                "dim": d, "name": meta["name"],
                "claim_level": cc[d], "claim_label": meta["levels"][cc[d] - 1],
                "evidence_level": ev_level, "evidence_label": meta["levels"][ev_level - 1],
                "pmid": best["pmid"], "sentence": best["key_sentence"], "sentence_ko": best["key_sentence_ko"]})
    if result["exceeded"]:
        result["category"] = "범위 초과"
        result["sub"] = "근거는 실재하나 주장이 " + "·".join(x["name"] for x in result["exceeded"]) + " 차원에서 문헌을 넘어섬"
    else:
        result["category"] = "지지"
        result["sub"] = "근거등급 가중 합에서 지지가 우세" + (" (반박 근거도 있음 — 혼재)" if result["mixed"] else "")
    return result


def judge(claim: str, hits: list[dict], llm, rules: dict | None = None) -> dict:
    """주장 + 검색 결과 → 사실성 판정. LLM 실패 시 '근거 없음'(nei_reason=error, 판독 실패)으로 강등한다."""
    rules = rules or load_rules()
    hits = list(hits or [])[:rules["k_evidence"]]
    if not hits:
        return {"category": "근거 없음", "nei_reason": "not_found", "sub": "검색 단계에서 후보 문헌을 찾지 못함 — 효과가 없다는 뜻이 아님", "mixed": False, "confidence": None,
                "support_w": 0, "refute_w": 0, "n_relevant": 0, "n_neutral": 0, "claim_coding": None,
                "exceeded": [], "evidence": [], "explanation_ko": "", "llm": None}
    try:
        reading = llm.call(system_for(rules), _prompt(claim, hits, rules), tool_for(rules))
    except Exception as e:                       # noqa: BLE001
        return {"category": "근거 없음", "nei_reason": "error", "sub": f"판독 실패: {e}", "mixed": False, "confidence": None,
                "support_w": 0, "refute_w": 0, "n_relevant": 0, "n_neutral": 0, "claim_coding": None,
                "exceeded": [], "evidence": [], "explanation_ko": "", "llm": llm.label, "error": str(e)}
    out = decide(claim, hits, reading, rules)
    out["llm"] = llm.label
    return out


# ───────────────────────────────────────────────────────────────── mock 판독기
_NEG = re.compile(r"\b(no (?:significant |clear |strong )?(?:effect|benefit|association|evidence|difference|reduction)|"
                  r"not (?:associated|effective|significant|reduce|improve|support)|did not|does not|failed to|"
                  r"no relationship|no link)\b", re.I)
_INCONC = re.compile(r"\b(insufficient evidence|lack of evidence|inconclusive)\b", re.I)   # 프롬프트 정의대로 neutral (9/25)
_CAUSAL = re.compile(r"\b(reduc|improv|prevent|lower|decreas|increas|protect|effective|efficac|benefi)", re.I)


def mock_reading(system: str, prompt: str, tool: dict) -> dict:
    """키 없이 파이프라인을 돌려 보기 위한 규칙 기반 판독. 실제 판정에 쓰지 않는다."""
    claim = prompt.split("\n", 1)[0].replace("[주장]", "").strip()
    blocks = re.findall(r"\[(\d+)\] PMID (\S+) · (\S+) · ([^\n]*)\n제목: ([^\n]*)\n초록: ([^\n]*)", prompt)
    ev = []
    for n, pmid, year, pt, title, ab in blocks:
        sents = re.split(r"(?<=[.!?])\s+", ab)
        neg = [s for s in sents if _NEG.search(s)]
        inc = [s for s in sents if _INCONC.search(s)]
        pos = [s for s in sents if _CAUSAL.search(s)]
        if neg:
            st, ks = "refutes", neg[-1]
        elif inc:
            st, ks = "neutral", inc[-1]
        elif pos:
            st, ks = "supports", pos[-1]
        else:
            st, ks = "neutral", (sents[-1] if sents else "")
        rel = int(n) <= 5 and bool(ab)
        ev.append({"n": int(n), "relevant": rel, "stance": st if rel else "",
                   "key_sentence": ks[:300], "key_sentence_ko": "(mock 번역) " + ks[:80],
                   "coding": {"causal": 3 if st == "neutral" else 6, "advice": 2,
                              "species": 1 if re.search(r"\b(mice|rats?|in vitro|cell)\b", ab, re.I) else 3,
                              "specificity": 3},
                   "note": "mock"})
    cc = {"causal": 7 if re.search("낮춘다|예방|치료|유발", claim) else 5,
          "advice": 4 if re.search("복용하면|드세요|추천", claim) else 2,
          "species": 3,
          "specificity": 4 if re.search("제품|캡슐|정\b|㈜", claim) else 3}
    return {"claim_coding": cc, "evidence": ev,
            "explanation_ko": f"(mock) 관련 문헌 {sum(e['relevant'] for e in ev)}건 중 "
                              f"지지 {sum(e['stance']=='supports' for e in ev)}건 · 반박 {sum(e['stance']=='refutes' for e in ev)}건."}


if __name__ == "__main__":
    import sys
    from komed_llm import LLM
    from retrieve import Retriever
    claim = " ".join(sys.argv[1:]) or "오메가-3가 중성지방을 낮춘다"
    r = Retriever(boost=0.0, verbose=False)
    res = r.search(claim, k=10)
    llm = LLM("mock", mock_fn=mock_reading, cache_path=None)
    print(json.dumps(judge(claim, res["hits"], llm), ensure_ascii=False, indent=1)[:3000])
