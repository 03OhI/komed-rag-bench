#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_evaluator.py — Retrieval Evaluator (CRAG 식 확신도 판정): 검색 결과 → correct / ambiguous / incorrect

    from retrieval_evaluator import evaluate
    res = Retriever(boost=0.0).search(claim, k=10)
    verdict = evaluate(res)          # {"verdict": "correct"|"ambiguous"|"incorrect", "why": ..., "signals": {...}}

    python retrieval_evaluator.py "오메가-3가 중성지방을 낮춘다"      # 한 줄 판정 (retrieve.py 필요 — 로컬 PC)

설계 (Retrieval_Evaluator_설계_0910.md)
    CRAG(Yan et al. 2024)의 evaluator 는 문서마다 확신도를 매겨 correct/incorrect/ambiguous 로 묶고,
    incorrect 면 외부 검색(우리는 Google News API)으로 보정한다. 우리는 **학습 모델 없이** 검색 계층이 이미
    돌려주는 원신호(signals) 위에 임계값 규칙을 얹는다 — 30건 주장으로 모델을 학습할 수는 없고, 규칙은 감사 가능하다.

    신호 (retrieve.py Retriever.search() 의 res["signals"])
        n_concepts  리졸브된 MeSH 개념 수      0 이면 의학 주장이 아니다 → incorrect (C15 손흥민)
        mode        후보 축소 방식             교집합(전체) > 교집합(2축) > 합집합 > 없음
        n_scored    BM25 점수가 붙은 후보 수    너무 적으면 코퍼스에 문헌이 없다는 뜻
        top_score   1위 BM25 점수              낮으면 질의어와 문서가 거의 안 겹친다
        gap_1_2     1위−2위                    크면 1위가 뚜렷, 작으면 평평

    규칙 (임계값은 thresholds.json — fit_evaluator.py 가 판정 결과로 맞춘다. 없으면 아래 기본값)
        incorrect : n_concepts == 0  또는  n_scored < N_MIN  또는  top_score < T_LOW
        correct   : mode ∈ {교집합(전체), 교집합(2축)}  그리고  top_score ≥ T_HIGH
        ambiguous : 그 외

    후속 동작 (end-to-end 에서)
        correct   → 상위 k 문헌을 근거로 사용
        ambiguous → 문헌 + 외부 검색 병행
        incorrect → 외부 검색(Google News API)만. 의학 개념이 없으면 "범위 밖" 으로 응답
"""
import json, os, sys

DEFAULT = {"N_MIN": 3, "T_LOW": 4.0, "T_HIGH": 8.0, "GOOD_MODES": ["교집합(전체)", "교집합(2축)"]}   # retrieve.py 의 mode 문자열 그대로
_HERE = os.path.dirname(os.path.abspath(__file__))


def load_thresholds(path=None):
    path = path or os.path.join(_HERE, "thresholds.json")
    if os.path.isfile(path):
        d = json.load(open(path, encoding="utf-8"))
        return {**DEFAULT, **d.get("thresholds", d)}
    return dict(DEFAULT)


def _norm_mode(m: str) -> str:
    """'교집합(2축)' 과 '2축 교집합' 을 같은 것으로 본다 (표기 흔들림 방지)"""
    return "".join(sorted(str(m).replace(" ", "").replace("(", "").replace(")", "")))


def evaluate(res: dict, th: dict = None) -> dict:
    th = th or load_thresholds()
    sig = dict(res.get("signals") or {})
    sig.setdefault("n_concepts", len(res.get("concepts") or []))
    sig.setdefault("mode", res.get("mode", "없음"))
    sig.setdefault("n_scored", res.get("n_scored", 0) or 0)
    sig.setdefault("top_score", 0.0)
    mode = sig["mode"]; top = float(sig.get("top_score") or 0.0); n = int(sig.get("n_scored") or 0)
    if sig["n_concepts"] == 0:
        return {"verdict": "incorrect", "why": "의학 개념 없음(리졸브 실패) — 범위 밖", "signals": sig}
    if n < th["N_MIN"]:
        return {"verdict": "incorrect", "why": f"점수 붙은 후보 {n}건 < {th['N_MIN']} — 검색 범위에서 후보 부족", "signals": sig}
    if top < th["T_LOW"]:
        return {"verdict": "incorrect", "why": f"1위 점수 {top:.2f} < {th['T_LOW']} — 질의어와 문서가 거의 안 겹침", "signals": sig}
    good = {_norm_mode(m) for m in th["GOOD_MODES"]}
    if _norm_mode(mode) in good and top >= th["T_HIGH"]:
        return {"verdict": "correct", "why": f"{mode} · 1위 점수 {top:.2f} ≥ {th['T_HIGH']}", "signals": sig}
    return {"verdict": "ambiguous", "why": f"{mode} · 1위 점수 {top:.2f} (문헌 + 외부 검색 병행)", "signals": sig}


if __name__ == "__main__":
    sys.path.insert(0, _HERE)
    from retrieve import Retriever
    claim = " ".join(sys.argv[1:]) or "오메가-3가 중성지방을 낮춘다"
    r = Retriever(boost=0.0, verbose=False)
    res = r.search(claim, k=10)
    v = evaluate(res)
    print(json.dumps({"claim": claim, **v}, ensure_ascii=False, indent=2))
