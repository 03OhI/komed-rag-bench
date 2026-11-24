#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — KoMed-RAG End-to-end: 기사(또는 주장) → 사실성 축 × 형식성 축 → 최종 판정

    python pipeline.py "오메가-3가 중성지방을 낮춘다"                 # 주장 한 문장
    python pipeline.py --article 기사.txt [--title "제목"]             # 기사 본문 (주장 추출 → 각 주장 판정 → 문체 → 결합)
    python pipeline.py --provider mock "..."                          # API 키 없이 (규칙 기반 가짜 판독 — 구조 확인용)
    python pipeline.py --json "..."                                   # JSON 으로

흐름 (설계서 3.2 · 3.4 · 4.7)
    기사 ─ claim_extract ─→ 주장들 ─┬─ retrieve.Retriever (동결 파일, 읽기만) ─→ top-k
                                    │      └─ retrieval_evaluator: correct / ambiguous / incorrect
                                    │             correct   → verdict.judge (LLM 판독 + 고정 규칙) → 지지·범위 초과·반박·근거 없음
                                    │             ambiguous → verdict.judge  +  news_fallback (병행)
                                    │             incorrect → 근거 없음      +  news_fallback 만
                                    └─ style_axis (KPF-BERT 또는 자리표시자) ─→ p_abnormal
    결합 (3.2절 고정 규칙 — 학습 가중치 없음)
        어느 주장이든 반박          → 킬스위치: 최종 점수 0 (형식 점수와 무관)
        범위 초과                   → 초과 차원 + 문헌 문장 병기 (킬스위치 아님)
        형식 축 비정상              → S(style) 병기
        셋 다 아님                  → 정상
        전 주장 근거 없음           → 사유별 판단 보류(검색 범위 내 근거 미확인 / 결론 유보 / 혼재) · 의학 개념 없으면 범위 밖

동결 파일(retrieve.py · bm25_util.py · pubmed_index.pkl · eval_claims.tsv)은 import 만 한다. 판정 기간 중 고치지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from claim_extract import extract_claims, mock_claims, real_concepts  # noqa: E402
from komed_llm import LLM                         # noqa: E402
from news_fallback import news_query, search_news  # noqa: E402
from retrieval_evaluator import evaluate, load_thresholds  # noqa: E402
from style_axis import StyleAxis                  # noqa: E402
from verdict import NEI_LABEL, fact_badge, fact_label, judge, load_rules, mock_reading  # noqa: E402

# 화면 라벨은 verdict.fact_label() / fact_badge() — '근거 없음' 은 사유(nei_reason)별로 다르게 쓴다 (2026-09-25).
# 전 주장이 '근거 없음' 일 때 최종 판정에 붙는 사유
NEI_FINAL_REASON = {
    "not_found": "검색 범위에서 주장의 개입·결과를 함께 다룬 문헌을 확인하지 못함 — 효과가 없다는 뜻이 아님",
    "inconclusive": "관련 문헌은 있으나 결론이 불충분·유보 — 범주를 정하지 않음",
    "tied": "지지·반박 근거의 가중 합이 같음 — 범주를 정하지 않음",
    "error": "문헌 판독 실패 — 다시 시도 필요",
}


class KoMedRAG:
    def __init__(self, provider: str = "gemini", model: str | None = None, key: str | None = None,
                 boost: float = 0.0, k: int = 10, news: bool = True, verbose: bool = False,
                 reading_version: str = "legacy"):
        if reading_version not in ('legacy', 'v3'):
            raise ValueError('reading_version must be legacy or v3')
        self.reading_version = reading_version
        from retrieve import Retriever
        self.retriever = Retriever(boost=boost, verbose=verbose)
        self.k = k
        self.use_news = news
        self.thresholds = load_thresholds()
        self.rules = load_rules()
        self.llm = LLM(provider, model=model, key=key,
                       mock_fn={"submit_reading": mock_reading, "submit_claims": mock_claims} if provider == "mock" else None)
        self.style = StyleAxis()
        self.boost = boost

    # ---------------------------------------------------------------- 주장 하나
    def verify_claim(self, claim: str, sentence: str | None = None) -> dict:
        t0 = time.time()
        res = self.retriever.search(claim, k=self.k)
        ev = evaluate(res, self.thresholds)
        if res.get("concepts") and not real_concepts(res["concepts"]):
            # 1음절 동음이의어(이→Tooth 등)만 잡힌 문장은 의학 주장이 아니다 — 검색 계층 동결 중이라 여기서 막는다
            ev = {"verdict": "incorrect", "why": "의학 개념 없음(1음절 동음이의어만 리졸브) — 범위 밖", "signals": res.get("signals")}
            res = {**res, "concepts": [], "hits": [], "signals": {**(res.get("signals") or {}), "n_concepts": 0}}
        news = []
        if ev["verdict"] == "incorrect":
            oos = (res.get("signals") or {}).get("n_concepts", len(res.get("concepts") or [])) == 0
            fact = {"category": "근거 없음", "nei_reason": "out_of_scope" if oos else "not_found",
                    "sub": ("의학 개념이 없는 문장 — 검증 대상 아님" if oos else
                            "검색 단계에서 주장과 맞는 후보 문헌을 확인하지 못함 — 효과가 없다는 뜻이 아님"),
                    "mixed": False, "confidence": None,
                    "support_w": 0, "refute_w": 0, "n_relevant": 0, "n_neutral": 0, "claim_coding": None,
                    "exceeded": [], "evidence": [], "explanation_ko": "", "llm": None,
                    "out_of_scope": oos}
            if self.use_news:
                news = search_news(news_query(claim, res.get("concepts")))
        else:
            judge_fn = judge
            if getattr(self, 'reading_version', 'legacy') == 'v3':
                from verdict_v3 import judge as judge_fn
            fact = judge_fn(claim, res["hits"], self.llm, self.rules)
            if ev["verdict"] == "ambiguous" and self.use_news:
                news = search_news(news_query(claim, res.get("concepts")))
        fact["label"] = fact_label(fact)
        fact["badge"] = fact_badge(fact)
        return {
            "claim": claim, "sentence": sentence or claim,
            "concepts": [{"span": c.get("span"), "name": c.get("name"), "ui": c.get("ui"), "method": c.get("method")}
                         for c in res.get("concepts") or []],
            "retrieval": {"mode": res.get("mode"), "n_cands": res.get("n_cands", 0), "n_scored": res.get("n_scored", 0),
                          "signals": res.get("signals"), "ms": round(res.get("ms", 0)),
                          "hits": [{k: h.get(k) for k in ("rank", "score", "pmid", "year", "journal", "pubtypes", "title")}
                                   for h in res.get("hits", [])]},
            "evaluator": {"verdict": ev["verdict"], "why": ev["why"]},
            "fact": fact, "news": news, "ms": round((time.time() - t0) * 1000),
        }

    # ---------------------------------------------------------------- 기사 전체
    def verify_article(self, text: str, title: str = "", max_claims: int = 3) -> dict:
        t0 = time.time()
        claims = extract_claims(text, self.retriever, self.llm, max_claims=max_claims)
        results = [self.verify_claim(c["claim"], c["sentence"]) for c in claims]
        style = self.style.score(title, text)
        final = combine(results, style)
        return {"input": {"mode": "article", "title": title, "n_chars": len(text)},
                "claims": results, "style": style, "final": final,
                "meta": self._meta(t0)}

    def verify(self, text: str, title: str = "", mode: str = "auto", max_claims: int = 3) -> dict:
        """mode=auto: 120자 이하 한 문장이면 주장, 아니면 기사."""
        if mode == "auto":
            mode = "claim" if len(text.strip()) <= 120 and text.count("다.") <= 1 else "article"
        if mode == "article":
            return self.verify_article(text, title, max_claims)
        t0 = time.time()
        r = self.verify_claim(text.strip())
        return {"input": {"mode": "claim", "title": title, "n_chars": len(text)},
                "claims": [r], "style": None, "final": combine([r], None), "meta": self._meta(t0)}

    def _meta(self, t0):
        return {"llm": self.llm.label, "boost": self.boost, "k": self.k, "style_backend": self.style.backend,
                "reading_version": getattr(self, 'reading_version', 'legacy'),
                "thresholds": {k: v for k, v in self.thresholds.items() if k != "GOOD_MODES"},
                "elapsed_ms": round((time.time() - t0) * 1000)}


def combine(claim_results: list[dict], style: dict | None) -> dict:
    """3.2절 결합 규칙. 학습 가중치 없음 — 판정 근거를 그대로 노출한다."""
    cats = [r["fact"]["category"] for r in claim_results]
    reasons = []
    style_bad = bool(style and style["label"] == "비정상")
    style_score = None if style is None else round((1.0 - style["p_abnormal"]) * 100)   # 형식적 신뢰도
    if "반박" in cats:
        n = cats.count("반박")
        reasons.append(f"근거가 반박하는 주장 {n}건 → 킬스위치 (형식 점수와 무관하게 0점)")
        if style_bad:
            reasons.append(f"형식 축도 비정상 (p={style['p_abnormal']})")
        return {"label": "거짓 — 근거가 주장을 반박", "score": 0, "kill_switch": True, "reasons": reasons,
                "fact_summary": _summary(cats), "style_score": style_score}
    parts = []
    if "범위 초과" in cats:
        dims = sorted({x["name"] for r in claim_results for x in r["fact"]["exceeded"]})
        parts.append("범위 초과")
        reasons.append(f"근거는 실재하나 주장이 {'·'.join(dims)} 차원에서 문헌을 넘어섬 — 초과 차원과 문헌 문장 병기 (킬스위치 아님)")
    if style_bad:
        parts.append("형식 이상")
        cue = ", ".join(f"{k} {len(v)}" for k, v in style["cues"].items() if v)
        reasons.append(f"형식 축 비정상 p={style['p_abnormal']} ({style['backend']}{'; 단서 ' + cue if cue else ''})")
    if parts:
        return {"label": ("근거 있음 · " if style is not None else "") + " · ".join(parts),
                "score": style_score, "kill_switch": False, "reasons": reasons,
                "fact_summary": _summary(cats), "style_score": style_score}
    if cats and all(c == "근거 없음" for c in cats):
        oos = all(r["fact"].get("out_of_scope") for r in claim_results)
        if oos:
            label = "범위 밖"
            reasons.append("의학 개념이 없는 문장 — 검증 범위 밖")
        else:
            kinds = [r["fact"].get("nei_reason") or "not_found" for r in claim_results if not r["fact"].get("out_of_scope")]
            if len(set(kinds)) == 1:
                label = NEI_LABEL.get(kinds[0], "판단 보류")
                reasons.append(NEI_FINAL_REASON.get(kinds[0], "판단 보류"))
            else:
                label = "판단 보류"
                reasons.append("주장별 사유: " + " · ".join(NEI_LABEL.get(k, k).split(" — ")[0] for k in kinds))
            if any(r.get("news") for r in claim_results):
                reasons.append("외부 검색(뉴스) 결과는 참고로만 제시 — 판정에는 쓰지 않음")
        return {"label": label, "score": None, "kill_switch": False,
                "reasons": reasons, "fact_summary": _summary(cats), "style_score": style_score}
    if not cats:
        return {"label": "검증할 주장 없음", "score": None, "kill_switch": False,
                "reasons": ["본문에서 검증 가능한 의학 주장을 찾지 못함"], "fact_summary": {}, "style_score": style_score}
    if style is None:
        reasons.append("근거가 주장을 뒷받침 (형식 축은 기사 본문이 있어야 평가)")
        label = "지지"
    else:
        reasons.append("근거가 주장을 뒷받침하고 형식 축도 정상")
        label = "정상"
    if "근거 없음" in cats:
        held = [r["fact"].get("badge") or "판단 보류" for r in claim_results if r["fact"]["category"] == "근거 없음"]
        reasons.append(f"판단 보류 주장 {len(held)}건({' · '.join(held)})은 결과에 병기")
    return {"label": label, "score": style_score, "kill_switch": False, "reasons": reasons,
            "fact_summary": _summary(cats), "style_score": style_score}


def _summary(cats):
    return {c: cats.count(c) for c in ("지지", "범위 초과", "반박", "근거 없음") if c in cats}


# ─────────────────────────────────────────────────────────────────── CLI
def _print(out: dict):
    print("=" * 78)
    for r in out["claims"]:
        f = r["fact"]
        print(f"주장: {r['claim']}")
        if r["sentence"] != r["claim"]:
            print(f"  (원문) {r['sentence']}")
        print("  개념: " + (" / ".join(f"{c['span']}→{c['name']}" for c in r["concepts"]) or "(없음)"))
        print(f"  검색: {r['retrieval']['mode']} · 후보 {r['retrieval']['n_cands']} · 점수 {r['retrieval']['n_scored']} · "
              f"evaluator={r['evaluator']['verdict']} ({r['evaluator']['why']})")
        print(f"  사실성: {f['label']}  [{f['sub']}]  S={f['support_w']} R={f['refute_w']}"
              + (f"  혼재" if f["mixed"] else ""))
        for x in f["exceeded"]:
            print(f"    ↳ {x['name']}: 주장 '{x['claim_label']}' > 문헌 '{x['evidence_label']}'  (PMID {x['pmid']}) “{x['sentence'][:110]}”")
        for e in f["evidence"]:
            if e["relevant"]:
                print(f"    [{e['n']}] {e['stance']:8s} w={e['weight']:.0f} {e['year']} {','.join(e['pubtypes']) or '-'} PMID {e['pmid']} "
                      f"{'' if e['verbatim'] else '(인용 불일치) '}“{e['key_sentence'][:100]}”")
        if f["explanation_ko"]:
            print("  설명: " + f["explanation_ko"])
        if r["news"]:
            print("  외부 검색: " + " | ".join(n["title"][:50] for n in r["news"][:3]))
        print("-" * 78)
    if out["style"]:
        s = out["style"]
        print(f"형식성: {s['label']} p={s['p_abnormal']} ({s['backend']}) 단서 " +
              "; ".join(f"{k}:{'/'.join(v)}" for k, v in s["cues"].items() if v))
    fin = out["final"]
    print(f"최종: {fin['label']}  점수={fin['score']}  " + " · ".join(fin["reasons"]))
    print(f"({out['meta']['llm']} · {out['meta']['elapsed_ms']} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--article", help="기사 본문 파일")
    ap.add_argument("--title", default="")
    ap.add_argument("--provider", default="gemini", choices=["anthropic", "gemini", "openai", "mock"],
                    help="판독 모델 공급자 (기본 gemini = gemini-3.1-pro-preview · 키 gemini_key.txt / GEMINI_API_KEY)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--boost", type=float, default=0.0)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--reading-version", choices=['legacy', 'v3'], default='legacy',
                    help='v3: 근거 인용 검증·결론 보존·단일 연구 범위 대조 개선판')
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sysm = KoMedRAG(a.provider, a.model, a.key, boost=a.boost, k=a.k, news=not a.no_news,
                   reading_version=a.reading_version)
    if a.article:
        out = sysm.verify(open(a.article, encoding="utf-8").read(), a.title, mode="article")
    else:
        out = sysm.verify(" ".join(a.text) or "오메가-3가 중성지방을 낮춘다", a.title, mode="claim")
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        _print(out)


if __name__ == "__main__":
    main()
