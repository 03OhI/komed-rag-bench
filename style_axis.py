#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_axis.py — 형식성 축 (KPF-BERT 문체 분류) 인터페이스

    from style_axis import StyleAxis
    sa = StyleAxis()                       # kpf_bert_style/ 폴더에 모델이 있으면 KPF-BERT, 없으면 규칙 기반 자리표시자
    sa.score(title, text) → {"p_abnormal": 0.83, "label": "비정상", "backend": "kpf-bert"|"heuristic", "cues": {...}}

백엔드 (있는 것부터 자동 선택)
    kpf-bert×3  `kpf_bert_광고성/` `kpf_bert_자극성/` `kpf_bert_낚시성/` — train_kpf_bert.py --type 으로 만든 유형별 모델 셋.
                3.3절 그대로: 세 분류기가 독립적으로 확률을 내고, p_abnormal = 세 확률의 최댓값, 세 값이 종합 추론기로 간다.
    kpf-bert    `kpf_bert_style/` 이진 모델 하나 (정상 vs INEC 전체). 유형 모델이 없을 때.
    heuristic   모델이 없을 때의 자리표시자 — INEC 기사심의규정 세 유형 단서어 계수. **논문 수치로 쓰지 않는다.**
    label 1 = 해당 유형(또는 비정상), 0 = 아님 — build_train_data.py · train_kpf_bert.py 규약. 본문은 학습 때와 같은 scrub 을 거친다.

라벨 출처 (9/10 정정)
    심의결정문 전문의 '적용조항'에서 유형이 나온다 (build_style_labels.py): 광고성 5,918 · 자극성 1,792 · 낚시성 388.
    9/6 문서의 "결정문에 조항이 없다"는 크롤 필드만 본 결론이었다.
"""
from __future__ import annotations

import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "kpf_bert_style")
TYPES = ("광고성", "자극성", "낚시성")
TYPE_DIRS = {t: os.path.join(HERE, f"kpf_bert_{t}") for t in TYPES}
_NEG = {"정상", "아님", "0", "label_0", "normal", "non"}

# build_train_data.py 의 _SCRUB 와 같은 규칙 (매체 표지 제거)
_SCRUB = [
    (re.compile(r"https?://\S+"), " "), (re.compile(r"[\w.\-]+@[\w.\-]+"), " "),
    (re.compile(r"한겨레(?:신문|21|S)?"), " "), (re.compile(r"hani\.co\.kr|hani21|한겨레닷컴"), " "),
    (re.compile(r"\[[^\]]{0,30}(?:기자|특파원|통신원)[^\]]{0,30}\]"), " "),
    (re.compile(r"[가-힣]{2,4}\s?(?:기자|특파원|논설위원|편집위원|선임기자)"), " "),
    (re.compile(r"무단\s?전재\s?(?:및|and)?\s?재배포\s?금지"), " "), (re.compile(r"저작권자?\s?[ⓒ©].{0,40}"), " "),
    (re.compile(r"[ⓒ©]\s?\S{0,20}"), " "), (re.compile(r"\S{0,10}(?:구독하기|연재\s?구독|뉴스레터).{0,20}"), " "),
    (re.compile(r"\[(?:AP신문|뉴시스|연합뉴스|서울와이어|마이데일리)[^\]]{0,30}\]"), " "),
    (re.compile(r"(?:AP신문|뉴시스|연합뉴스|메디컬투데이|서울와이어|마이데일리)"), " "),
    (re.compile(r"\s{2,}"), " "),
]


def scrub(t: str) -> str:
    for pat, rep in _SCRUB:
        t = pat.sub(rep, t or "")
    return t.strip()


# 자리표시자 단서어 — INEC 기사심의규정 유형별 (제17조 광고성 · 낚시성 제목 · 선정·자극)
CUES = {
    "광고성": re.compile(r"문의|상담|예약|할인|이벤트|특가|무료|증정|출시|판매|구매|가격|원장|대표원장|의원은|한의원|병원은|"
                       r"센터는|클리닉|㈜|주식회사|브랜드|제품|런칭|론칭|프로모션|사은품|공식몰|홈페이지|전화|010-|02-|1[568]\d\d-"),
    "낚시성": re.compile(r"충격|경악|알고 보니|알고보니|했더니|이유는|비결|비밀|놀라운|믿기지|\.\.\.$|…$|\?\!|화제|논란"),
    "자극성": re.compile(r"!{2,}|★|☆|♥|반드시|무조건|절대|100%|완치|기적|특효|만병|즉시|단번에|획기적|탁월"),
}


def _load(model_dir: str):
    """(tokenizer, model, 양성 인덱스). transformers/torch 가 없거나 폴더가 없으면 None."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label = {int(i): str(l) for i, l in (getattr(model.config, "id2label", {}) or {}).items()}
    pos = next((i for i, l in id2label.items() if l.lower() not in _NEG and not l.lower().startswith("non")), 1)
    return tok, model, pos


class StyleAxis:
    def __init__(self, model_dir: str = MODEL_DIR, threshold: float = 0.5, max_len: int = 512, max_chars: int = 0):
        self.threshold = threshold
        self.max_len = max_len
        self.max_chars = max_chars          # 매체 대응 모델을 200자로 학습했으면 추론도 200자로 (train 과 같은 값)
        self.backend = "heuristic"
        self.models: dict[str, tuple] = {}
        try:
            import torch                                                # noqa: F401
            for t, d in TYPE_DIRS.items():
                if os.path.isdir(d):
                    self.models[t] = _load(d)
            if self.models:
                self.backend = "kpf-bert×3" if len(self.models) == 3 else f"kpf-bert×{len(self.models)}"
            elif os.path.isdir(model_dir):
                self.models["비정상"] = _load(model_dir)
                self.backend = "kpf-bert"
        except Exception as e:                                          # noqa: BLE001
            print(f"[style_axis] 모델 로드 실패 → heuristic: {e}")
            self.models, self.backend = {}, "heuristic"

    def score(self, title: str = "", text: str = "") -> dict:
        body = scrub(f"{title or ''}\n{text or ''}")
        if self.max_chars:
            body = body[:self.max_chars]
        cues = {k: sorted(set(m.group(0) for m in pat.finditer(body)))[:8] for k, pat in CUES.items()}
        types = None
        if self.models:
            types = {t: round(self._bert(body, *m), 3) for t, m in self.models.items()}
            p = max(types.values())
        else:
            p = self._heuristic(body, cues)
        return {"p_abnormal": round(p, 3), "label": "비정상" if p >= self.threshold else "정상",
                "types": types, "backend": self.backend, "cues": cues, "n_chars": len(body)}

    def _bert(self, body: str, tok, model, pos: int) -> float:
        import torch
        enc = tok(body, truncation=True, max_length=self.max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        return float(torch.softmax(logits, -1)[pos])

    def _heuristic(self, body: str, cues: dict) -> float:
        n = sum(len(v) for v in cues.values())
        per_k = n / max(len(body), 1) * 1000          # 1,000자당 단서어 수
        return 1.0 / (1.0 + math.exp(-(1.2 * per_k - 3.0)))   # 2.5개/1,000자 ≈ 0.5


if __name__ == "__main__":
    import sys
    sa = StyleAxis()
    txt = " ".join(sys.argv[1:]) or "○○의원 김○○ 원장은 “본 제품을 복용하면 중성지방이 감소한다”며 지금 바로 문의하라고 밝혔다!! 할인 이벤트 진행 중."
    print(sa.score("", txt))
