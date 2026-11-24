#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_fallback.py — Retrieval Evaluator 가 incorrect / ambiguous 일 때의 외부 검색 (설계서 3.4절 "Google News API")

    from news_fallback import search_news
    search_news("오메가-3 중성지방", n=5) → [{"title", "link", "source", "published"}, ...]

Google News 의 RSS 검색(키 불필요)을 쓴다. 실패(오프라인·차단)하면 빈 목록 — 파이프라인은 멈추지 않는다.
전시장에 인터넷이 없으면 이 칸만 비고 나머지는 그대로 나온다.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

RSS = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def news_query(claim: str, concepts: list[dict] | None = None) -> str:
    """리졸브된 개념의 한국어 표면형(span)이 있으면 그것으로, 없으면 주장 문장 그대로."""
    spans = [c.get("span", "") for c in (concepts or []) if c.get("span")]
    spans = [re.sub(r"(은|는|이|가|을|를|의|에|로|으로|와|과|도)$", "", s) for s in spans]
    return " ".join(s for s in spans if len(s) >= 2) or claim


def search_news(query: str, n: int = 5, timeout: int = 8) -> list[dict]:
    url = RSS.format(q=urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (KoMed-RAG demo)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            root = ET.fromstring(r.read())
    except Exception:                         # noqa: BLE001 — 오프라인·차단·파싱 실패 전부 빈 목록
        return []
    out = []
    for item in root.iter("item"):
        t = html.unescape(item.findtext("title") or "")
        src = item.find("source")
        out.append({"title": t, "link": item.findtext("link") or "",
                    "source": (src.text if src is not None else "") or "", "published": item.findtext("pubDate") or ""})
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    import json, sys
    q = " ".join(sys.argv[1:]) or "오메가-3 중성지방"
    print(json.dumps(search_news(q), ensure_ascii=False, indent=1))
