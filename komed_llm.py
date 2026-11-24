#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
komed_llm.py — End-to-end 에서 쓰는 LLM 호출 한 곳 (anthropic / gemini / mock)

    from komed_llm import LLM
    llm = LLM("gemini")                          # 키: --key → GEMINI_API_KEY → gemini_key.txt  (기본 공급자. anthropic 도 같은 규약)
    out = llm.call(system, prompt, TOOL)         # 도구 호출을 강제해서 항상 dict 로 받는다

원칙
    · 자유 서술이 아니라 **도구 스키마**로만 받는다 — 판정 범주는 LLM 이 정하지 않고 verdict.py 의 고정 규칙이 정한다.
    · 같은 (모델·프롬프트·도구) 는 캐시(komed_llm_cache.jsonl)에서 돌려준다 — 전시 데모에서 같은 주장을 반복해도 재과금 없음.
    · 실패는 3회 재시도 후 예외. 호출부(verdict.py)가 '근거 없음(LLM 실패)' 로 강등한다.
    · translate_ko.py / llm_judge.py 와 같은 키 규약. 키 파일은 .gitignore 에 있다.

모델 기본값
    gemini     gemini-3.1-pro-preview  ← 기본 (설계서 3.5절 Gemini 3 Pro 의 현행 API 이름. 무료 티어는 AI Studio 키)
    anthropic  claude-sonnet-5      (--provider anthropic. 번역과 같은 모델이어도 된다 — 이것은 평가가 아니라 시스템 본체)
    mock       규칙 기반 가짜 응답   (키 없이 파이프라인 전체를 돌려 보는 용도)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
KEY_FILE = {"anthropic": "anthropic_key.txt", "gemini": "gemini_key.txt", "openai": "openai_key.txt"}
DEFAULT_MODEL = {"anthropic": "claude-sonnet-5", "gemini": "gemini-3.1-pro-preview", "openai": "gpt-5", "mock": "mock"}
DEFAULT_PROVIDER = "gemini"   # 2026-09-12 팀 결정: 공개 벤치마크 동급 · 가격 동급($2/$12 vs $2/$10) · 번역(Claude)과 공급자 분리. 설계서 3.5절과도 일치


def resolve_key(provider: str, cli: str | None = None) -> str:
    if provider == "mock":
        return ""
    if cli:
        return cli.strip()
    if os.environ.get(KEY_ENV[provider]):
        return os.environ[KEY_ENV[provider]].strip()
    for d in (HERE, os.getcwd()):
        p = os.path.join(d, KEY_FILE[provider])
        if os.path.isfile(p):
            return open(p, encoding="utf-8").readline().strip()
    return ""


def _post(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, key: str | None = None,
                 cache_path: str | None = "komed_llm_cache.jsonl", max_tokens: int = 2500,
                 mock_fn=None, verbose: bool = False):
        provider = provider or DEFAULT_PROVIDER
        assert provider in ("anthropic", "gemini", "openai", "mock"), provider
        self.provider = provider
        self.model = model or DEFAULT_MODEL[provider]
        self.key = resolve_key(provider, key)
        self.max_tokens = max_tokens
        self.mock_fn = mock_fn          # mock 백엔드: (system, prompt, tool) → dict, 또는 {tool_name: fn}
        self.verbose = verbose
        self.cache_path = os.path.join(HERE, cache_path) if cache_path and not os.path.isabs(cache_path) else cache_path
        self.cache: dict[str, dict] = {}
        self.n_calls = 0
        if self.cache_path and os.path.isfile(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["v"]
                except Exception:
                    pass
        if provider != "mock" and not self.key:
            raise RuntimeError(f"{provider} API 키가 없습니다 — 환경변수 {KEY_ENV[provider]} 또는 {KEY_FILE[provider]}")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------ public
    def call(self, system: str, prompt: str, tool: dict, use_cache: bool = True) -> dict:
        """도구 입력(dict)을 돌려준다. 캐시 → 호출(3회 재시도)."""
        k = hashlib.sha1(f"{self.label}\n{system}\n{prompt}\n{tool['name']}".encode("utf-8")).hexdigest()
        if use_cache and k in self.cache:
            return self.cache[k]
        last = None
        for attempt in range(3):
            try:
                out = self._dispatch(system, prompt, tool)
                self.n_calls += 1
                if self.cache_path is not None:
                    self.cache[k] = out
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": k, "v": out, "m": self.label, "t": time.strftime("%Y-%m-%d %H:%M")},
                                           ensure_ascii=False) + "\n")
                return out
            except urllib.error.HTTPError as e:      # 4xx 는 재시도해도 같다
                body = e.read().decode("utf-8", "replace")[:300]
                last = RuntimeError(f"HTTP {e.code}: {body}")
                if 400 <= e.code < 500 and e.code not in (408, 429):
                    break
            except Exception as e:                   # noqa: BLE001
                last = e
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM 호출 실패 ({self.label}): {last}")

    # ---------------------------------------------------------------- backends
    def _dispatch(self, system, prompt, tool):
        if self.provider == "mock":
            fn = self.mock_fn.get(tool["name"]) if isinstance(self.mock_fn, dict) else self.mock_fn
            if fn is None:
                raise RuntimeError(f"mock 백엔드에 {tool['name']} 처리기가 없습니다")
            return fn(system, prompt, tool)
        if self.provider == "anthropic":
            return self._anthropic(system, prompt, tool)
        if self.provider == "gemini":
            return self._gemini(system, prompt, tool)
        return self._openai(system, prompt, tool)

    def _anthropic(self, system, prompt, tool):
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]},
                "messages": [{"role": "user", "content": prompt}]}
        j = _post("https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        for b in j.get("content", []):
            if b.get("type") == "tool_use":
                return b["input"]
        # 도구 호출이 없으면 텍스트에서 JSON 을 건져 본다
        txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        return _extract_json(txt)

    def _gemini(self, system, prompt, tool):
        # 함수 선언 + 호출 강제(ANY). 스키마는 OpenAPI 부분집합이라 input_schema 를 그대로 쓴다.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"function_declarations": [{"name": tool["name"], "description": tool.get("description", ""),
                                                      "parameters": _strip_schema(tool["input_schema"])}]}],
                "tool_config": {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tool["name"]]}},
                "generation_config": {"temperature": 0}}
        j = _post(url, body, {})
        for cand in j.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    return part["functionCall"].get("args", {})
        txt = "".join(p.get("text", "") for c in j.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        return _extract_json(txt)

    def _openai(self, system, prompt, tool):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "tools": [{"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                                            "parameters": tool["input_schema"]}}],
                "tool_choice": {"type": "function", "function": {"name": tool["name"]}}}
        j = _post("https://api.openai.com/v1/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
        tc = j["choices"][0]["message"]["tool_calls"][0]
        return json.loads(tc["function"]["arguments"])


def _strip_schema(s):
    """Gemini 가 모르는 키(additionalProperties 등)를 걷어낸다."""
    if isinstance(s, dict):
        return {k: _strip_schema(v) for k, v in s.items() if k not in ("additionalProperties", "default", "examples")}
    if isinstance(s, list):
        return [_strip_schema(x) for x in s]
    return s


def _extract_json(txt: str) -> dict:
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("응답에 JSON 이 없음: " + txt[:200])
    return json.loads(txt[a:b + 1])
