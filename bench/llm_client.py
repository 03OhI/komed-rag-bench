#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_client.py — 벤치마크용 LLM 호출 (mesh/komed_llm.py 의 LLM 을 상속 · 공급자 확장 · 온도 0 · 사용량 기록)

    from llm_client import BenchLLM, PROVIDERS
    llm = BenchLLM("upstage")                      # solar-pro4 (OpenAI 호환)
    llm = BenchLLM("gemini", "gemini-3.1-pro-preview")
    llm = BenchLLM("clova", "HCX-007")             # CLOVA Studio OpenAI 호환 엔드포인트
    llm = BenchLLM("vllm", "skt/A.X-4.0-Light")    # 코랩 vLLM 서버 (http://localhost:8000/v1)
    out = llm.call(system, prompt, tool)           # → dict (도구 입력). 실패 시 예외

공급자 (키는 환경변수 또는 mesh/ 의 *_key.txt 한 줄)
    gemini      GEMINI_API_KEY        generativelanguage (komed_llm 그대로, temperature 0)
    anthropic   ANTHROPIC_API_KEY     messages API (temperature 0 추가)
    openai      OPENAI_API_KEY        chat/completions (temperature 0; 거부하면 빼고 재시도)
    upstage     UPSTAGE_API_KEY       https://api.upstage.ai/v1            기본 solar-pro4
    clova       CLOVASTUDIO_API_KEY   https://clovastudio.stream.ntruss.com/v1/openai   기본 HCX-007  (thinking 끔)
    deepseek    DEEPSEEK_API_KEY      https://api.deepseek.com             기본 deepseek-flash
    qwen        DASHSCOPE_API_KEY     https://dashscope-intl.aliyuncs.com/compatible-mode/v1   기본 qwen3.8-flash
    openrouter  OPENROUTER_API_KEY    https://openrouter.ai/api/v1         모델 id 는 "회사/모델"
    vllm        VLLM_API_KEY(임의)     http://localhost:8000/v1             코랩에서 띄운 오픈 모델
    mock        —                     규칙 기반 (smoke_test 용)

원칙
    · 도구 호출 강제로 항상 dict 를 받는다. tool_choice 를 거부하는 서버는 "JSON 만 출력" 프롬프트로 자동 폴백.
    · 같은 (모델·시스템·프롬프트·도구) 는 캐시에서 돌려준다 → 재실행 무과금. `--no-cache` 로 끄면 결정성 검사(3회 반복)가 된다.
    · usage(토큰 수)가 오면 bench_usage.jsonl 에 남긴다 → cost.py 가 비용을 집계한다.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)                      # mesh/ (komed_llm.py · verdict.py 가 있는 곳)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from komed_llm import LLM, _extract_json, _post, DEFAULT_MODEL, KEY_ENV, KEY_FILE  # noqa: E402

PROVIDERS = {
    # OpenAI 호환 서버들 — base_url 만 다르다
    "upstage":    {"base_url": "https://api.upstage.ai/v1", "key_env": "UPSTAGE_API_KEY", "model": "solar-pro4"},
    "clova":      {"base_url": "https://clovastudio.stream.ntruss.com/v1/openai", "key_env": "CLOVASTUDIO_API_KEY", "model": "HCX-007"},
    "deepseek":   {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-flash"},
    "qwen":       {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "key_env": "DASHSCOPE_API_KEY", "model": "qwen3.8-flash"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY", "model": "upstage/solar-pro4"},
    "vllm":       {"base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), "key_env": "VLLM_API_KEY", "model": None},
}
COMPAT = set(PROVIDERS)
# 공급자별 추가 본문 파라미터 (필요 시). clova: thinking 과 tools 동시 불가 → effort none
EXTRA_BODY = {"clova": {"reasoning_effort": "none"}}

USAGE_LOG = os.path.join(HERE, "bench_usage.jsonl")
# 2026-11-10: 모델이 거부한 설정은 프로세스 안에서 한 번만 배우고(경고도 한 번) 모든 작업자가 공유한다 — 키는 "provider:model"
_NO_TEMP: set = set()        # temperature 거부
_ANTH_AUTO: set = set()      # anthropic 도구 강제 거부 → tool_choice auto + 지시문
_REASON_NONE: set = set()    # openai 추론 켠 채 함수 호출 거부 → reasoning_effort none


def _learn(bag: set, label: str, msg: str):
    if label not in bag:
        bag.add(label)
        print(f"[경고] {label}: {msg}", file=sys.stderr)


class BenchLLM(LLM):
    def __init__(self, provider: str | None = None, model: str | None = None, key: str | None = None,
                 cache_path: str | None = "bench_llm_cache.jsonl", max_tokens: int = 2500,
                 mock_fn=None, verbose: bool = False, temperature: float | None = 0.0, timeout: int = 180):
        provider = provider or "gemini"
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = None
        self.extra_body = {}
        self.last_usage = None
        if provider in COMPAT:
            cfg = PROVIDERS[provider]
            self.base_url = cfg["base_url"]
            self.extra_body = dict(EXTRA_BODY.get(provider, {}))
            if not (model or cfg["model"]):
                raise RuntimeError(f"{provider}: 'provider:model' 형식으로 모델 id 를 주세요 (예: vllm:skt/A.X-4.0-Light)")
            k = key or _env_key(cfg["key_env"], provider) or ("EMPTY" if provider == "vllm" else "")
            if not k:
                raise RuntimeError(f"{provider} API 키가 없습니다 — 환경변수 {cfg['key_env']} 또는 mesh/{provider}_key.txt")
            # 부모 생성자를 openai 로 속이고 키·모델을 직접 채운다 (호출 경로는 아래 _openai 오버라이드)
            super().__init__("openai", model or cfg["model"], k,
                             cache_path=cache_path, max_tokens=max_tokens, mock_fn=mock_fn, verbose=verbose)
            self.provider_name = provider
        else:
            super().__init__(provider, model, key, cache_path=cache_path, max_tokens=max_tokens,
                             mock_fn=mock_fn, verbose=verbose)
            self.provider_name = provider

    @property
    def label(self) -> str:
        return f"{self.provider_name}:{self.model}"

    # ------------------------------------------------------------ overrides
    def _anthropic(self, system, prompt, tool):
        # 2026-11-10: 5.5 계열은 temperature 를 거부하거나(haiku) 도구 강제(tool_choice tool/any)를 거부한다(sonnet).
        #   → 400 을 보고 한 번씩 빼고 재시도. 도구 강제가 안 되면 auto + "반드시 이 도구를 호출하라" 한 줄을 시스템 끝에 붙인다.
        if self.label in _NO_TEMP:
            self.temperature = None
        auto = self.label in _ANTH_AUTO
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "system": system + (f"\n\nYou must answer by calling the `{tool['name']}` tool exactly once." if auto else ""),
                "tools": [tool], "tool_choice": {"type": "auto"} if auto else {"type": "tool", "name": tool["name"]},
                "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        try:
            j = _post("https://api.anthropic.com/v1/messages", body,
                      {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            low = txt.lower()
            if e.code == 400 and "temperature" in low and self.temperature is not None:
                _learn(_NO_TEMP, self.label, "temperature 거부 → 빼고 호출")
                self.temperature = None
                return self._anthropic(system, prompt, tool)
            if e.code == 400 and "tool_choice" in low and not auto:
                _learn(_ANTH_AUTO, self.label, "도구 강제 거부 → tool_choice auto + 지시문")
                return self._anthropic(system, prompt, tool)
            raise RuntimeError(f"HTTP {e.code}: {txt[:300]}")
        self._log_usage(j.get("usage"))
        for b in j.get("content", []):
            if b.get("type") == "tool_use":
                return b["input"]
        txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        return _extract_json(txt)

    def _gemini(self, system, prompt, tool):
        # komed_llm._gemini 와 동일 (temperature 0) + usageMetadata 기록
        from komed_llm import _strip_schema
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"function_declarations": [{"name": tool["name"], "description": tool.get("description", ""),
                                                      "parameters": _drop_empty_enum(_strip_schema(tool["input_schema"]))}]}],
                "tool_config": {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tool["name"]]}},
                "generation_config": {"temperature": 0 if self.temperature is None else self.temperature}}
        j = _post(url, body, {}, timeout=self.timeout)
        self._log_usage(j.get("usageMetadata"))
        for cand in j.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    return part["functionCall"].get("args", {})
        txt = "".join(p.get("text", "") for c in j.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        return _extract_json(txt)

    def _openai(self, system, prompt, tool):
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}"}
        if self.label in _NO_TEMP:
            self.temperature = None
        if self.label in _REASON_NONE:
            self.extra_body["reasoning_effort"] = "none"
        base = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **self.extra_body}
        if self.temperature is not None:
            base["temperature"] = self.temperature
        # 1) 도구 호출 강제
        body = dict(base, tools=[{"type": "function", "function": {"name": tool["name"],
                                                                    "description": tool.get("description", ""),
                                                                    "parameters": tool["input_schema"]}}],
                    tool_choice={"type": "function", "function": {"name": tool["name"]}})
        try:
            j = _post(url, body, headers, timeout=self.timeout)
            self._log_usage(j.get("usage"))
            msg = j["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if tcs:
                args = tcs[0]["function"]["arguments"]
                return json.loads(args) if isinstance(args, str) else args
            if msg.get("content"):
                return _extract_json(msg["content"])
            raise RuntimeError("도구 호출도 본문도 없음")
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            low = txt.lower()
            if e.code == 400 and "temperature" in low and self.temperature is not None:
                # 추론 모델 등 temperature 를 거부하면 빼고 재시도
                _learn(_NO_TEMP, self.label, "temperature 거부 → 빼고 호출")
                self.temperature = None
                return self._openai(system, prompt, tool)
            if e.code in (400, 404, 422) and any(k in low for k in ("tool", "function")):
                msg = " ".join(txt.split())[:300]
                if "reasoning" in low and "reasoning_effort" not in self.extra_body:
                    # 2026-11-10: 추론을 켠 채로는 chat/completions 함수 호출을 거부하는 모델(gpt-6-luna 등) → 추론 none 으로 재시도
                    _learn(_REASON_NONE, self.label, f"도구 호출 거부(HTTP {e.code}) → reasoning_effort none 으로 재시도: {msg}")
                    self.extra_body["reasoning_effort"] = "none"
                    return self._openai(system, prompt, tool)
                print(f"[경고] {self.label}: 도구 호출 거부(HTTP {e.code}) → JSON 프롬프트로 폴백: {msg}", file=sys.stderr)
                return self._openai_json_fallback(base, url, headers, tool)
            raise RuntimeError(f"HTTP {e.code}: {txt[:300]}")

    def _openai_json_fallback(self, base, url, headers, tool):
        """tools 를 지원하지 않는 서버: 스키마를 프롬프트에 넣고 JSON 만 받는다."""
        schema = json.dumps(tool["input_schema"], ensure_ascii=False)
        msgs = list(base["messages"])
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] +
                    f"\n\n반드시 아래 JSON Schema 를 만족하는 JSON 객체 **하나만** 출력하라. 설명·마크다운 금지.\n{schema}"}
        body = dict(base, messages=msgs)
        try:
            body["response_format"] = {"type": "json_object"}
            j = _post(url, body, headers, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            low = e.read().decode("utf-8", "replace").lower()
            if "temperature" in low and "temperature" in body:      # 2026-11-10: 폴백 경로에서도 temperature 거부를 처리
                _learn(_NO_TEMP, self.label, "temperature 거부 → 빼고 호출")
                self.temperature = None
                body.pop("temperature")
                try:
                    j = _post(url, body, headers, timeout=self.timeout)
                except urllib.error.HTTPError:
                    body.pop("response_format", None)
                    j = _post(url, body, headers, timeout=self.timeout)
            else:
                body.pop("response_format", None)
                j = _post(url, body, headers, timeout=self.timeout)
        self._log_usage(j.get("usage"))
        return _extract_json(j["choices"][0]["message"]["content"])

    # ------------------------------------------------------------ usage log
    def _log_usage(self, usage):
        self.last_usage = usage
        if not usage:
            return
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "m": self.label,
               "in": usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("promptTokenCount"),
               "out": usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidatesTokenCount")}
        try:
            with open(USAGE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


def _drop_empty_enum(s):
    """Gemini 는 enum 의 빈 문자열을 400 으로 거부한다(2026-11-10 확인: verdict.TOOL 의 stance enum[3] = "") → 빈 값만 뺀다.
    stance 는 required 가 아니라서 무관 문헌은 필드를 생략하면 되고, 판정 규칙이 읽는 값은 그대로다."""
    if isinstance(s, dict):
        out = {k: _drop_empty_enum(v) for k, v in s.items()}
        if isinstance(out.get("enum"), list):
            out["enum"] = [x for x in out["enum"] if x != ""]
        return out
    if isinstance(s, list):
        return [_drop_empty_enum(x) for x in s]
    return s


def _env_key(env: str, provider: str) -> str:
    if os.environ.get(env):
        return os.environ[env].strip()
    for d in (MESH, HERE, os.getcwd()):
        p = os.path.join(d, f"{provider}_key.txt")
        if os.path.isfile(p):
            return open(p, encoding="utf-8").readline().strip()
    return ""


def _default_mock():
    """키 없이 전체 흐름을 돌려 보는 규칙 기반 가짜 응답 (smoke_test 용 · 실제 판정에 쓰지 않는다)."""
    import re as _re
    from verdict import mock_reading
    _neg = _re.compile(r"\b(no (?:significant )?(?:effect|benefit|association|evidence|difference)|not (?:associated|effective|significant)|did not|does not)\b", _re.I)
    _pos = _re.compile(r"\b(reduc|improv|prevent|lower|decreas|increas|protect|effective|benefi|associated)", _re.I)
    _strong = _re.compile(r"\b(causes?|prevents?|reduces?|cures?|improves?)\b", _re.I)
    _cond = _re.compile(r"\b(may|might|could|can|suggest|possibly|likely)\b", _re.I)
    _corr = _re.compile(r"\b(associated|linked|correlat|related)\b", _re.I)

    def submit_verdict(system, prompt, tool):
        body = prompt.split("[문헌]", 1)[1] if "[문헌]" in prompt else ""
        if not body:
            return {"label": "NEI", "reason": "mock closed-book", "citations": []}
        lab = "REFUTE" if _neg.search(body) else ("SUPPORT" if _pos.search(body) else "NEI")
        sents = _re.split(r"(?<=[.!?])\s+", _re.sub(r"\s+", " ", body))
        quote = (sents[-1] if sents else "")[:200]
        return {"label": lab, "key_sentence": quote, "reason": "mock", "citations": [{"n": 1, "key_sentence": quote}]}

    def strength7(s):
        return 7 if _strong.search(s) and not _cond.search(s) else (5 if _cond.search(s) else (3 if _corr.search(s) else 2))

    def code_strength(system, prompt, tool):
        props = tool["input_schema"]["properties"]
        four = props.get("s", props.get("s1", {})).get("maximum") == 3
        conv = (lambda v: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3}[v]) if four else (lambda v: v)
        if "s" in props:
            return {"s": conv(strength7(prompt))}
        m = _re.search(r"\[문장 1[^\]]*\] (.*)\n\[문장 2[^\]]*\] (.*)", prompt)
        s1, s2 = (m.group(1), m.group(2)) if m else (prompt, prompt)
        return {"s1": conv(strength7(s1)), "s2": conv(strength7(s2))}

    def judge_exaggeration(system, prompt, tool):
        o = code_strength(system, prompt, {"input_schema": {"properties": {"s1": {"maximum": 7}, "s2": {"maximum": 7}}}})
        return {"label": "exaggerates" if o["s1"] > o["s2"] else ("downplays" if o["s1"] < o["s2"] else "same"), "reason": "mock"}

    return {"submit_reading": mock_reading, "submit_verdict": submit_verdict,
            "code_strength": code_strength, "judge_exaggeration": judge_exaggeration,
            "ping": lambda s, p, t: {"ok": True, "model_says": "mock"}}


def make_llm(spec: str, cache: bool = True, mock_fn=None) -> BenchLLM:
    """'provider' 또는 'provider:model' 문자열 → BenchLLM."""
    provider, _, model = spec.partition(":")
    if provider == "mock" and mock_fn is None:
        mock_fn = _default_mock()
    return BenchLLM(provider, model or None, cache_path=("bench_llm_cache.jsonl" if cache else None), mock_fn=mock_fn)


if __name__ == "__main__":
    # 연결 시험:  python llm_client.py upstage:solar-pro4
    spec = sys.argv[1] if len(sys.argv) > 1 else "mock"
    tool = {"name": "ping", "description": "연결 시험", "input_schema": {"type": "object",
            "properties": {"ok": {"type": "boolean"}, "model_says": {"type": "string"}}, "required": ["ok"]}}
    llm = make_llm(spec, cache=False, mock_fn=lambda s, p, t: {"ok": True, "model_says": "mock"})
    print(llm.label, "→", llm.call("You are a test endpoint.", "Reply ok=true and say hello in Korean.", tool))
