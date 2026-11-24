#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_model.py — 판독기 5종의 모델 ID 를 한곳에 적어 두고, 별칭으로 실행한다 (2026-11-10).
각 실행기에는 **전체 ID**(provider:model)가 --llm 으로 넘어가므로 출력 파일 이름·지문(fingerprint)은 손으로 적을 때와 같다.

    python run_model.py ping                       # 5종 연결 시험 (모델당 1회 호출)
    python run_model.py ping luna sonnet           # 일부만
    python run_model.py list                       # 별칭 ↔ 모델 ID
    python run_model.py solar run_verdict.py --dataset scifact --claims … --corpus … --cond ours free closed --out 새폴더
    python run_model.py all   run_verdict.py …     # 5종을 README §2 순서대로 차례로 (과금 주의)

모델 ID 가 바뀌면 MODELS 만 고친다. 고친 뒤 실행은 새 출력 폴더(--out)를 쓴다.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {  # 별칭 → --llm 값. README §2 표의 실행 순서
    "solar":  "upstage:solar-pro4",
    "gemini": "gemini:gemini-3.1-pro-preview",     # 주 판독기 — "우리 시스템" 행
    "flash":  "gemini:gemini-3.8-flash",
    "luna":   "openai:gpt-6-luna",
    # "haiku": "anthropic:claude-haiku-5-5",   # 2026-11-10 예행(SciFact 50건) 뒤 본실행에서 제외 — README 2026-11-10 기록
    "sonnet": "anthropic:claude-sonnet-5-5",
}


def main():
    av = sys.argv[1:]
    if not av or av[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if av[0] == "list":
        for k, v in MODELS.items():
            print(f"{k:8s} {v}")
        return 0
    if av[0] == "ping":
        names = av[1:] or list(MODELS)
        bad = [n for n in names if n not in MODELS]
        if bad:
            sys.exit(f"모르는 별칭: {bad} — {list(MODELS)}")
        fail = 0
        for n in names:
            r = subprocess.run([sys.executable, os.path.join(HERE, "llm_client.py"), MODELS[n]],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
            ok = r.returncode == 0
            fail += not ok
            err = r.stderr.strip()
            i = err.rfind("RuntimeError")
            msg = r.stdout.strip() if ok else " ".join((err[i:] if i >= 0 else err).split())[:500] or "(출력 없음)"
            warn = " ".join(l for l in r.stderr.splitlines() if l.startswith("[경고]"))
            print(f"[{'OK ' if ok else '실패'}] {n:7s} {msg} {warn}".rstrip())
        return 1 if fail else 0
    name, rest = av[0], av[1:]
    if name != "all" and name not in MODELS:
        sys.exit(f"모르는 별칭: {name} — {list(MODELS)} · all · ping · list")
    if not rest:
        sys.exit("실행할 스크립트를 주세요 — 예) python run_model.py solar run_verdict.py --dataset …")
    if "--llm" in rest:
        sys.exit("--llm 은 적지 않습니다 (별칭이 대신 채웁니다)")
    script = rest[0] if os.path.isabs(rest[0]) else os.path.join(HERE, rest[0])
    for n in (list(MODELS) if name == "all" else [name]):
        cmd = [sys.executable, script, *rest[1:], "--llm", MODELS[n]]
        print(f"\n▶ {n} = {MODELS[n]}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"✗ {n} 종료 코드 {rc} — 여기서 멈춥니다")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
