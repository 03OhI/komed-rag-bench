#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_gemini.py — Gemini Batch API 로 E1(run_verdict.py) 호출을 미리 채워 넣는다 (2026-11-13)

왜
    Gemini 3.1 Pro 의 일반 호출은 하루 250건(RPD)으로 묶여 있다. Batch API 는 한도가 따로(대기 토큰 양)이고 요금이 절반이다.
    이 스크립트는 판정 코드를 바꾸지 않는다. run_verdict.py 가 보낼 요청을 **그대로** 모아 일괄로 보내고,
    받은 응답을 bench_llm_cache.jsonl 에 넣는다. 그 뒤 run_verdict.py 를 평소 명령으로 돌리면 전부 캐시에서 나온다.

      요청 본문  llm_client.BenchLLM._gemini 가 만드는 본문 그대로(시스템 지시 · 도구 선언 · 도구 강제 · temperature 0)
      캐시 키    komed_llm.LLM.call 과 같은 sha1(label \n system \n prompt \n tool 이름)
      응답 해석  _gemini 와 같은 순서(functionCall.args → 없으면 본문에서 JSON)

순서 (mesh/bench 에서, PC 파이썬)
    1) 모으기   python batch_gemini.py collect --job cc_pilot --dataset checkcovid --claims … --corpus … --cond ours free --limit 50
                 → batch_jobs/cc_pilot/part_000.jsonl …  (이미 캐시에 있는 호출은 빼고 모은다)
    2) 보내기   python batch_gemini.py submit --job cc_pilot            # 아직 안 보낸 묶음 하나. --all 이면 전부
    3) 상태     python batch_gemini.py status --job cc_pilot
    4) 받기     python batch_gemini.py fetch --job cc_pilot             # 끝난 묶음의 응답을 캐시에 넣는다
    5) 채점용 실행  python run_verdict.py (1 과 같은 인자) --llm gemini:gemini-3.1-pro-preview --out <새 폴더>
                 → "n_calls 0" 에 가깝게 캐시에서만 나온다. 일괄에서 실패한 몇 건만 일반 호출로 간다.

3번 실험 (과장 탐지 · 인과 언어 — run_exagg.py, 2026-11-14 연결)
    1) python batch_gemini.py collect --job exagg_dev_r1 --dataset exagg --data data/exagg_train.csv --cond ours7 ours4 direct directreview
       → submit · status · fetch 는 위와 같다
    2) directreview · aligned4 처럼 두 번째 호출이 첫 응답에 달린 조건은 fetch 뒤 **새 job 이름으로 collect 를 다시**:
       python batch_gemini.py collect --job exagg_dev_r2 --dataset exagg --data data/exagg_train.csv --cond ours7 ours4 direct directreview
       "보낼 것 0" 이 나올 때까지 되풀이한다 (보통 r2 에서 끝난다)
    3) python run_exagg.py --task exagg --data data/exagg_train.csv --llm gemini:gemini-3.1-pro-preview --mode ours7 ours4 direct directreview --out preds_exagg_dev
       --dataset exagg|causal 은 run_exagg 의 --task, --cond 는 --mode, --data · --limit · --per-class · --exclude-items 는 같은 이름

주의
    · collect 의 인자(dataset · claims · corpus · csv · cond · limit · exclude-items · rules · k)는 5) 의 run_verdict 인자와 같아야 한다.
    · Tier 1 의 Gemini 3.1 Pro 대기 토큰 한도는 500만이다. collect 가 묶음마다 추정 토큰을 찍는다. 한 번에 한 묶음씩 보내는 것이 안전하다.
    · 목표 처리 시간은 24시간, 48시간이 지나면 만료된다. 만료 · 실패한 묶음은 submit --retry 로 다시 보낸다.
    · 논문에는 "Batch API 로 호출했다(모델 · 프롬프트 · temperature 는 같다)"고 적는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import llm_client                                   # noqa: E402
from komed_llm import _extract_json                 # noqa: E402

API = "https://generativelanguage.googleapis.com"
JOBS = os.path.join(HERE, "batch_jobs")
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
               "BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"}


# ------------------------------------------------------------------ collect
class _Captured(Exception):
    pass


class Pending(Exception):
    """일괄로 보낼 호출 — run_item 이 오류 레코드로 받지만 collect 는 그 레코드를 버린다."""


class Recorder(llm_client.BenchLLM):
    """호출하지 않고 요청만 모은다. 캐시에 있으면 평소처럼 돌려준다."""

    def __init__(self, provider, model, key=None):
        super().__init__(provider, model, key, cache_path="bench_llm_cache.jsonl")
        self.requests: dict[str, dict] = {}
        self.hits = 0

    def call(self, system, prompt, tool, use_cache=True):
        k = hashlib.sha1(f"{self.label}\n{system}\n{prompt}\n{tool['name']}".encode("utf-8")).hexdigest()
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        if k not in self.requests:
            self.requests[k] = {"key": k, "request": self._body(system, prompt, tool)}
        raise Pending(k)

    def _body(self, system, prompt, tool):
        box = {}

        def fake_post(url, body, headers, timeout=120):
            box["body"] = body
            raise _Captured()

        orig = llm_client._post
        llm_client._post = fake_post
        try:
            self._gemini(system, prompt, tool)      # 본문을 만드는 코드를 그대로 탄다
        except _Captured:
            pass
        finally:
            llm_client._post = orig
        return box["body"]


def cmd_collect(a):
    provider, _, model = a.llm.partition(":")
    assert provider == "gemini", "Batch API 는 gemini 만"
    if a.dataset in ("exagg", "causal"):
        return _collect_exagg(a, provider, model)
    import run_verdict as rv
    import verdict
    a.cond = a.cond or ["ours", "free"]
    if not set(a.cond) <= {"ours", "free", "closed", "norule"}:
        sys.exit(f"[중단] E1 조건은 ours/free/closed/norule — 받은 것 {a.cond}")
    if a.dataset in ("scifact", "checkcovid"):
        assert a.claims and a.corpus, "--claims --corpus 필요"
        items = rv.load_scifact(a.claims, a.corpus, dataset=a.dataset, limit=a.limit)
    else:
        assert a.csv, "--csv 필요"
        items = rv.load_healthver(a.csv, limit=a.limit)
    items = rv.exclude_items(items, a.exclude_items)
    rules = verdict.load_rules(a.rules)
    if a.k:
        rules["k_evidence"] = a.k
    rec = Recorder(provider, model or None, key=None if _has_key() else "DUMMY-collect-only")
    n = 0
    for cond in a.cond:
        for it in items:
            rv.run_item(it, cond, rec, rules)
            n += 1
    print(f"[{a.dataset}] {len(items)} items × {a.cond} = {n} 항목·조건 · 캐시에 이미 있음 {rec.hits} · 보낼 것 {len(rec.requests)}")
    _write_job(a, rec)


def _collect_exagg(a, provider, model):
    """3번 실험(과장 탐지 · 인과 언어) — run_exagg.py 가 보낼 요청을 그대로 모은다 (2026-11-14).

    directreview(초안 → 검토)와 aligned4(검증 실패 시 수리 1회)는 두 번째 호출이 첫 응답에 달려 있다.
    그래서 한 번에 다 모이지 않는다: collect → submit → fetch 뒤 **새 --job 이름으로 collect 를 다시** 돌리면
    다음 단계 호출만 모인다. '보낼 것 0' 이 나오면 끝이고, 그때 run_exagg.py 를 같은 인자로 돌린다."""
    import run_exagg as rx
    if not a.data:
        sys.exit("[중단] --data 필요 (예: data/exagg_train.csv)")
    modes = a.cond or rx.DEFAULT_MODES[a.dataset]
    if not set(modes) <= rx.TASK_MODES[a.dataset]:
        sys.exit(f"[중단] {a.dataset} 에 없는 조건: {sorted(set(modes) - rx.TASK_MODES[a.dataset])}")
    a.cond = modes
    items = rx.select_items(a.dataset, a.data, a.limit, a.exclude_items, a.per_class)
    fn = rx.run_exagg_item if a.dataset == "exagg" else rx.run_causal_item
    rec = Recorder(provider, model or None, key=None if _has_key() else "DUMMY-collect-only")
    waiting, other_err = Counter(), Counter()
    for mode in modes:
        for it in items:
            r = fn(it, mode, rec)
            if r.get("error"):
                (waiting if r["error"] in rec.requests else other_err)[mode] += 1
    print(f"[{a.dataset}] {len(items)} items × {modes} · 캐시에 이미 있음 {rec.hits} · 보낼 것 {len(rec.requests)}")
    print(f"  조건별 응답 대기 항목: {dict(waiting) or 0}")
    if other_err:
        print(f"  [주의] 캐시 응답으로 이미 끝났지만 오류인 항목(형식 오류 등, 일괄로 다시 보내지 않음): {dict(other_err)}")
    _write_job(a, rec)


def _write_job(a, rec):
    reqs = list(rec.requests.values())
    if not reqs:
        runner = "run_exagg.py" if a.dataset in ("exagg", "causal") else "run_verdict.py"
        print(f"보낼 것이 없습니다. {runner} 를 collect 와 같은 인자로 바로 돌리면 됩니다.")
        return
    d = os.path.join(JOBS, a.job)
    if os.path.isdir(d) and os.listdir(d) and not a.force:
        sys.exit(f"[중단] {d} 가 이미 있습니다. 다른 --job 이름을 쓰거나 --force")
    os.makedirs(d, exist_ok=True)
    parts = []
    for i in range(0, len(reqs), a.chunk):
        chunk = reqs[i:i + a.chunk]
        name = f"part_{len(parts):03d}.jsonl"
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        chars = sum(len(json.dumps(r["request"], ensure_ascii=False)) for r in chunk)
        parts.append({"file": name, "n": len(chunk), "est_tokens": chars // 4, "batch": None, "state": None, "ingested": False})
        print(f"  {name}: {len(chunk)}건 · 추정 입력 토큰 {chars // 4:,} (글자 수 ÷ 4, 도구 선언 포함)")
    meta = {"job": a.job, "llm": rec.label, "model": rec.model, "created": time.strftime("%Y-%m-%d %H:%M"),
            "args": {k: v for k, v in vars(a).items() if k not in ("fn",)}, "parts": parts}
    _save(a.job, meta)
    print(f"→ {d}\n다음:  python batch_gemini.py submit --job {a.job}")


# ------------------------------------------------------------------ http
def _has_key() -> bool:
    return bool(llm_client.resolve_key("gemini")) if hasattr(llm_client, "resolve_key") else bool(_key(silent=True))


def _key(silent=False) -> str:
    from komed_llm import resolve_key
    k = resolve_key("gemini")
    if not k and not silent:
        sys.exit("gemini API 키가 없습니다 — GEMINI_API_KEY 또는 mesh/gemini_key.txt")
    return k


def _http(method, url, body=None, headers=None, raw=False, timeout=300):
    h = {"x-goog-api-key": _key(), **(headers or {})}
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            return (payload, r.headers) if raw else (json.loads(payload.decode("utf-8") or "{}"), r.headers)
    except urllib.error.HTTPError as e:
        sys.exit(f"[HTTP {e.code}] {method} {url.split('?')[0]}\n{e.read().decode('utf-8', 'replace')[:800]}")


def _upload(path: str, display: str) -> str:
    size = os.path.getsize(path)
    _, hdr = _http("POST", f"{API}/upload/v1beta/files", {"file": {"display_name": display}},
                   {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": "application/jsonl"})
    up = hdr.get("x-goog-upload-url") or hdr.get("X-Goog-Upload-URL")
    if not up:
        sys.exit("업로드 주소(x-goog-upload-url)를 받지 못했습니다")
    j, _ = _http("POST", up, open(path, "rb").read(),
                 {"Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize",
                  "Content-Type": "application/jsonl"})
    return j["file"]["name"]                        # files/xxxx


def _state(j: dict) -> str:
    return (j.get("metadata") or {}).get("state") or j.get("state") or ("JOB_STATE_SUCCEEDED" if j.get("done") and not j.get("error") else "?")


def _result_file(j: dict):
    for box in (j.get("response") or {}, j.get("dest") or {}, (j.get("metadata") or {}).get("output") or {}):
        for k in ("responsesFile", "responses_file", "fileName", "file_name"):
            if box.get(k):
                return box[k]
    return None


def _meta(job):
    p = os.path.join(JOBS, job, "meta.json")
    if not os.path.isfile(p):
        sys.exit(f"{p} 없음 — 먼저 collect")
    return json.load(open(p, encoding="utf-8"))


def _save(job, meta):
    with open(os.path.join(JOBS, job, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ submit / status / fetch
def cmd_submit(a):
    meta = _meta(a.job)
    todo = [p for p in meta["parts"] if p["batch"] is None or (a.retry and p["state"] in DONE_STATES and "SUCCEEDED" not in p["state"])]
    if not todo:
        print("보낼 묶음이 없습니다 (status 로 확인).")
        return
    for p in (todo if a.all else todo[:1]):
        path = os.path.join(JOBS, a.job, p["file"])
        fname = _upload(path, f"{a.job}-{p['file']}")
        j, _ = _http("POST", f"{API}/v1beta/models/{meta['model']}:batchGenerateContent",
                     {"batch": {"display_name": f"{a.job}-{p['file']}", "input_config": {"file_name": fname}}})
        p.update(batch=j.get("name"), state=_state(j), input_file=fname, submitted=time.strftime("%Y-%m-%d %H:%M"), ingested=False)
        _save(a.job, meta)
        print(f"  {p['file']}: {p['n']}건 → {p['batch']}  [{p['state']}]")
    left = sum(1 for p in meta["parts"] if p["batch"] is None)
    print(f"남은 묶음 {left}개.  상태:  python batch_gemini.py status --job {a.job}")


def cmd_status(a):
    meta = _meta(a.job)
    for p in meta["parts"]:
        if p["batch"] and p["state"] not in DONE_STATES:
            j, _ = _http("GET", f"{API}/v1beta/{p['batch']}")
            p["state"] = _state(j)
            if a.verbose:
                print(json.dumps(j, ensure_ascii=False)[:1500])
        print(f"  {p['file']}: {p['n']}건 · {p['batch'] or '아직 안 보냄'} · {p['state'] or '-'} · {'캐시에 넣음' if p['ingested'] else '아직'}")
    _save(a.job, meta)


def parse_response(resp: dict) -> dict:
    """llm_client.BenchLLM._gemini 와 같은 순서로 도구 입력을 꺼낸다."""
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "functionCall" in part:
                return part["functionCall"].get("args", {})
    txt = "".join(p.get("text", "") for c in resp.get("candidates", []) for p in c.get("content", {}).get("parts", []))
    return _extract_json(txt)


def ingest(lines, label: str, cache_path: str, usage_path: str, known: set) -> dict:
    ok = dup = bad = 0
    errors = []
    now = time.strftime("%Y-%m-%d %H:%M")
    with open(cache_path, "a", encoding="utf-8") as fc, open(usage_path, "a", encoding="utf-8") as fu:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                k = rec.get("key") or (rec.get("metadata") or {}).get("key")
                if "response" not in rec:
                    raise ValueError("응답 없음: " + json.dumps(rec.get("error") or rec, ensure_ascii=False)[:200])
                out = parse_response(rec["response"])
                if not isinstance(out, dict) or not k:
                    raise ValueError("도구 입력이 dict 가 아님")
            except Exception as e:                                     # noqa: BLE001
                bad += 1
                if len(errors) < 5:
                    errors.append(str(e)[:200])
                continue
            if k in known:
                dup += 1
                continue
            known.add(k)
            fc.write(json.dumps({"k": k, "v": out, "m": label, "t": now, "via": "batch"}, ensure_ascii=False) + "\n")
            u = rec["response"].get("usageMetadata") or {}
            fu.write(json.dumps({"t": now + ":00", "m": label, "in": u.get("promptTokenCount"),
                                 "out": u.get("candidatesTokenCount"), "batch": True}, ensure_ascii=False) + "\n")
            ok += 1
    return {"ok": ok, "dup": dup, "bad": bad, "errors": errors}


def _known(cache_path):
    known = set()
    if os.path.isfile(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            try:
                known.add(json.loads(line)["k"])
            except Exception:                                          # noqa: BLE001
                pass
    return known


def cmd_fetch(a):
    meta = _meta(a.job)
    cache_path = os.path.join(MESH, "bench_llm_cache.jsonl")
    known = _known(cache_path)
    for p in meta["parts"]:
        if not p["batch"] or p["ingested"]:
            continue
        j, _ = _http("GET", f"{API}/v1beta/{p['batch']}")
        p["state"] = _state(j)
        if "SUCCEEDED" not in p["state"]:
            print(f"  {p['file']}: {p['state']} — 아직 받을 수 없음")
            continue
        rf = _result_file(j)
        if not rf:
            print(f"  {p['file']}: 결과 파일 이름을 못 찾음. status --verbose 출력을 보여 주세요.")
            continue
        raw, _ = _http("GET", f"{API}/download/v1beta/{rf}:download?alt=media", raw=True)
        out = os.path.join(JOBS, a.job, p["file"].replace("part_", "result_"))
        with open(out, "wb") as f:
            f.write(raw)
        r = ingest(raw.decode("utf-8").splitlines(), meta["llm"], cache_path, llm_client.USAGE_LOG, known)
        p.update(ingested=True, result=r)
        print(f"  {p['file']}: 캐시에 넣음 {r['ok']} · 이미 있음 {r['dup']} · 실패 {r['bad']} / {p['n']}")
        for e in r["errors"]:
            print("     실패 예:", e)
    _save(a.job, meta)
    if all(p["ingested"] for p in meta["parts"]):
        if meta["args"].get("dataset") in ("exagg", "causal"):
            print("모든 묶음을 캐시에 넣었습니다. 새 --job 이름으로 collect 를 같은 인자로 다시 돌려 다음 단계 호출을 모으세요. "
                  "'보낼 것 0' 이면 run_exagg.py 를 같은 인자로 돌립니다.")
        else:
            print("모든 묶음을 캐시에 넣었습니다. 이제 run_verdict.py 를 collect 와 같은 인자로 돌리세요 (실패 건만 일반 호출).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--job", required=True)
    c.add_argument("--dataset", required=True, choices=["scifact", "healthver", "checkcovid", "exagg", "causal"],
                   help="exagg · causal = 3번 실험(run_exagg.py 의 --task)")
    c.add_argument("--claims"); c.add_argument("--corpus"); c.add_argument("--csv")
    c.add_argument("--data", help="3번 실험 자료 (run_exagg.py 의 --data)")
    c.add_argument("--llm", default="gemini:gemini-3.1-pro-preview")
    c.add_argument("--cond", nargs="+", default=None,
                   help="E1: ours free closed (기본 ours free) · 3번 실험: run_exagg.py 의 --mode (기본 ours7 ours4 direct)")
    c.add_argument("--per-class", type=int, default=0, help="3번 실험 run_exagg.py 의 --per-class 와 같게")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--exclude-items", nargs="*", default=[])
    c.add_argument("--rules", default=os.path.join(MESH, "verdict_rules.json"))
    c.add_argument("--k", type=int, default=0)
    c.add_argument("--chunk", type=int, default=1000, help="묶음 하나의 요청 수 (기본 1000)")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_collect)
    s = sub.add_parser("submit"); s.add_argument("--job", required=True)
    s.add_argument("--all", action="store_true", help="안 보낸 묶음을 전부 (기본: 하나)")
    s.add_argument("--retry", action="store_true", help="실패 · 만료 · 취소된 묶음을 다시")
    s.set_defaults(fn=cmd_submit)
    t = sub.add_parser("status"); t.add_argument("--job", required=True); t.add_argument("--verbose", action="store_true")
    t.set_defaults(fn=cmd_status)
    f = sub.add_parser("fetch"); f.add_argument("--job", required=True); f.set_defaults(fn=cmd_fetch)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
