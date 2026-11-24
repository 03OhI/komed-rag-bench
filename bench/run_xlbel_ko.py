#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_xlbel_ko.py — E4 한국어 경로: XL-BEL KO 1,000 멘션(Liu et al., ACL 2021)에서 한국어 직접 매핑 리졸버를 평가 (2026-09-21)

    cd mesh/bench
    python run_xlbel_ko.py                        # ① 질의 파일 자동 다운로드 ② UMLS API 로 CUI→MeSH ③ ko/kob 채점 → report_el_xlbel_ko.md
    python run_xlbel_ko.py --systems ko kob --k 5
    python run_xlbel_ko.py --cui2mesh data/xlbel/cui2mesh.json --no-api      # 매핑이 이미 있으면 API 없이

전제
    · mesh/umls_key.txt 에 UMLS API 키 한 줄 (UTS 로그인 → My Profile → API Key). .gitignore 의 *_key.txt 에 걸린다. 키는 출력하지 않는다.
    · 인터넷(raw.githubusercontent.com · uts-ws.nlm.nih.gov). PC 파이썬으로 실행 — 코워크 VM 은 네트워크가 막혀 있다.
    · mesh/ko_resolver.py · ko_backoff.py · mesh_ko_full.json · ko_alias.json (있음)

프로토콜 (논문에 그대로)
    · XL-BEL KO: 위키피디아 앵커 텍스트 1,000건 → UMLS CUI(복수 가능). 문헌 지표는 P@1·P@5(gold CUI 중 하나라도 맞으면 정답).
    · 우리 리졸버는 MeSH 를 내므로 gold CUI 를 UMLS Metathesaurus 의 MSH 원자(atom)로 MeSH 기술자(D-번호)에 대응시킨다.
      대응이 하나도 없는 멘션(예: MeSH 에 없는 생물 분류군)은 **범위 밖**으로 세고 별도 보고. 주 지표는 대응 가능한 부분집합의 P@1·Acc@5·무응답률.
      보수적 수치로 '전체 1,000 기준 P@1'(범위 밖 = 오답)도 같이 낸다 — 문헌 22.1 / 40.6 은 CUI 공간(전체) 기준이므로 이 열이 직접 비교에 가깝다.
    · 무응답(빈 결과)은 오답과 구분해 보고한다(리졸버 설계: 오답 < 무응답).
    · gold 의 MeSH 트리(C 질환·D 화학·B 생물·A 해부 …)별 P@1 도 낸다 — 우리 사전·규칙은 질환·화학 중심이라 어디서 지는지 보인다.

출력  data/xlbel/ko_1k_test_query.txt · data/xlbel/cui2mesh.json · preds/el_xlbel_ko_<system>.jsonl · report_el_xlbel_ko.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.dirname(HERE)
for p in (MESH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

QUERY_URL = "https://raw.githubusercontent.com/cambridgeltl/sapbert/main/evaluation/xl_bel/xlbel_v0.0/ko_1k_test_query.txt"
UMLS_API = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"
REFERENCE = [  # (시스템, P@1 / P@5, 비고) — 문헌은 UMLS-399931(62k 개념) 후보 공간, 1,000건 기준. 출처 확인 9/22 (XLBEL_KO_문헌수치표_0922.md)
    ("mBERT · XLM-R zero-shot (Liu et al. 2021 표 2)", "0.0 / 0.0 · 0.0 / 0.3", ""),
    ("SapBERT 영어 전용 (Liu et al. 2021 표 2)", "2.0 / 2.4", ""),
    ("XLM-R + SAP all-lang (Liu et al. 2021 표 2; 공개 모델)", "16.7 / 21.4", ""),
    ("XLM-R + SAP + en-ko 번역쌍 wt / wt+MUSE (Liu et al. 2021 표 3)", "22.1 / 27.9 · 24.4 / 29.2", "2021 논문 최고 24.4"),
    ("mSapBERT · mCODER · BERGAMOT zero-shot (Sakhovskiy et al. 2024 표 4)", "18.1 / 22.1 · 2.2 / 2.4 · 18.5 / 22.7", "Acc@1/@5"),
    ("SapBERT_multi · mGENRE · Con2GEN · BERGAMOT (Wang et al. 2026 표 1 재측정)", "14.8 · 15.7 · 17.4 · 18.5 / —", "R@1"),
    ("BioELX retriever / + Qwen3 reranker (Wang et al. 2026 v1)", "37.8 / 40.6 (R@1)", "공개 최고"),
]
TREE_NAMES = {"A": "해부", "B": "생물", "C": "질환", "D": "화학·약물", "E": "기법·장비", "F": "정신·심리", "G": "현상·과정",
              "H": "학문·직업", "I": "인류·사회", "J": "기술·산업·농업", "K": "인문", "L": "정보과학", "M": "인구집단",
              "N": "보건의료", "V": "출판특성", "Z": "지리"}


# ───────────────────────────────────────────── 데이터
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_query(path):
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[다운로드] {QUERY_URL}")
    req = urllib.request.Request(QUERY_URL, headers={"User-Agent": "komed-rag-bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


def parse_query(path):
    """한 줄 = CUI(|CUI)*||멘션 → [{'cuis': [...], 'mention': str, 'line': n}]"""
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        if "||" not in line:
            print(f"  ⚠ {n}행: 구분자 '||' 없음 — 건너뜀: {line[:60]}")
            continue
        cuis, mention = line.split("||", 1)
        cuis = [c.strip() for c in cuis.split("|") if re.fullmatch(r"C\d{7}", c.strip())]
        if not cuis:
            print(f"  ⚠ {n}행: CUI 형식 아님 — 건너뜀: {line[:60]}")
            continue
        rows.append({"line": n, "cuis": cuis, "mention": mention.strip()})
    return rows


# ───────────────────────────────────────────── UMLS API: CUI → MeSH
def read_key(path):
    """키 파일 한 줄. BOM·따옴표·'apiKey=' 접두어는 벗겨 준다 (메모장이 BOM 을 붙이거나 복사할 때 따옴표가 딸려 오는 경우)."""
    if not path or not os.path.isfile(path):
        return None
    k = open(path, encoding="utf-8-sig").read().strip().splitlines()
    if not k:
        return None
    key = k[0].strip().strip('"').strip("'").strip()
    if key.lower().startswith("apikey="):
        key = key[7:].strip()
    return key or None


def umls_mesh_codes(cui, key, retries=4):
    """CUI 의 MSH 원자 → MeSH 기술자 UI 집합 (D 번호; C 는 보충개념, Q 는 한정어 — 따로 돌려준다). 404 = MSH 원자 없음."""
    codes, page = collections.OrderedDict(), 1
    while True:
        q = urllib.parse.urlencode({"sabs": "MSH", "pageSize": 100, "pageNumber": page, "apiKey": key})
        url = UMLS_API.format(cui=cui) + "?" + q
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "komed-rag-bench/1.0"}), timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {}
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1)); continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1)); continue
                raise
        for atom in data.get("result", []):
            code = str(atom.get("code") or atom.get("sourceDescriptor") or "").rstrip("/").split("/")[-1]
            if re.fullmatch(r"[DCQ]\d{6,9}", code):
                codes.setdefault(code, {"tty": set(), "names": []})
                codes[code]["tty"].add(atom.get("termType") or "")
                if atom.get("termType") == "MH" and atom.get("name"):
                    codes[code]["names"].append(atom["name"])
        if page >= int(data.get("pageCount") or 1):
            break
        page += 1
    return {c: {"tty": sorted(v["tty"]), "names": v["names"][:2]} for c, v in codes.items()}


def check_key(key):
    """키 확인: 알려진 CUI 하나(C0019348 Herpes Simplex → MeSH D006561)를 조회한다. 키 문자열은 절대 출력하지 않는다."""
    if not key:
        return False, "키 파일이 비어 있음"
    try:
        codes = umls_mesh_codes("C0019348", key, retries=1)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"HTTP {e.code} — 키가 틀렸거나 아직 활성화되지 않음 (UTS My Profile 의 API Key 를 그대로 한 줄, 공백·따옴표·'apiKey=' 없이)"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    if "D006561" in codes:
        return True, f"OK — C0019348 → {sorted(codes)} (Herpes Simplex D006561 포함)"
    return False, f"응답은 왔지만 기대한 D006561 이 없음: {sorted(codes)}"


def map_cuis(rows, cache_path, key, sleep=0.12):
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.isfile(cache_path) else {}
    todo = sorted({c for r in rows for c in r["cuis"] if c not in cache})
    if todo and not key:
        sys.exit(f"[에러] CUI {len(todo)}개의 MeSH 대응이 없고 UMLS API 키도 없습니다 — mesh/umls_key.txt 를 만들거나 --cui2mesh 를 주세요")
    if todo:
        ok, msg = check_key(key)
        print(f"[UMLS] 키 확인: {msg}")
        if not ok:
            sys.exit("[에러] UMLS API 키 확인 실패 — 위 메시지를 보고 mesh/umls_key.txt 를 고친 뒤 다시 실행")
        print(f"[UMLS] CUI {len(todo)}개 조회 (캐시 {len(cache)}개 있음) …")
    t0 = time.time()
    for i, cui in enumerate(todo, 1):
        try:
            cache[cui] = umls_mesh_codes(cui, key)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                sys.exit(f"[에러] HTTP {e.code} at {cui} — 키가 도중에 거부됨. 캐시 {len(cache)}개 저장. 잠시 뒤 다시 실행")
            cache[cui] = {"_error": f"HTTP {e.code}"}
        except Exception as e:
            cache[cui] = {"_error": f"{type(e).__name__}: {str(e)[:120]}"}
        if i % 50 == 0 or i == len(todo):
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            print(f"  {i}/{len(todo)} · {time.time() - t0:.0f}s", flush=True)
        time.sleep(sleep)
    if todo:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    errs = [c for c in cache if isinstance(cache[c], dict) and "_error" in cache[c]]
    if errs:
        print(f"  ⚠ 조회 오류 {len(errs)}건 (다시 실행하면 재시도): {errs[:5]}")
        for c in errs:
            del cache[c]
    return cache


def gold_descriptors(row, cui2mesh):
    """행의 gold MeSH 기술자(D 번호) 집합. 보충개념(C…)·한정어(Q…)는 우리 사전 범위 밖이라 gold 로 안 쓴다."""
    g = set()
    for c in row["cuis"]:
        for code in (cui2mesh.get(c) or {}):
            if code.startswith("D"):
                g.add(code)
    return g


# ───────────────────────────────────────────── 리졸버
def build_systems(names):
    systems = {}
    if "ko" in names:
        from ko_resolver import KoResolver
        systems["ko"] = KoResolver()
    if "kob" in names:
        from ko_backoff import KoBackoffResolver
        systems["kob"] = KoBackoffResolver()
    return systems


def trees_of(ui, mesh_ko):
    d = mesh_ko.get(ui) or {}
    return d.get("tree_numbers") or []


def top_category(uis, mesh_ko):
    cats = collections.Counter()
    for ui in uis:
        for t in trees_of(ui, mesh_ko):
            cats[t[0]] += 1
    return cats.most_common(1)[0][0] if cats else "?"


# ───────────────────────────────────────────── 평가
def evaluate(rows, cui2mesh, systems, k, mesh_ko, out_dir, limit=0):
    os.makedirs(out_dir, exist_ok=True)
    mappable = [r for r in rows if gold_descriptors(r, cui2mesh)]
    if limit:
        mappable = mappable[:limit]
    report = {}
    for name, sysm in systems.items():
        t0 = time.time()
        preds, stats = [], collections.Counter()
        per_cat = collections.defaultdict(collections.Counter)
        methods = collections.Counter()
        for r in mappable:
            gold = gold_descriptors(r, cui2mesh)
            try:
                res = sysm.resolve(r["mention"], k=k) or []
            except Exception as e:
                res = []; stats["error"] += 1; err = f"{type(e).__name__}: {e}"
            else:
                err = None
            uis = [x.get("ui") for x in res if x.get("ui")]
            hit1 = bool(uis) and uis[0] in gold
            hit5 = any(u in gold for u in uis[:k])
            cat = top_category(gold, mesh_ko)
            stats["n"] += 1; stats["hit1"] += hit1; stats["hit5"] += hit5
            stats["noans"] += (not uis); stats["wrong1"] += (bool(uis) and not hit1)
            per_cat[cat]["n"] += 1; per_cat[cat]["hit1"] += hit1; per_cat[cat]["noans"] += (not uis)
            if uis:
                methods[res[0].get("method", "?")] += 1
            preds.append({"line": r["line"], "mention": r["mention"], "cuis": r["cuis"], "gold": sorted(gold),
                          "gold_names": [ (mesh_ko.get(g) or {}).get("en_main") for g in sorted(gold)],
                          "cat": cat, "top": [{"ui": x.get("ui"), "name": x.get("name"), "score": x.get("score"), "method": x.get("method")} for x in res[:k]],
                          "hit1": hit1, "hit5": hit5, "noans": not uis, "error": err})
        with open(os.path.join(out_dir, f"el_xlbel_ko_{name}.jsonl"), "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        report[name] = {"stats": stats, "per_cat": per_cat, "methods": methods, "sec": time.time() - t0}
    return mappable, report


def pct(a, b):
    return f"{100.0 * a / b:.1f}" if b else "—"


def wilson(k_, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k_ / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * (c - h), 100 * (c + h))


def write_report(path, rows, cui2mesh, mappable, report, k, mesh_ko, query_path, api_used):
    n_all = len(rows)
    n_map = sum(1 for r in rows if gold_descriptors(r, cui2mesh))
    n_multi = sum(1 for r in rows if len(r["cuis"]) > 1)
    n_c_only = sum(1 for r in rows if not gold_descriptors(r, cui2mesh) and any(code.startswith(("C", "Q")) for c in r["cuis"] for code in (cui2mesh.get(c) or {})))
    n_nomsh = n_all - n_map - n_c_only
    cat_all = collections.Counter(top_category(gold_descriptors(r, cui2mesh), mesh_ko) for r in rows if gold_descriptors(r, cui2mesh))
    L = [f"# E4 한국어 경로 — XL-BEL KO ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"질의 `{os.path.relpath(query_path, HERE)}` {n_all:,}건 (sha256 {sha256(query_path)[:12]}) · 복수 CUI {n_multi}건 · "
         f"UMLS MSH 원자로 MeSH 기술자 대응 **{n_map:,}건 ({pct(n_map, n_all)}%)** · 보충개념/한정어만 {n_c_only}건 · MSH 원자 없음 {n_nomsh}건 → 범위 밖 {n_all - n_map:,}건"
         + (" · UMLS REST API(current) 로 대응" if api_used else " · 대응 파일 사용"), "",
         "gold 트리 분포(대응 가능분): " + " · ".join(f"{TREE_NAMES.get(c, c)}({c}) {v}" for c, v in cat_all.most_common()), "",
         f"## 주 결과 — MeSH 대응 가능 {len(mappable):,}건 (top-{k})", "",
         "| 시스템 | P@1 [95% CI] | Acc@5 | 무응답률 | 오답률(응답 중 top-1 오답 / 전체) | 전체 1,000 기준 P@1 (범위 밖 = 오답) | 초/1k |",
         "|---|---|---|---|---|---|---|"]
    for name, rep in report.items():
        s = rep["stats"]; n = s["n"]
        lo, hi = wilson(s["hit1"], n)
        L.append(f"| {name} | **{pct(s['hit1'], n)}** [{lo:.1f}, {hi:.1f}] | {pct(s['hit5'], n)} | {pct(s['noans'], n)} ({s['noans']}) | "
                 f"{pct(s['wrong1'], n - s['noans'])} / {pct(s['wrong1'], n)} | {pct(s['hit1'], n_all)} | {1000 * rep['sec'] / max(n, 1):.1f} |")
    for name, val, note in REFERENCE:
        L.append(f"| (문헌) {name} | {val} | | | | CUI 공간 전체 기준 | {note} |")
    L += ["", "주: 문헌 수치는 UMLS CUI 전체를 후보 공간으로 하는 P@1(또는 R@1)이고, 본 실행은 gold 를 MeSH 기술자로 옮겨 MeSH 사전(31,110 개념) 안에서 잰다. "
          "직접 비교는 '전체 1,000 기준' 열(범위 밖을 오답으로 침)이 보수적 근사이며, 대응 가능분 P@1 은 우리 사전 범위 안의 성능이다. "
          "무응답은 리졸버가 후보를 내지 않은 것(오답과 구분). CI 는 Wilson.", ""]
    for name, rep in report.items():
        L += [f"## {name} — gold 트리별 (대응 가능분)", "", "| 트리 | n | P@1 | 무응답률 |", "|---|---|---|---|"]
        for cat, c in sorted(rep["per_cat"].items(), key=lambda x: -x[1]["n"]):
            L.append(f"| {TREE_NAMES.get(cat, cat)} ({cat}) | {c['n']} | {pct(c['hit1'], c['n'])} | {pct(c['noans'], c['n'])} |")
        L += ["", "top-1 을 낸 방법: " + " · ".join(f"{m} {v}" for m, v in rep["methods"].most_common()), ""]
    L += ["## 범위 밖 예시 (MeSH 기술자 대응 없음, 앞 15건)", ""]
    shown = 0
    for r in rows:
        if gold_descriptors(r, cui2mesh):
            continue
        codes = sorted({code for c in r["cuis"] for code in (cui2mesh.get(c) or {})})
        L.append(f"- {r['mention']} — {'|'.join(r['cuis'])}" + (f" (MSH 원자: {', '.join(codes)})" if codes else ""))
        shown += 1
        if shown >= 15:
            break
    L += ["", f"예측: `preds/el_xlbel_ko_<system>.jsonl` (gold·top-{k}·방법) — 오답 유형 분석은 여기서.", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"→ {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=os.path.join(HERE, "data", "xlbel", "ko_1k_test_query.txt"))
    ap.add_argument("--cui2mesh", default=os.path.join(HERE, "data", "xlbel", "cui2mesh.json"))
    ap.add_argument("--umls-key-file", default=os.path.join(MESH, "umls_key.txt"))
    ap.add_argument("--no-api", action="store_true", help="UMLS API 를 부르지 않는다 (--cui2mesh 가 완전해야 함)")
    ap.add_argument("--systems", nargs="+", default=["ko", "kob"], help="ko = KoResolver · kob = KoBackoffResolver(상위 개념 후퇴)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 대응 가능분 앞 N건만")
    ap.add_argument("--mesh-ko", default=os.path.join(MESH, "mesh_ko_full.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "preds"))
    ap.add_argument("--report", default=os.path.join(HERE, "report_el_xlbel_ko.md"))
    ap.add_argument("--check-key", action="store_true", help="UMLS API 키만 확인하고 끝낸다 (CUI 하나 조회)")
    a = ap.parse_args()

    if a.check_key:
        key = read_key(a.umls_key_file)
        print(f"키 파일 {a.umls_key_file}: " + (f"{len(key)}자" if key else "없음/비어 있음"))
        ok, msg = check_key(key)
        print(("✔ " if ok else "✘ ") + msg)
        sys.exit(0 if ok else 1)

    fetch_query(a.query)
    rows = parse_query(a.query)
    print(f"[xlbel-ko] {a.query}: {len(rows):,}행 (기대 1,000) · sha256 {sha256(a.query)[:12]}")
    key = None if a.no_api else read_key(a.umls_key_file)
    if not a.no_api and not key:
        print(f"  ⚠ UMLS 키 파일 없음: {a.umls_key_file} — 캐시에 없는 CUI 는 조회하지 못한다")
    cui2mesh = map_cuis(rows, a.cui2mesh, key)
    mesh_ko = json.load(open(a.mesh_ko, encoding="utf-8"))
    systems = build_systems(a.systems)
    mappable, report = evaluate(rows, cui2mesh, systems, a.k, mesh_ko, a.out, a.limit)
    write_report(a.report, rows, cui2mesh, mappable, report, a.k, mesh_ko, a.query, api_used=bool(key))


if __name__ == "__main__":
    main()
