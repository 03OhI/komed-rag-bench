#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pubmed_index.py — 검색 계층용 인덱스 구축

무엇을 만드는가
    pubmed_db.jsonl (+ pubmed_db_evidence.jsonl) 을 한 번 훑어서

      offsets   문서 i 의 (파일번호, 바이트오프셋)  ← 본문은 질의 때 seek 로 읽는다
      dl        문서 i 의 토큰 수
      df        용어 → 문서빈도          (전역 IDF 용)
      ui2docs   MeSH UI → 문서 id 배열   (후보 좁히기용)
      ui2terms  MeSH UI → 영문 표면형들  (한국어 주장을 영문 초록에 잇는 다리)
      postings  용어 → (문서 id 배열, tf 배열)   (전역 BM25 상위 100 — 게이트 순위와 RRF 융합용, 11/2 추가)

    를 pubmed_index.pkl 로 저장한다 (version 2).

전치색인(postings)을 11/2 에 넣은 이유
    원래는 MeSH UI 로 후보를 먼저 좁힌 뒤 그 안에서만 랭킹했으므로 df 와 dl 만 전역으로
    두면 충분했다. 그런데 E3(NFCorpus) 실측에서 게이트 단독은 BM25 보다 nDCG@10 이 0.034
    낮았고, 게이트 순위를 전역 BM25 순위와 RRF(k=60) 로 합치면 같은 수준(−0.003)에 Recall@100
    은 더 높았다(E3_NFCorpus_결과_1102.md). 그래서 검색 규칙을 게이트+RRF 로 정했고, 전역 상위
    100 을 질의마다 문서 10만 건을 다 읽지 않고 구하려면 포스팅이 필요하다. array('i')+array('H')
    로 두면 문서 10만 건 기준 약 100MB 이고, 옛 색인(version 1, 포스팅 없음)으로도 retrieve.py
    는 경고만 내고 게이트 단독으로 돌아간다.

ui2docs 를 query_ui 만으로 만들지 않는 이유
    수집 쿼리(query_ui)는 그 문서를 '받아온 이유'일 뿐이다. 실제 문서에는 MeSH
    표목이 10~15개 붙어 있고, 거기에 주장의 다른 축(예: 중성지방)이 들어 있다.
    두 축을 모두 가진 문서를 **교집합**으로 뽑을 수 있어야 판정 근거가 된다.

사용
    python build_pubmed_index.py            # 필요할 때만 다시 빌드
    python build_pubmed_index.py --force
    python build_pubmed_index.py --extra pubmed_db_pairs.jsonl --out pubmed_index_pairs.pkl   # 쌍 보강분을 넣은 별도 색인 (11/10)
        → 쓰려면 환경변수 KOMED_INDEX=pubmed_index_pairs.pkl  (retrieve.py 가 읽는다. 없으면 기존 색인)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

from bm25_util import doc_tokens                      # noqa: E402
from mesh_resolve import normalize, _load_lookup      # noqa: E402

SOURCES = ["pubmed_db.jsonl", "pubmed_db_evidence.jsonl"]
KB = "mesh_lookup.pkl"
OUT = "pubmed_index.pkl"
EXTRA: set = set()   # --extra 로 준 소스 (11/12)

# 체크태그 — 거의 모든 문서에 붙어 주제 변별력이 없다. 포스팅만 수백만 건 불린다.
CHECKTAGS = {
    "D006801",  # Humans
    "D000818",  # Animals
    "D008297",  # Male
    "D005260",  # Female
    "D000328",  # Adult
    "D000368",  # Aged
    "D000369",  # Aged, 80 and over
    "D008875",  # Middle Aged
    "D000293",  # Adolescent
    "D002648",  # Child
    "D002675",  # Child, Preschool
    "D007223",  # Infant
    "D007231",  # Infant, Newborn
    "D055815",  # Young Adult
    "D051379",  # Mice
    "D051381",  # Rats
}


def _pick_terms(ui: str, ui2name: dict, ui2aliases: dict, n: int = 6) -> list[str]:
    """표목명 + 정보량 있는 별칭 n개. 표목명이 항상 0번."""
    name = ui2name.get(ui, "")
    cand = {a for a in ui2aliases.get(ui, ())
            if 4 <= len(a) <= 40 and len(a.split()) <= 4}
    cand.discard(normalize(name))
    ranked = sorted(cand, key=lambda a: (-len(a.split()), len(a)))
    return [name] + ranked[:n]


def sig(paths):
    """빌드 시점의 소스 파일 상태. 나중에 인덱스가 낡았는지 판별한다."""
    out = []
    for p in paths:
        st = os.stat(p)
        out.append((p, int(st.st_mtime), st.st_size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="최신이어도 다시 빌드")
    ap.add_argument("--extra", nargs="*", default=[], help="추가 소스 jsonl (예: pubmed_db_pairs.jsonl) — 2026-11-10")
    ap.add_argument("--out", default=None, help="색인 파일 이름 (기본 pubmed_index.pkl). --extra 를 쓰면 기존 색인을 덮어쓰지 않게 따로 줄 것")
    a = ap.parse_args()
    global SOURCES, OUT, EXTRA
    EXTRA = set(a.extra)
    if a.extra and not a.out:
        sys.exit("[에러] --extra 를 쓰면 --out 으로 새 색인 이름을 주세요 (기존 pubmed_index.pkl 은 10/23 임계값·기존 수치의 기준이라 덮어쓰지 않는다)")
    if a.extra:
        SOURCES = SOURCES + [p for p in a.extra if p not in SOURCES]
    if a.out:
        OUT = a.out

    files = [p for p in SOURCES if os.path.isfile(p)]
    if not files:
        sys.exit(f"[에러] 소스 없음: {SOURCES}")
    if not os.path.isfile(KB):
        sys.exit(f"[에러] {KB} 없음")

    if os.path.isfile(OUT) and not a.force:
        try:
            old = pickle.load(open(OUT, "rb"))
            if old.get("sources") == sig(files):
                print(f"{OUT} 이 이미 최신입니다. 다시 만들려면 --force")
                return
        except Exception:
            pass

    print("=" * 74)
    print("PubMed 검색 인덱스 구축")
    print("=" * 74)
    for p in files:
        print(f"  소스 {p}  ({os.path.getsize(p)/1e6:.0f} MB)")

    print("\nMeSH 사전 로드 중...", flush=True)
    lookup = _load_lookup(KB)
    name2ui = {}          # 정규화된 표목 → ui  (문서의 mesh 표목을 UI 로 되돌린다)
    ui2aliases = collections.defaultdict(set)
    for alias, v in lookup.items():
        ui = v["ui"]
        name2ui.setdefault(alias, ui)
        ui2aliases[ui].add(alias)
    ui2name = {}
    for v in lookup.values():
        ui2name.setdefault(v["ui"], v["name"])
    print(f"  별칭 {len(lookup):,}개 · descriptor {len(ui2name):,}개")

    offs_file = array("B")
    offs_byte = array("q")
    dl = array("i")
    df = collections.Counter()
    post: dict[str, tuple[array, array]] = {}   # 용어 → (문서 id, tf). 전역 BM25 용 (11/2)
    ui2docs = collections.defaultdict(set)
    pmid2id = {}          # 같은 PMID 가 여러 UI 쿼리로 중복 수집돼 있다 (실측 12.4%)
    dup = 0
    skipped = 0

    n = 0
    t0 = time.time()
    for fi, path in enumerate(files):
        with open(path, "rb") as f:
            off = f.tell()
            for raw in f:
                cur, off = off, off + len(raw)
                try:
                    d = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue

                # 11/12: 추가 소스(--extra)에서는 초록이 비었거나 자리표시("Not available.")인 문헌을 넣지 않는다.
                # 짧은 문서라 BM25 길이 정규화로 상위에 뜬다. 기본 두 소스는 기존 색인 수치를 지키려고 그대로 둔다.
                if path in EXTRA and (d.get("abstract") or "").strip().lower().rstrip(".") in ("", "not available"):
                    skipped += 1
                    continue

                uis = set()
                qu = d.get("query_ui")
                if qu:
                    uis.update(qu.split("+"))   # 11/12: 쌍 수집분의 query_ui 는 "D1+D2" — 쪼개서 두 표목 모두에 올린다
                for m in (d.get("mesh") or []):
                    u = name2ui.get(normalize(m))
                    if u:
                        uis.add(u)
                # 중복 PMID 는 문서를 새로 만들지 않고, 그 UI 들만 기존 문서에 붙인다.
                # 이렇게 해야 df 가 부풀지 않고 결과에 같은 논문이 두 번 나오지 않는다.
                pmid = d.get("pmid")
                if pmid in pmid2id:
                    prev = pmid2id[pmid]
                    for u in uis - CHECKTAGS:
                        ui2docs[u].add(prev)
                    dup += 1
                    continue

                toks = doc_tokens(d)
                dl.append(len(toks))
                for t, f in collections.Counter(toks).items():
                    df[t] += 1
                    pl = post.get(t)
                    if pl is None:
                        pl = post[t] = (array("i"), array("H"))
                    pl[0].append(n)
                    pl[1].append(min(f, 65535))
                for u in uis - CHECKTAGS:
                    ui2docs[u].add(n)

                pmid2id[pmid] = n
                offs_file.append(fi)
                offs_byte.append(cur)
                n += 1
                if n % 20000 == 0:
                    print(f"  {n:>8,}건  ({n/(time.time()-t0):,.0f}건/초)", flush=True)
    print(f"  중복 PMID {dup:,}건은 UI 만 병합했습니다", flush=True)
    if skipped:
        print(f"  추가 소스의 초록 없는 문헌 {skipped:,}건은 넣지 않았습니다", flush=True)

    avgdl = sum(dl) / n if n else 0.0
    idx = {
        "version": 2,          # 2 = postings 포함 (11/2). retrieve.py 는 1 도 읽는다(융합 없이)
        "sources": sig(files),
        "files": files,
        "n_docs": n,
        "avgdl": avgdl,
        "offs_file": offs_file,
        "offs_byte": offs_byte,
        "dl": dl,
        "df": dict(df),
        "ui2docs": {u: array("i", sorted(v)) for u, v in ui2docs.items()},
        "ui2name": {u: ui2name.get(u, u) for u in ui2docs},
        # 질의 확장용 표면형. **0번은 반드시 표목명**(retrieve.py 가 그렇게 읽는다).
        # 짧은 순 정렬은 'n3','oil' 같은 파편만 남겨 오히려 검색을 망친다.
        # 어절 수가 많은 별칭을 먼저 두고, 3자 이하 조각은 버린다.
        "ui2terms": {u: _pick_terms(u, ui2name, ui2aliases) for u in ui2docs},
        "postings": post,
    }

    with open(OUT, "wb") as f:
        pickle.dump(idx, f, protocol=4)

    post = sum(len(v) for v in idx["ui2docs"].values())  # UI 포스팅 수(출력용)
    print("\n" + "=" * 74)
    print(f"문서       {n:,}건")
    print(f"평균 길이  {avgdl:.0f} 토큰")
    print(f"어휘       {len(df):,}개")
    print(f"UI         {len(idx['ui2docs']):,}개 · 포스팅 {post:,}건 "
          f"(문서당 {post/max(n,1):.1f}개 UI)")
    tpost = sum(len(v[0]) for v in idx["postings"].values())
    print(f"용어 포스팅 {tpost:,}건 (문서당 고유 토큰 {tpost/max(n,1):.0f}개) — 전역 RRF 융합용")
    print(f"소요       {time.time()-t0:.0f}초")
    print(f"→ {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
