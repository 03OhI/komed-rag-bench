# -*- coding: utf-8 -*-
"""
KoMed-RAG Track C : 한국어 MeSH 별칭 테이블 구축 + 충돌 감사

입력  : mesh_ko_full.json      (LLM 번역 결과, 원본 보존)
출력  : ko_alias.json          (정규화 한국어 키 -> 정책순 후보 UI 리스트)
        ko_collision.tsv       (충돌 전수 + 적용된 해소 근거)
        ko_review_sample.tsv   (층화 표본 300건, 사람 검수용)

원본 JSON은 수정하지 않는다. 충돌 해소는 파괴적 삭제가 아니라
'별칭 테이블의 순위 정책'으로 처리한다.
"""
import json, re, unicodedata, collections, random, os

# ---------- 경로 (인자 없이 실행) ----------
KO_JSON   = "mesh_ko_full.json"
OUT_DIR   = "."
SEED      = 20260806
SAMPLE_N  = 300

for _c in ("mesh_ko_full.json", "mesh_ko_full__1_.json",
           "/mnt/user-data/uploads/mesh_ko_full__1_.json"):
    if os.path.exists(_c):
        KO_JSON = _c
        break


# ---------- 정규화 ----------
# 한국어는 띄어쓰기가 유동적이므로(급성 골수성 백혈병 / 급성골수성백혈병)
# 영어 트랙과 달리 공백을 '제거'한다. 그 외는 트랙 A와 동일 정책.
_KEEP = re.compile(r"[^0-9a-z\uac00-\ud7a3]+")

def norm_ko(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return _KEEP.sub("", s)


# ---------- 트리 유틸 ----------
def is_ancestor(trees_a, trees_b) -> bool:
    """a가 b의 상위 개념인가 (MeSH 트리번호 접두 포함)"""
    return any(tb.startswith(ta + ".") for ta in trees_a for tb in trees_b)

def min_depth(trees) -> int:
    return min((t.count(".") for t in trees), default=99)


def main():
    ko = json.load(open(KO_JSON, encoding="utf-8"))
    if isinstance(ko, list):
        ko = {v["mesh_id"]: v for v in ko}
    print(f"[load] {KO_JSON}  descriptor={len(ko):,}")

    # ---------- 1. 별칭 수집 ----------
    # alias_key -> [(ui, src)]   src: 'main' | 'syn'
    raw = collections.defaultdict(list)
    for ui, v in ko.items():
        seen = set()
        for src, term in ([("main", v.get("ko_main"))] +
                          [("syn", t) for t in (v.get("ko_synonyms") or [])]):
            key = norm_ko(term)
            if not key or key in seen:
                continue
            seen.add(key)
            raw[key].append((ui, src))

    n_keys = len(raw)
    coll = {k: v for k, v in raw.items() if len(v) > 1}
    print(f"[alias] 별칭 키 {n_keys:,}  |  충돌 키 {len(coll):,} ({len(coll)/n_keys:.2%})")

    # ---------- 2. 충돌 분류 + 순위 정책 ----------
    # 정책 (보고서 매핑정책 준수: '가장 흔히 색인되는 단일 descriptor,
    #        하위형은 명시될 때만')
    #   1) ko_main 이 ko_synonym 보다 우선
    #      -- ko_main 은 '이 descriptor 의 표준 역어' 라는 강한 배정이고
    #         ko_synonym 은 느슨한 구어 배정이라 오배정 확률이 높다.
    #         (혈액암: Hematologic Neoplasms(main) > Leukemia(syn))
    #   2) 상위 개념 우선  (치매 -> Dementia, not Alzheimer)
    #   3) 트리 깊이 얕은 쪽 우선
    #   4) UI 오름차순 (재현성)
    #
    # 후보가 모두 ko_main 이고 계층 관계도 없으면 = 순수 동음이의어.
    # 이 경우 자동 해소가 원리적으로 불가능하므로 ambiguous 로 표시하여
    # 리졸버가 낮은 신뢰도로 전량 반환하게 하고, 사람 검수 큐로 보낸다.
    alias, ambiguous, rows = {}, {}, []
    stat = collections.Counter()

    for key, cands in raw.items():
        if len(cands) == 1:
            alias[key] = [cands[0][0]]
            continue

        uis = [u for u, _ in cands]
        src = dict(cands)
        trees = {u: ko[u]["tree_numbers"] for u in uis}

        # 계층 관계 존재 여부
        hier = any(is_ancestor(trees[a], trees[b])
                   for a in uis for b in uis if a != b)
        kind = "hierarchy" if hier else (
            "homograph" if len({src[u] for u in uis}) == 1 and
            all(src[u] == "main" for u in uis) else "sibling")
        stat[kind] += 1

        # 자신을 상위로 갖는 후보 수가 많을수록(= 더 상위일수록) 앞으로
        def anc_score(u):
            return -sum(1 for o in uis if o != u and is_ancestor(trees[u], trees[o]))

        ordered = sorted(
            uis,
            key=lambda u: (0 if src[u] == "main" else 1, anc_score(u),
                           min_depth(trees[u]), u)
        )
        alias[key] = ordered
        if kind == "homograph":
            ambiguous[key] = ordered

        rows.append([
            key, kind, str(len(uis)),
            " | ".join(f"{u}:{ko[u]['en_main']}({src[u]})" for u in ordered),
            "" if kind == "homograph" else ordered[0],
            "(자동해소불가)" if kind == "homograph" else ko[ordered[0]]["en_main"],
        ])

    print(f"[collision] 계층 {stat['hierarchy']:,} / 동음이의 {stat['homograph']:,} "
          f"/ 형제·기타 {stat['sibling']:,}")

    # ---------- 3. 저장 ----------
    with open(os.path.join(OUT_DIR, "ko_alias.json"), "w", encoding="utf-8") as f:
        json.dump({"alias": alias, "ambiguous": sorted(ambiguous)},
                  f, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "ko_collision.tsv"), "w", encoding="utf-8") as f:
        f.write("alias_key\tkind\tn_cand\tcandidates_ranked\tresolved_ui\tresolved_en\n")
        for r in sorted(rows, key=lambda x: (x[1], -int(x[2]))):
            f.write("\t".join(r) + "\n")

    # ---------- 4. 층화 검수 표본 ----------
    # 트리 대분류(A~Z 첫 글자) 비례 층화
    by_branch = collections.defaultdict(list)
    for ui, v in ko.items():
        br = v["tree_numbers"][0][0] if v["tree_numbers"] else "Z"
        by_branch[br].append(ui)

    rng = random.Random(SEED)
    total = len(ko)
    sample = []
    for br, uis in sorted(by_branch.items()):
        n = max(1, round(SAMPLE_N * len(uis) / total))
        sample += rng.sample(uis, min(n, len(uis)))
    rng.shuffle(sample)
    sample = sample[:SAMPLE_N]

    with open(os.path.join(OUT_DIR, "ko_review_sample.tsv"), "w", encoding="utf-8") as f:
        f.write("ui\tbranch\ten_main\tko_main\tko_synonyms\t"
                "verdict\tcorrected_ko_main\tnote\n")
        for ui in sample:
            v = ko[ui]
            br = v["tree_numbers"][0][0] if v["tree_numbers"] else "Z"
            f.write(f"{ui}\t{br}\t{v['en_main']}\t{v['ko_main']}\t"
                    f"{'; '.join(v.get('ko_synonyms') or [])}\t\t\t\n")

    print(f"[write] ko_alias.json ({len(alias):,} keys)")
    print(f"[write] ko_collision.tsv ({len(rows):,} rows)")
    print(f"[write] ko_review_sample.tsv ({len(sample)} rows)")


if __name__ == "__main__":
    main()
