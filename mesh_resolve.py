"""
mesh_resolve.py — MeSH resolver (exact → fuzzy 폴백)

I/O 계약 v1.0 준수:
    resolve(term, k=5) -> list[dict]   # score 내림차순, 길이 0~k, ui 유일(dedupe)
    각 원소: ui(str "D…"), name(str), trees(list[str], 없으면 []),
             score(float 원점수), method("exact"|"fuzzy")

동작:
    1) exact  : 정규화 term 이 사전에 있으면 그대로 반환. score=100.0, method="exact".
                도치형("Neoplasms, Breast")은 de-invert 후보도 exact 로 시도.
    2) fuzzy  : rapidfuzz token_sort_ratio (어순 무관) 폴백. 기본 임계값 85.
                복수형·철자변형·도치형은 여기서 대부분 잡힌다.
    약어(TNBC, RCC 등)는 fuzzy 로 못 잡는 게 정상 → 빈 결과로 떨어져
    다음 단계(embedding/eutils)가 담당한다. (gold set 실측상 abbrev 최고점 ~40)

주의:
    - 여기 normalize() 는 mesh_lookup 사전 빌드에 쓰인 정규화와 **반드시 동일**해야
      exact 매칭이 깨지지 않는다. 실제로는 공용 모듈로 빼서 양쪽이 import 하는 게 안전.
    - 악센트 보존 버그 수정 포함(NFKD→ascii). 이 수정도 사전 빌드/질의 양쪽 동일 적용 필수.
"""
from __future__ import annotations

import pickle
import re
import unicodedata
from typing import Callable, Optional

# ----------------------------------------------------------------------------
# 정규화 (mesh_lookup 빌드와 동일해야 함)
# ----------------------------------------------------------------------------
def normalize(s: str) -> str:
    if not s:
        return ""
    # 악센트 보존: 'Sézary' -> 'sezary', 'Waldenström' -> 'waldenstrom'
    # (기존 [^a-z0-9]->공백 규칙이 악센트 문자를 파괴하던 버그를 앞에서 차단)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _deinvert_candidates(raw: str) -> list[str]:
    """MeSH 도치형 대응. 'Neoplasms, Breast' -> ['breast neoplasms'] (정규화된 형태)."""
    if "," not in raw:
        return []
    parts = [normalize(p) for p in raw.split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return []
    return [" ".join(reversed(parts))]


# ----------------------------------------------------------------------------
# scorer: rapidfuzz 우선, 없으면 순수파이썬 폴백(느림, 개발/오프라인용)
# ----------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz, process  # type: ignore
    _HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    _HAVE_RAPIDFUZZ = False

    def _lcs_len(a: str, b: str) -> int:
        n, m = len(a), len(b)
        if not n or not m:
            return 0
        prev = [0] * (m + 1)
        for i in range(1, n + 1):
            cur = [0] * (m + 1)
            ai = a[i - 1]
            for j in range(1, m + 1):
                cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else (
                    prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
                )
            prev = cur
        return prev[m]

    def _token_sort_ratio(a: str, b: str) -> float:
        a = " ".join(sorted(a.split()))
        b = " ".join(sorted(b.split()))
        if not a and not b:
            return 100.0
        if not a or not b:
            return 0.0
        return 200.0 * _lcs_len(a, b) / (len(a) + len(b))


# ----------------------------------------------------------------------------
# lookup 로더
# ----------------------------------------------------------------------------
def _coerce_trees(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [t for t in re.split(r"[|;]", v) if t]
    try:
        return [str(t) for t in v]
    except TypeError:
        return []


def _load_lookup(path: str) -> dict:
    """
    반환: {정규화된_별칭: {"ui","name","trees"}}
    - .pkl/.pickle : mesh_lookup 이 저장한 dict 를 그대로 로드
    - .parquet     : 컬럼 [term, ui, name, trees] 가정 (다르면 이 함수만 수정)
    """
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".parquet"):
        import pandas as pd  # 지연 import
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        term_c = cols.get("term") or cols.get("key") or list(df.columns)[0]
        ui_c, name_c = cols["ui"], cols["name"]
        trees_c = cols.get("trees")
        lut: dict = {}
        for i in range(len(df)):
            term = df[term_c].iat[i]
            lut[term] = {
                "ui": df[ui_c].iat[i],
                "name": df[name_c].iat[i],
                "trees": _coerce_trees(df[trees_c].iat[i] if trees_c else None),
            }
        return lut
    raise ValueError(f"지원하지 않는 lookup 포맷: {path}")


# ----------------------------------------------------------------------------
# resolver 팩토리 (B 트랙도 make_resolver 만 교체해 계약 재사용)
# ----------------------------------------------------------------------------
def make_resolver(
    lookup: Optional[dict] = None,
    lookup_path: Optional[str] = None,
    threshold: float = 85.0,
    fuzzy_limit: int = 40,
    tree_prefixes: Optional[tuple] = None,
) -> Callable[[str, int], list]:
    """
    lookup        : {정규화_별칭: {"ui","name","trees"}} dict (직접 주입)
    lookup_path   : 위 dict 를 담은 .pkl/.parquet 경로 (lookup 미지정 시)
    threshold     : fuzzy 채택 최소 token_sort_ratio (기본 85)
    fuzzy_limit   : dedupe 전에 뽑을 후보 수 (k보다 넉넉히)
    tree_prefixes : 도메인 트리 prefix 화이트리스트. 예: ("C04",) = 신생물.
                    None 이면 필터 없음(기본).
                    ※ 주의: 정답이 해당 트리 밖이면 영구히 못 잡는다.
                      예) 골수형성이상증후군 D009190, MGUS D008998 은 C15 소속이라
                          ("C04",) 만 쓰면 배제된다 → ("C04","C15") 처럼 넓혀야 함.
    """
    if lookup is None:
        if not lookup_path:
            raise ValueError("lookup 또는 lookup_path 가 필요합니다")
        lookup = _load_lookup(lookup_path)

    tree_prefixes = tuple(tree_prefixes) if tree_prefixes else None

    def _in_domain(entry) -> bool:
        if not tree_prefixes:
            return True
        return any(str(t).startswith(tree_prefixes) for t in _coerce_trees(entry.get("trees")))

    keys = list(lookup.keys())  # 이미 정규화되어 저장됐다고 가정

    if _HAVE_RAPIDFUZZ:
        def _candidates(qnorm: str, limit: int):
            # (choice, score, index) 리스트, score_cutoff 미만은 제외
            return process.extract(
                qnorm, keys,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
                limit=limit,
            )
    else:  # pragma: no cover
        def _candidates(qnorm: str, limit: int):
            scored = [
                (kk, _token_sort_ratio(qnorm, kk), None)
                for kk in keys
            ]
            scored = [t for t in scored if t[1] >= threshold]
            scored.sort(key=lambda t: -t[1])
            return scored[:limit]

    def resolve(term: str, k: int = 5) -> list:
        if term is None:
            return []
        raw = term.strip()
        if not raw:
            return []
        qnorm = normalize(raw)
        if not qnorm:
            return []

        # 1) exact (도치형 de-invert 후보 포함)
        for cand in [qnorm, *_deinvert_candidates(raw)]:
            hit = lookup.get(cand)
            if hit and _in_domain(hit):
                return [{
                    "ui": hit["ui"],
                    "name": hit["name"],
                    "trees": _coerce_trees(hit.get("trees")),
                    "score": 100.0,
                    "method": "exact",
                }]

        # 2) fuzzy 폴백 (token_sort_ratio) — ui 기준 dedupe, 최고점 유지
        best: dict = {}
        for matched_key, score, *_ in _candidates(qnorm, fuzzy_limit):
            e = lookup[matched_key]
            if not _in_domain(e):
                continue
            ui = e["ui"]
            if ui not in best or score > best[ui]["score"]:
                best[ui] = {
                    "ui": ui,
                    "name": e["name"],
                    "trees": _coerce_trees(e.get("trees")),
                    "score": float(score),
                    "method": "fuzzy",
                }
        return sorted(best.values(), key=lambda d: -d["score"])[:k]

    return resolve


if __name__ == "__main__":
    # 최소 자체검증: 작은 in-memory 사전으로 도치형/복수형/오타/약어 동작 확인
    demo = {
        "breast neoplasms": {"ui": "D001943", "name": "Breast Neoplasms", "trees": ["C04.588.180"]},
        "breast cancer":    {"ui": "D001943", "name": "Breast Neoplasms", "trees": ["C04.588.180"]},
        "ovarian neoplasms":{"ui": "D010051", "name": "Ovarian Neoplasms", "trees": []},
        "carcinoma renal cell": {"ui": "D002292", "name": "Carcinoma, Renal Cell", "trees": []},
        "triple negative breast neoplasms": {"ui": "D064726", "name": "Triple Negative Breast Neoplasms", "trees": []},
    }
    r = make_resolver(lookup=demo, threshold=85.0)
    print("rapidfuzz:", _HAVE_RAPIDFUZZ)
    for q in ["Neoplasms, Breast", "breast neoplasm", "renal cell carcinoma",
              "brest cancer", "TNBC", ""]:
        out = r(q, k=3)
        top = out[0] if out else None
        print(f"{q!r:26s} -> {'[]' if not out else (top['ui'], round(top['score'],1), top['method'])}")
