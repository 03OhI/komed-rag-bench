# -*- coding: utf-8 -*-
"""
KoMed-RAG Track C : 한국어 직접 매핑 리졸버

입출력 계약 v1.0 준수
    resolve(term, k=5) -> [{ui, name, trees, score, method}, ...]
    score 는 정규화하지 않음 (0~100, 어휘 트랙과 동일 스케일)
    트랙 간 융합은 RRF 로만 수행

캐스케이드
    ko_exact -> ko_particle -> ko_suffix -> ko_modifier -> ko_fuzzy

사용
    from ko_resolver import KoResolver
    r = KoResolver()
    r.resolve("급성골수성백혈병이", k=5)
"""
import json, re, unicodedata, os

VERSION = "v4 (2026-08-08) — 이형태 규칙 + 자모 퍼지"

try:
    from rapidfuzz import process, fuzz
    _HAS_RF = True
except ImportError:                      # 로컬 검증용 폴백
    import difflib
    _HAS_RF = False

# ---------- 경로 ----------
# PyCharm 은 작업 디렉터리를 프로젝트 루트로 잡으므로
# 상대경로 대신 '이 파일이 있는 폴더' 기준으로 찾는다.
HERE = os.path.dirname(os.path.abspath(__file__))

def _find(*names):
    for d in (HERE, os.getcwd(), os.path.dirname(HERE), "/mnt/user-data/uploads"):
        for n in names:
            fp = os.path.join(d, n)
            if os.path.exists(fp):
                return fp
    return os.path.join(HERE, names[0])

ALIAS_JSON = _find("ko_alias.json")
STOP_TXT   = _find("ko_stoplist.txt")
OVERRIDE   = _find("ko_override.tsv")
KO_JSON    = _find("mesh_ko_full.json", "mesh_ko_full__1_.json")

# 자모 기준 잠정값. 음성표본 150건 실측 오탐률: 82->7.3% / 85->2.0% / 90->0%
# gold set 확보 후 tune_threshold.py 로 재선정할 것 (트랙 A 는 85).
FUZZY_THRESHOLD = 88

_KEEP = re.compile(r"[^0-9a-z\uac00-\ud7a3]+")

# 문장 말 조사 — 뒤에서 1~2음절만, 잔여가 2음절 이상일 때만 제거
_PARTICLES = ("에서는", "으로는", "에게서", "이라는", "라는", "에서", "으로",
              "에게", "부터", "까지", "이나", "든지", "은", "는", "이", "가",
              "을", "를", "의", "에", "와", "과", "도", "만", "로")

# 접미 이형 — 갑상선암 <-> 갑상선종양 같은 표기 흔들림
#
# ※ 접미 '추가' 규칙(("", "증") 등)은 제거했다.
#    매칭 안 된 토큰 뒤에 증/병/질환을 붙여 사전을 재조회하므로
#      '환경' 오염   + 병 -> 환경병(Environmental Illness)      실측 276회
#      '수면' 부족   + 병 -> 수면병(아프리카 수면병)             실측 220회
#      '다음' 주     + 증 -> 다음증(Polydipsia)                실측 135회
#      아파트 '단지' + 증 -> 단지증(Brachydactyly)              실측 724회
#    처럼 일반어가 질병으로 둔갑한다. 원문 grep 결과 해당 문자열은 0회였다.
#    치환·제거 규칙만 남긴다.
_SUFFIX_PAIRS = (("병", ""), ("증", ""), ("증후군", ""), ("질환", ""),
                 ("암", "종양"), ("종양", "암"),
                 ("암", "신생물"), ("신생물", "암"))

# 접미 변형 최소 음절. 2음절 토큰은 일반어와 충돌 위험이 커서 제외.
_SUFFIX_MIN_LEN = 3

# ---------- 자모 분해 ----------
# 한국어 음절 단위 퍼지는 작동하지 않는다. 단어가 짧아 1글자 차이가
# 유사도를 크게 떨어뜨린다 (뇌졸증/뇌졸중 = 67점). 자모로 풀면 88점.
_CHO  = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

def to_jamo(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out.append(_CHO[c // 588])
            out.append(_JUNG[(c % 588) // 28])
            j = _JONG[c % 28]
            if j != " ":
                out.append(j)
        else:
            out.append(ch)
    return "".join(out)


# 체계적 이형태 — 순화어·표기 흔들림. 퍼지보다 규칙이 정확하고 안전하다.
# (치환 대상, 치환 결과). 빈 문자열이면 삽입/삭제.
_VARIANT_RULES = (("샘", "선"), ("선", "샘"),        # 갑상샘/갑상선, 전립샘/전립선
                  ("씨병", "병"), ("병", "씨병"),     # 파킨슨씨병/파킨슨병
                  ("성", ""), ("", "성"),            # 아토피성/아토피
                  ("쯔", "츠"), ("츠", "쯔"),        # 알쯔하이머/알츠하이머
                  ("그", "크"), ("크", "그"))        # 외래어 표기

# 수식어 — 제거 대상
_MODIFIERS = ("말기", "초기", "중기", "진행성", "진행된", "전이성", "재발성",
              "재발한", "원발성", "속발성", "양측성", "편측성", "국소",
              "국한된", "수술후", "수술전", "중증", "경증", "조기")

# 보호 목록 — 자체 descriptor 를 갖는 수식어는 절대 제거하지 않는다
# (트랙 A 의 보호 목록 정책과 동일)
_PROTECTED = ("급성", "만성", "악성", "양성", "소아", "성인", "노인", "신생아",
              "임신", "가족성", "선천성", "후천성", "유전성", "특발성",
              "여성", "남성", "치료저항성", "불응성", "거세저항성")


def norm_ko(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return _KEEP.sub("", s)


class KoResolver:
    METHOD_SCORES = {"ko_exact": 100.0, "ko_override": 100.0,
                     "ko_particle": 96.0, "ko_suffix": 92.0,
                     "ko_variant": 91.0, "ko_modifier": 90.0}

    def __init__(self, alias_json=ALIAS_JSON, ko_json=KO_JSON,
                 threshold=FUZZY_THRESHOLD, tree_filter=None,
                 jamo_fuzzy=True):
        blob = json.load(open(alias_json, encoding="utf-8"))
        self.alias = blob["alias"]
        self.ambiguous = set(blob.get("ambiguous", []))

        ko = json.load(open(ko_json, encoding="utf-8"))
        if isinstance(ko, list):
            ko = {v["mesh_id"]: v for v in ko}
        self.ko = ko
        self.threshold = threshold
        self.tree_filter = tree_filter      # 예: ("C04", "C15")
        # 스톱리스트: 아예 매칭시키지 않는다
        self.stop = set()
        if os.path.exists(STOP_TXT):
            for line in open(STOP_TXT, encoding="utf-8"):
                w = norm_ko(line.split("#")[0].strip())
                if w:
                    self.stop.add(w)

        # 강제 매핑: 사전 결과를 지정 UI 로 치환
        self.override = {}
        if os.path.exists(OVERRIDE):
            for line in open(OVERRIDE, encoding="utf-8"):
                if line.startswith("#") or "\t" not in line:
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and norm_ko(p[0]) and p[1].strip():
                    self.override[norm_ko(p[0])] = p[1].strip()

        self.jamo_fuzzy = jamo_fuzzy
        self._keys = [k for k in self.alias if k not in self.stop]
        self._j2k = {to_jamo(k): k for k in self._keys}
        self._jkeys = list(self._j2k)
        # 추가 층 (11/19): 환경변수 KOMED_KO_EXTRA=<tsv> 일 때만. 정확 일치·변형 단계에서만 쓰고
        # 자모 퍼지 후보(_keys/_j2k)에는 넣지 않는다. v1 키가 있으면 v1 우선. 규칙: claude/사전보강_상병마스터_규칙_1119.md §3-e
        self.extra_n = 0
        extra = os.environ.get("KOMED_KO_EXTRA")
        if extra and os.path.exists(extra):
            import csv as _csv
            for row in _csv.DictReader(open(extra, encoding="utf-8"), delimiter="\t"):
                k = norm_ko(row.get("ko_norm") or row.get("ko") or "")
                u = (row.get("ui") or "").strip()
                if k and u and k not in self.alias and k not in self.stop and u in self.ko:
                    self.alias[k] = [u]
                    self.extra_n += 1
            print(f"[ko_resolver] 추가 층 {os.path.basename(extra)}: {self.extra_n}개")
        print(f"[ko_resolver] {VERSION}  stop={len(self.stop)} "
              f"override={len(self.override)} "
              f"fuzzy={'jamo' if jamo_fuzzy else 'syllable'}@{threshold}")

    # ---------- 내부 ----------
    def _pack(self, uis, method, score):
        out = []
        for u in uis:
            v = self.ko.get(u)
            if not v:
                continue
            if self.tree_filter and not any(
                    t.startswith(p) for t in v["tree_numbers"]
                    for p in self.tree_filter):
                continue
            out.append({"ui": u, "name": v["en_main"],
                        "trees": v["tree_numbers"],
                        "score": score, "method": method,
                        "ko_main": v["ko_main"],
                        "ambiguous": method == "ko_exact" and False})
        return out

    def _lookup(self, key, method):
        if key in self.stop:
            return []
        uis = self.alias.get(key)
        if not uis:
            return []
        score = self.METHOD_SCORES.get(method, 89.0)
        if key in self.ambiguous:
            score -= 15.0                   # 동음이의 = 자동해소 불가, 신뢰도 강등
        res = self._pack(uis, method, score)
        for r in res:
            r["ambiguous"] = key in self.ambiguous
        return res

    def _variants(self, key):
        """조사 -> 접미 -> 수식어 순으로 질의 문자열만 재작성"""
        # 1) 조사
        for p in _PARTICLES:
            if key.endswith(p) and len(key) - len(p) >= 2:
                yield key[: -len(p)], "ko_particle"
        # 2) 접미 이형 (치환·제거만, 3음절 이상에만 적용)
        if len(key) >= _SUFFIX_MIN_LEN:
            for a, b in _SUFFIX_PAIRS:
                if key.endswith(a) and len(key) - len(a) >= 2:
                    yield key[: -len(a)] + b, "ko_suffix"
        # 3) 체계적 이형태 (순화어·표기 흔들림)
        for a, b in _VARIANT_RULES:
            if a:
                i = key.find(a)
                while i >= 0:
                    yield key[:i] + b + key[i + len(a):], "ko_variant"
                    i = key.find(a, i + 1)
            else:
                for i in range(1, len(key)):
                    yield key[:i] + b + key[i:], "ko_variant"

        # 4) 수식어 스트리핑 (보호 목록 우선 차단)
        for m in _MODIFIERS:
            if m in _PROTECTED:
                continue
            if key.startswith(m) and len(key) - len(m) >= 2:
                yield key[len(m):], "ko_modifier"

    def _fuzzy(self, key, k):
        """자모 분해 후 매칭. jamo_fuzzy=False 면 음절 단위(구버전 동작)."""
        if len(key) < 2:
            return []
        q = to_jamo(key) if self.jamo_fuzzy else key
        pool = self._jkeys if self.jamo_fuzzy else self._keys
        if _HAS_RF:
            hits = process.extract(q, pool, scorer=fuzz.ratio,
                                   limit=k * 3, score_cutoff=self.threshold)
            cand = [(h[0], h[1]) for h in hits]
        else:
            m = difflib.get_close_matches(q, pool, n=k * 3,
                                          cutoff=self.threshold / 100)
            cand = [(c, difflib.SequenceMatcher(None, q, c).ratio() * 100)
                    for c in m]
        if self.jamo_fuzzy:
            cand = [(self._j2k[c], sc) for c, sc in cand if c in self._j2k]
        out, seen = [], set()
        for ckey, sc in cand:
            for r in self._pack(self.alias[ckey], "ko_fuzzy", round(sc, 1)):
                if r["ui"] in seen:
                    continue
                seen.add(r["ui"])
                r["ambiguous"] = ckey in self.ambiguous
                out.append(r)
        return out

    # ---------- 계약 ----------
    def resolve(self, term, k=5):
        key = norm_ko(term)
        if not key or key in self.stop:
            return []
        if key in self.override:
            r = self._pack([self.override[key]], "ko_override", 100.0)
            if r:
                return r[:k]

        res = self._lookup(key, "ko_exact")
        if not res:
            # 1홉
            seen = {key}
            hop1 = []
            for vkey, method in self._variants(key):
                if vkey in seen:
                    continue
                seen.add(vkey)
                hop1.append((vkey, method))
                res = self._lookup(vkey, method)
                if res:
                    break
            # 2홉 — 정규화 단계끼리 연쇄 (갑상샘암 -> 갑상선암 -> 갑상선종양)
            if not res:
                for vkey, m1 in hop1:
                    for wkey, m2 in self._variants(vkey):
                        if wkey in seen:
                            continue
                        seen.add(wkey)
                        res = self._lookup(wkey, m2 if m1 == m2 else
                                           f"{m1}>{m2}".replace("ko_", "", 1))
                        if res:
                            break
                    if res:
                        break
        if not res:
            res = self._fuzzy(key, k)

        # ui 기준 중복 제거 (최고 점수 유지)
        best = {}
        for r in res:
            if r["ui"] not in best or r["score"] > best[r["ui"]]["score"]:
                best[r["ui"]] = r
        return sorted(best.values(), key=lambda x: -x["score"])[:k]


if __name__ == "__main__":
    r = KoResolver()
    tests = ["유방암", "급성골수성백혈병", "위암이", "당뇨병", "혈액암",
             "말기 폐암", "자궁경부암은", "간세포암종", "치매", "소세포폐암",
             "물혹", "교모세포종", "췌장암", "전립선암에서", "대장암"]
    print(f"[fuzzy backend] {'rapidfuzz' if _HAS_RF else 'difflib (폴백)'}")
    for t in tests:
        hits = r.resolve(t, k=3)
        top = hits[0] if hits else None
        flag = " *AMB" if top and top.get("ambiguous") else ""
        print(f"{t:14s} -> " + (f"{top['ui']} {top['name']}  "
              f"[{top['method']} {top['score']}]{flag}" if top else "무응답"))
