# -*- coding: utf-8 -*-
"""
ko_backoff.py — 한국어 직접 매핑의 '상위 개념 후퇴' 프로토타입 (KoResolver 위에 얹는 확장)

ko_resolver.py 는 손대지 않는다. KoResolver 를 상속해 resolve() 의 무응답 구간에
두 단계를 끼워 넣는다.

    ko_exact -> ko_particle/ko_suffix/ko_variant/ko_modifier (기존)
             -> ko_compose : [수식어][일반 머리어] 를 사전에 있는 합성어로 재조립
                              비뇨기계암 = 비뇨기(계) + 암 -> '비뇨기종양' (D014571)
             -> ko_head    : 사전에 없는 합성어를 '사전에 있는 가장 긴 꼬리(머리어)'로 후퇴
                              편평세포후두암 -> 후두암 (D007822), 수식어 '편평세포' 를 qualifier 로 보존
                              소아암 -> 암 (D009369),        수식어 '소아' 를 qualifier 로 보존
             -> ko_fuzzy (기존)

반환 항목에는 기존 계약(ui, name, trees, score, method)에 더해
    modifier : 후퇴하면서 떼어낸 앞부분 (검색 계층이 BM25 질의어·체크태그로 쓸 수 있다)
를 붙인다. 계약 v1.0 의 필드는 그대로이므로 기존 호출자는 영향이 없다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ko_resolver import KoResolver, norm_ko, _PROTECTED, _SUFFIX_PAIRS  # noqa: E402

# 추가 접미 이형 — 수질암/수질암종, 췌장장애/췌장질환, 1형당뇨병/제1형당뇨병
EXTRA_SUFFIX_PAIRS = (("암", "암종"), ("암종", "암"),
                      ("증", "증후군"), ("증후군", "증"),
                      ("병", "질환"), ("질환", "병"))
# '장애' <-> '질환' 은 어간이 해부학 개념일 때만 (췌장장애 -> 췌장질환 O, 중증장애 -> 중증질환 X)
ANATOMY_SUFFIX_PAIRS = (("장애", "질환"), ("질환", "장애"))
# 후퇴를 허용하는 수식어의 종류 — 해부(A)·질환(C)·인구집단(M01) 개념이거나 보호 목록(연령·성별 등).
# 지리(Z)·경제(I)·기관(N) 등은 제외: '중동'+호흡기증후군, '경제'+성장 같은 오탐을 막는다.
MODIFIER_TREES = ("A", "C", "M01")
MODIFIER_MIN_LEN = 2
# n-gram 조각이 만드는 가짜 합성어 차단: '세포'+암, '사람'+유두종 (사람유두종바이러스의 조각)
MODIFIER_DENY = {"사람", "세포", "인간"}

# 일반 머리어와 그 동의 머리어 — 합성(ko_compose)에 쓴다
HEAD_SYNONYMS = {
    "암":   ("암", "종양", "신생물", "악성종양", "암종"),
    "종양": ("종양", "암", "신생물"),
    "염":   ("염", "염증"),
    "증":   ("증",),
    "질환": ("질환", "병", "장애"),
    "병":   ("병", "질환"),
}
# 부위 수식어 꼬리 — '비뇨기계' -> '비뇨기', '소화기계통' -> '소화기'
SITE_TAILS = ("계통", "계", "부", "성")

HEAD_MIN_LEN = 2          # 후퇴한 머리어는 2음절 이상 (예외: 아래 GENERIC_HEADS)
GENERIC_HEADS = {"암": "D009369", "종양": "D009369"}   # 1~2음절 일반 머리어 → 우산 개념 고정
SCORE_COMPOSE = 90.0
SCORE_HEAD = 89.0         # 검색 계층 KO_MIN_SCORE(88) 를 넘기되 정확 매칭보다 낮게


class KoBackoffResolver(KoResolver):
    VERSION_EXT = "backoff-proto (2026-09-12)"

    # ---- 보조 ----
    def _most_general(self, uis):
        """복수 UI 면 트리번호가 가장 짧은(상위) 개념을 고른다."""
        def depth(u):
            v = self.ko.get(u)
            if not v or not v["tree_numbers"]:
                return 99
            return min(len(t.split(".")) for t in v["tree_numbers"])
        return sorted(uis, key=depth)[0]

    def _in_trees(self, key, prefixes):
        uis = self.alias.get(key) or []
        for u in uis:
            v = self.ko.get(u)
            if v and any(t.startswith(prefixes) for t in v["tree_numbers"]):
                return True
        return False

    def _modifier_ok(self, mod):
        if len(mod) < MODIFIER_MIN_LEN or mod in MODIFIER_DENY:
            return False
        if mod in _PROTECTED:
            return True
        if self._in_trees(mod, MODIFIER_TREES):
            return True
        # 조직형 수식어: '편평세포'는 단독 표제어가 아니지만 '편평세포암종'(C04)이 사전에 있다
        return any(self._in_trees(mod + s, ("C04",)) for s in ("암종", "암", "종양", "종"))

    def _extra_suffix(self, key):
        if len(key) < 3:
            return
        for a, b in EXTRA_SUFFIX_PAIRS:
            if not key.endswith(a):
                continue
            stem = key[: -len(a)]
            need = 4 if "증후군" in (a, b) else 2      # 폐쇄증->폐쇄증후군 같은 오탐 방지
            if len(stem) >= need:
                yield stem + b, "ko_suffix2"
        for a, b in ANATOMY_SUFFIX_PAIRS:
            if key.endswith(a) and len(key) - len(a) >= 2 and self._in_trees(key[: -len(a)], ("A",)):
                yield key[: -len(a)] + b, "ko_suffix2"
        if key and key[0].isdigit():                      # 1형당뇨병 -> 제1형당뇨병
            yield "제" + key, "ko_suffix2"

    def _compose(self, key):
        """[수식어][일반 머리어] -> 사전에 있는 합성어로 재조립"""
        for head, syns in HEAD_SYNONYMS.items():
            if not key.endswith(head) or len(key) - len(head) < 2:
                continue
            mod = key[: -len(head)]
            # 어간이 이미 질환 개념이면 머리어는 군더더기다: 유잉육종암 -> 유잉육종, 육종암 -> 육종
            if self._in_trees(mod, ("C",)) and mod not in self.stop:
                c_uis = [u for u in self.alias[mod]
                         if any(tn.startswith("C") for tn in self.ko[u]["tree_numbers"])]
                return mod, c_uis, ""
            mods = [mod]
            for t in SITE_TAILS:
                if mod.endswith(t) and len(mod) - len(t) >= 2:
                    mods.append(mod[: -len(t)])
            for m in mods:
                for h in syns:
                    cand = m + h
                    if cand == key or cand in self.stop:
                        continue
                    uis = self.alias.get(cand)
                    if uis:
                        return cand, uis, m
        return None

    def _head_backoff(self, key):
        """사전에 있는 가장 긴 꼬리(머리어)로 후퇴. 떼어낸 앞부분은 modifier 로 보존."""
        n = len(key)
        for i in range(1, n):
            head, mod = key[i:], key[:i]
            if head in self.stop:
                continue
            # 떼어내는 수식어는 '아는 것'이어야 한다 (보호 목록 또는 해부·질환·인구집단 개념)
            if not self._modifier_ok(mod):
                continue
            if head in GENERIC_HEADS:
                return head, [GENERIC_HEADS[head]], mod
            if len(head) < HEAD_MIN_LEN:
                continue
            uis = self.alias.get(head)
            if uis:
                return head, uis, mod
            # 머리어에 접미 이형 적용 (편평세포후두암 -> 후두암 은 직접, 수질암 -> 수질암종 은 접미)
            for a, b in tuple(_SUFFIX_PAIRS) + EXTRA_SUFFIX_PAIRS:
                if head.endswith(a) and len(head) - len(a) >= 2:
                    alt = head[: -len(a)] + b
                    uis = self.alias.get(alt)
                    if uis and alt not in self.stop:
                        return alt, uis, mod
        return None

    # ---- 계약 ----
    def resolve(self, term, k=5):
        res = super().resolve(term, k=k)
        # 기존 캐스케이드가 퍼지로만 답했거나 무응답이면 후퇴 단계를 시도한다
        if res and not all(r["method"] == "ko_fuzzy" for r in res):
            return res
        key = norm_ko(term)
        if not key or key in self.stop:
            return res
        # 1) 추가 접미 이형
        for vkey, method in self._extra_suffix(key):
            uis = self.alias.get(vkey)
            if uis and vkey not in self.stop:
                out = self._pack(uis, method, 92.0)
                if out:
                    return out[:k]
        # 2) 합성
        c = self._compose(key)
        if c:
            cand, uis, mod = c
            out = self._pack([self._most_general(uis)] if len(uis) > 1 else uis,
                             "ko_compose", SCORE_COMPOSE)
            for r in out:
                r["modifier"] = mod
                r["composed_from"] = cand
            if out:
                return out[:k]
        # 3) 머리어 후퇴
        h = self._head_backoff(key)
        if h:
            head, uis, mod = h
            out = self._pack([self._most_general(uis)], "ko_head", SCORE_HEAD)
            for r in out:
                r["modifier"] = mod
                r["head"] = head
            if out:
                return out[:k]
        return res


if __name__ == "__main__":
    r = KoBackoffResolver()
    for t in ["소아암", "비뇨기계암", "편평세포폐암", "편평세포후두암", "수질암",
              "1형당뇨병", "췌장장애", "말기콩팥병", "무호흡증", "지방간질환",
              "노인우울증", "소아당뇨", "중증외상", "폐암", "아파트단지", "국내"]:
        hits = r.resolve(t, k=3)
        top = hits[0] if hits else None
        print(f"{t:10s} -> " + (f"{top['ui']} {top['name']} [{top['method']} {top['score']}]"
              + (f" modifier={top.get('modifier')}" if top.get('modifier') else "") if top else "무응답"))
