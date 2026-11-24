"""Read E4 predictions without re-running/tuning the resolver or overwriting old results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from metrics import wilson


def summarize(rows):
    ok = [r for r in rows if not r.get("error")]
    def answered(r):
        return bool(r.get("answered", not r.get("noans", not bool(r.get("pred") or r.get("top")))))
    ans = [r for r in ok if answered(r)]
    correct = sum(bool(r.get("hit1")) for r in ok)
    wrong = sum(not r.get("hit1") for r in ans)
    return {"n": len(rows), "valid": len(ok), "errors": len(rows)-len(ok), "answered": len(ans),
            "correct": correct, "wrong": wrong, "precision": (len(ans)-wrong)/len(ans) if ans else None}


def ratio(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["# E4 응답률–정밀도 보조 분석", "",
             "기존 결과를 읽는 사후 기술 분석이다. 새로운 사전 등록 결과나 새 테스트 성능으로 표현하지 않는다. "
             "전체 행의 분모는 해당 예측 파일에 실제 들어 있는 항목이다. MeSH 대응 610건 파일을 전체 999건 평가로 바꾸지 않는다. "
             "C+D는 gold 기반 분야별 진단이며 운영 중 정답을 알고 항목을 선택하는 방법이 아니다.", "",
             "| 파일 | 범위 | 유효/전체 · 오류 | P@1 % [CI] | 응답률 % | 응답 중 정답 % [CI] | 오답 응답/전체 |",
             "|---|---|---|---|---|---|---|"]
    for path in a.preds:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        scopes = [("파일 전체", rows)]
        if any("cat" in r for r in rows):
            scopes += [("질환·약물 C+D", [r for r in rows if r.get("cat") in ("C", "D")])]
        for name, rs in scopes:
            s = summarize(rs)
            lines.append(f"| {Path(path).name} | {name} | {s['valid']}/{s['n']} · {s['errors']} | "
                         f"{ratio(s['correct'],s['n'])} | {ratio(s['answered'],s['n'])} | "
                         f"{ratio(s['answered']-s['wrong'],s['answered'])} | {s['wrong']}/{s['n']} |")
    lines.append("\n오류는 명시적 무응답과 구분한다. 낮은 응답률의 높은 정밀도를 항상 응답하는 기준선과 동등 조건의 우월성으로 해석하지 않는다. "
                 "동일 응답률 비교는 별도 dev에서 기준선 문턱을 동결한 뒤 새 test에서 측정해야 한다.")
    with open(a.out, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
