"""과장 탐지 자료의 두 문장 자리를 바꾼 판을 만든다 — 실험 3 자리 바꾸기 검사.

보도자료 칸에 초록 문장, 초록 칸에 보도자료 문장을 넣고 강도 주석도 함께 바꾼다.
정답 라벨은 뒤집는다(exaggerates ↔ downplays, same → same). id 는 그대로 둔다.
정답 라벨이 두 강도 주석을 견준 값과 다른 행이 있으면 멈춘다(뒤집은 정답을 믿을 수 없으므로).

사용: python make_exagg_swap.py --src data/exagg_test.csv --dst data/exagg_test_swap.csv
사전 기록: claude_backup/실험3_자리바꾸기_사전기록_1116.md
"""
import argparse
import csv

MIRROR = {"same": "same", "exaggerates": "downplays", "downplays": "exaggerates"}
PR, AB = "press_release_conclusion", "abstract_conclusion"
PR_S, AB_S = "press_release_strength", "abstract_strength"
LABEL = "exaggeration_label"


def label_from_strengths(pr_strength, abs_strength):
    pr, ab = int(pr_strength), int(abs_strength)
    return "exaggerates" if pr > ab else "downplays" if pr < ab else "same"


def swap_row(row):
    out = dict(row)
    out[PR], out[AB] = row[AB], row[PR]
    out[PR_S], out[AB_S] = row[AB_S], row[PR_S]
    out[LABEL] = MIRROR[row[LABEL].strip().lower()]
    return out


def swap_rows(rows):
    out = []
    for r in rows:
        derived = label_from_strengths(r[PR_S], r[AB_S])
        if r[LABEL].strip().lower() != derived:
            raise ValueError(f"{r.get('original_file_id')}: 라벨 {r[LABEL]!r} 이 강도 비교 {derived!r} 와 다르다")
        out.append(swap_row(r))
    return out


def write_swapped(src, dst):
    with open(src, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    swapped = swap_rows(rows)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(swapped)
    return len(swapped)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    print(f"{a.src} → {a.dst}: {write_swapped(a.src, a.dst)}쌍")


if __name__ == "__main__":
    main()
