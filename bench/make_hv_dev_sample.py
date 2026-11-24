"""HealthVer dev 에서 개선 작업용 표본 두 벌을 시드 고정으로 뽑는다 (2026-11-15).

  dev_a 300 : 원인 확인 · 개선안 조정 (여러 번 본다)
  dev_b 300 : 고정한 개선안을 한 번 확인 (dev_a 에 과적합됐는지 본다)

두 표본은 겹치지 않는다. test 는 건드리지 않는다.
    python make_hv_dev_sample.py
"""
import csv
import random
from collections import Counter

SRC = "data/healthver/data/healthver_dev.csv"
SEED = 20261115
N = 300

with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

rng = random.Random(SEED)
idx = list(range(len(rows)))
rng.shuffle(idx)
for name, part in (("dev_a", idx[:N]), ("dev_b", idx[N:2 * N])):
    out = f"data/healthver/data/healthver_{name}300_s{SEED}.csv"
    sel = [rows[i] for i in sorted(part)]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sel)
    print(out, len(sel), dict(Counter(r["label"] for r in sel)),
          "topics", len({r["topic_ip"] for r in sel}))
