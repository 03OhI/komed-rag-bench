"""E1 합산 · 주장 단위 분석 (2026-11-13)

입력: preds_e1_1113/ (HealthVer 1,823 · Check-COVID composed 180) + preds_e1_1113_ccext/ (Check-COVID extracted 49)
출력: preds_e1_1113/report_e1_pooled.md · preds_e1_1113/e1_pooled_1113.json

분석 집합
  HV1823        HealthVer test
  CC229         Check-COVID test composed 180 + extracted 49 (원본 행 그대로)
  CC226         CC229 에서 (주장 문장, 문헌, 라벨)이 같은 중복 3행을 뺀 것 (먼저 나온 주장 id 유지)
  POOL2052      HV1823 + CC229 (항목 단위로 합침)
  POOL2049      HV1823 + CC226
  (보조) CC180 composed · CC49 extracted

주장 단위(군집) 정의
  HealthVer   : 주장 문장 (230개)
  Check-COVID : 주장 id 접미사(_0/_1/_2)를 뗀 원 주장 id 와 주장 문장을 union-find 로 묶음
                (같은 뉴스 주장에서 만든 지지·반박·NEI 변형, 그리고 다른 id 로 중복 등록된 같은 문장을 한 군집으로)
  합산 집합에서는 데이터셋 접두사로 군집을 분리한다.

통계
  CI      : 항목 재표집 부트스트랩(기존 report.py 와 같은 방식) + 주장 단위 군집 부트스트랩, 2,000회, 백분위
  짝지은 p: 주장 단위 군집 무작위화 검정 — 군집마다 ours/free 예측을 통째로 1/2 확률로 맞바꿈, 20,000회 (양측)
            검정 통계: ① Δtruthfulness ② Δ위험한 뒤집힘 비율 ③ Δmacro-F1 ④ Δ정확도
  위험한 뒤집힘: 정답 SUPPORT/REFUTE 중 예측이 반대 극인 비율 (metrics.dangerous_flip 과 같음), Wilson CI 병기
  truthfulness 민감도: 정답 NEI 를 NEI 로 맞힌 경우 +1(기본, metrics.truthfulness) / 0 (M5 요청)
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import (macro_f1 as ref_macro_f1, dangerous_flip as ref_flip, truthfulness as ref_truth,  # noqa: E402
                     wilson, holm, mcnemar_exact, mcnemar_bool, flip_flags, paired_signflip, truth_scores)

LLM = "gemini-gemini-3.1-pro-preview"
LAB = {"SUPPORT": 0, "REFUTE": 1, "NEI": 2}
N_BOOT, N_PERM, SEED = 2000, 20000, 0


def read_preds(path):
    """report.read_preds 와 같은 규칙: id 별 마지막 기록, 지문 혼합 금지."""
    by_id, fps = {}, set()
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            fps.add(r.get("run_fingerprint", "legacy"))
            if len(fps) > 1:
                raise ValueError(f"Mixed protocols in {path}")
            by_id[r["id"]] = r
    return by_id


def load(ds, folder):
    f = read_preds(os.path.join(HERE, folder, f"{ds}_free_{LLM}.jsonl"))
    o = read_preds(os.path.join(HERE, folder, f"{ds}_ours_{LLM}.jsonl"))
    if set(f) != set(o):
        raise ValueError(f"{ds}/{folder}: free/ours id 불일치")
    items = []
    for i in f:
        a, b = f[i], o[i]
        if a["gold"] != b["gold"]:
            raise ValueError(f"gold mismatch {i}")
        for r in (a, b):
            if r.get("error") or r.get("pred") not in LAB:
                raise ValueError(f"오류/비정상 예측 남음: {i} {r.get('pred')}")
        items.append({"id": i, "ds": ds, "claim": a["claim"].strip(), "meta": a.get("meta") or {},
                      "gold": a["gold"], "free": a["pred"], "ours": b["pred"], "sub": folder})
    return items


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def assign_clusters(items):
    uf = UF()
    for it in items:
        if it["ds"] == "checkcovid":
            base = it["meta"]["claim_id"].rsplit("_", 1)[0]
            uf.u(("cc-id", base), ("cc-txt", it["claim"]))
    for it in items:
        if it["ds"] == "healthver":
            it["cl"] = "hv::" + it["claim"]
        else:
            it["cl"] = "cc::" + repr(uf.f(("cc-txt", it["claim"])))


# ───────────────────────────── 벡터화 지표
def conf_counts(g, p):
    return np.bincount(g * 3 + p, minlength=9).reshape(3, 3)


def m_macro_f1(g, p):
    c = conf_counts(g, p)
    tp = np.diag(c).astype(float)
    fp = c.sum(0) - tp
    fn = c.sum(1) - tp
    P = np.divide(tp, tp + fp, out=np.zeros(3), where=(tp + fp) > 0)
    R = np.divide(tp, tp + fn, out=np.zeros(3), where=(tp + fn) > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros(3), where=(P + R) > 0)
    return F.mean()


def m_acc(g, p):
    return float((g == p).mean())


def m_flip(g, p):
    sr = g < 2
    return float(((p == 1 - g) & sr).sum() / max(1, sr.sum()))


def m_truth(g, p, nei_correct=1.0):
    s = np.where(g == p, 1.0, np.where(p == 2, 0.0, -1.0))
    s = np.where((g == 2) & (p == 2), nei_correct, s)
    return float(s.mean())


def m_truth0(g, p):
    return m_truth(g, p, 0.0)


METRICS = {"macro_f1": m_macro_f1, "acc": m_acc, "flip": m_flip, "truth": m_truth, "truth_nei0": m_truth0}


def pct_ci(vals):
    vals = np.sort(np.asarray(vals))
    n = len(vals)
    return float(vals[int(0.025 * n)]), float(vals[min(n - 1, int(0.975 * n))])


def analyse(name, items):
    g = np.array([LAB[i["gold"]] for i in items])
    pf = np.array([LAB[i["free"]] for i in items])
    po = np.array([LAB[i["ours"]] for i in items])
    cl_names = sorted({i["cl"] for i in items})
    cl_idx = {c: k for k, c in enumerate(cl_names)}
    cid = np.array([cl_idx[i["cl"]] for i in items])
    members = [np.where(cid == k)[0] for k in range(len(cl_names))]
    n, K = len(items), len(cl_names)

    rng = np.random.default_rng(SEED)
    item_bs = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    rng = np.random.default_rng(SEED + 1)
    clus_bs = [np.concatenate([members[k] for k in rng.integers(0, K, K)]) for _ in range(N_BOOT)]

    out = {"name": name, "n": n, "n_clusters": K, "gold": dict(Counter(i["gold"] for i in items)),
           "n_gold_SR": int((g < 2).sum()), "cond": {}, "paired": {}}
    for cond, p in (("free", pf), ("ours", po)):
        d = {}
        for mk, fn in METRICS.items():
            d[mk] = float(fn(g, p))
            d[mk + "_ci_item"] = pct_ci([fn(g[ix], p[ix]) for ix in item_bs])
            d[mk + "_ci_claim"] = pct_ci([fn(g[ix], p[ix]) for ix in clus_bs])
        k_flip = int(((p == 1 - g) & (g < 2)).sum())
        d["flip_k"] = k_flip
        d["flip_wilson"] = list(wilson(k_flip, int((g < 2).sum())))
        d["pred_dist"] = dict(Counter(i[cond] for i in items))
        # 기존 metrics.py 와 수치 일치 확인
        gl = [i["gold"] for i in items]
        pl = [i[cond] for i in items]
        assert abs(ref_macro_f1(gl, pl) - d["macro_f1"]) < 1e-12
        assert abs(ref_flip(gl, pl)[0] - d["flip"]) < 1e-12
        assert abs(ref_truth(gl, pl) - d["truth"]) < 1e-12
        out["cond"][cond] = d

    # 짝지은 차이 (ours − free)
    rng = np.random.default_rng(SEED + 2)
    swaps = rng.random((N_PERM, K)) < 0.5
    gl = [i["gold"] for i in items]
    fl, ol = [i["free"] for i in items], [i["ours"] for i in items]
    for mk, fn in METRICS.items():
        obs = fn(g, po) - fn(g, pf)
        ci_item = pct_ci([fn(g[ix], po[ix]) - fn(g[ix], pf[ix]) for ix in item_bs])
        ci_claim = pct_ci([fn(g[ix], po[ix]) - fn(g[ix], pf[ix]) for ix in clus_bs])
        cnt = 0
        for s in swaps:
            m = s[cid]
            a = np.where(m, po, pf)
            b = np.where(m, pf, po)
            if abs(fn(g, a) - fn(g, b)) >= abs(obs) - 1e-12:
                cnt += 1
        out["paired"][mk] = {"diff": float(obs), "ci_item": ci_item, "ci_claim": ci_claim,
                             "p_claim": (cnt + 1) / (N_PERM + 1)}
    # 기존 항목 단위 탐색 검정 (report.py 와 같은 함수)
    _, p_item_truth = paired_signflip(truth_scores(gl, fl), truth_scores(gl, ol))
    mb = mcnemar_bool(flip_flags(gl, ol), flip_flags(gl, fl))
    mc = mcnemar_exact(gl, fl, ol)
    out["paired"]["truth"]["p_item"] = p_item_truth
    out["paired"]["flip"]["p_item"] = mb["p"]
    out["paired"]["flip"]["ours_only"], out["paired"]["flip"]["free_only"] = mb["a_only"], mb["b_only"]
    out["paired"]["acc"]["p_item"] = mc["p"]
    out["paired"]["acc"]["ours_only"], out["paired"]["acc"]["free_only"] = mc["b_only"], mc["a_only"]
    return out


def main():
    hv = load("healthver", "preds_e1_1113")
    cc180 = load("checkcovid", "preds_e1_1113")
    cc49 = load("checkcovid", "preds_e1_1113_ccext")
    if set(i["id"] for i in cc180) & set(i["id"] for i in cc49):
        raise ValueError("composed/extracted id 겹침")
    cc229 = cc180 + cc49
    assign_clusters(hv + cc229)

    # 226쌍과의 차이: (주장 문장, 문헌, 라벨) 중복
    seen, dups = {}, []
    for it in cc229:
        key = (it["claim"], it["meta"]["doc_id"], it["gold"])
        if key in seen:
            dups.append((seen[key], it))
        else:
            seen[key] = it
    drop = {d["id"] for _, d in dups}
    cc226 = [i for i in cc229 if i["id"] not in drop]
    dup_info = [{"kept": a["id"], "dropped": b["id"], "gold": a["gold"],
                 "free": [a["free"], b["free"]], "ours": [a["ours"], b["ours"]], "claim": a["claim"]} for a, b in dups]

    sets = [("HV1823", hv), ("CC229", cc229), ("CC226", cc226), ("POOL2052", hv + cc229), ("POOL2049", hv + cc226),
            ("CC180_composed", cc180), ("CC49_extracted", cc49)]
    res = {"meta": {"llm": LLM, "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED}, "duplicates": dup_info, "sets": {}}
    for name, items in sets:
        print("…", name, len(items), file=sys.stderr)
        res["sets"][name] = analyse(name, items)

    # Holm: 사전 등록 주 검정 ①② 를 데이터셋 간(HV, CC) 보정 — 주장 단위 p
    holm_rows = {}
    for cc_name in ("CC229", "CC226"):
        for mk in ("truth", "flip"):
            ps = [res["sets"]["HV1823"]["paired"][mk]["p_claim"], res["sets"][cc_name]["paired"][mk]["p_claim"]]
            holm_rows[f"{cc_name}:{mk}"] = holm(ps)
    res["holm_claim"] = holm_rows

    with open(os.path.join(HERE, "preds_e1_1113", "e1_pooled_1113.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_md(res)


def write_md(res):
    S = res["sets"]
    L = ["# E1 합산 · 주장 단위 분석 (자동 생성 — e1_pooled_1113.py, 2026-11-13)\n",
         f"판독 LLM {LLM}. 조건 free(직접 판정) · ours(규칙 판정). 오류 0건(재시도 반영 후, id별 마지막 기록). "
         f"CI: 항목 재표집과 주장 단위 군집 재표집, 각 {N_BOOT}회 백분위. "
         f"주장 단위 p: 군집마다 두 조건 예측을 통째로 맞바꾸는 무작위화 검정 {N_PERM}회(양측). "
         "주장 단위: HealthVer는 주장 문장, Check-COVID는 원 주장 id(접미사 _0/_1/_2 제거)와 주장 문장을 union-find로 묶은 군집. "
         "합산 집합은 항목을 그대로 합쳐 계산했다(데이터셋 가중 없음).\n"]

    L.append("\n## 1. 조건별 지표\n")
    L.append("| 집합 | n (주장 군집) | 조건 | macro-F1 [항목 CI] [주장 CI] | Acc | 위험한 뒤집힘 % (k/n S·R) [Wilson] [주장 CI] | truthfulness [주장 CI] | truthfulness, 정답 NEI=0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in S.items():
        for cond in ("free", "ours"):
            d = s["cond"][cond]
            L.append(f"| {name} | {s['n']} ({s['n_clusters']}) | {cond} | "
                     f"{100*d['macro_f1']:.1f} [{100*d['macro_f1_ci_item'][0]:.1f}, {100*d['macro_f1_ci_item'][1]:.1f}] "
                     f"[{100*d['macro_f1_ci_claim'][0]:.1f}, {100*d['macro_f1_ci_claim'][1]:.1f}] | {100*d['acc']:.1f} | "
                     f"{100*d['flip']:.1f} ({d['flip_k']}/{s['n_gold_SR']}) [{100*d['flip_wilson'][0]:.1f}, {100*d['flip_wilson'][1]:.1f}] "
                     f"[{100*d['flip_ci_claim'][0]:.1f}, {100*d['flip_ci_claim'][1]:.1f}] | "
                     f"{d['truth']:+.3f} [{d['truth_ci_claim'][0]:+.3f}, {d['truth_ci_claim'][1]:+.3f}] | {d['truth_nei0']:+.3f} |")

    L.append("\n## 2. 짝지은 차이 (ours − free)\n")
    L.append("주 검정(2026-09-26 계획)은 ① truthfulness ② 위험한 뒤집힘이다. macro-F1·정확도는 표준 지표로 병기. "
             "'항목 p'는 기존 report.py의 탐색 검정(부호 뒤집기·McNemar)이고 '주장 p'가 확증용이다.\n")
    L.append("| 집합 | ① Δtruthfulness [주장 CI] · 주장 p (항목 p) | ② Δ위험한 뒤집힘 %p (ours만:free만) [주장 CI] · 주장 p (항목 p) | Δmacro-F1 [항목 CI] [주장 CI] · 주장 p | ΔAcc (ours만:free만) · 주장 p (항목 p) | Δtruthfulness(NEI=0) · 주장 p |")
    L.append("|---|---|---|---|---|---|")
    for name, s in S.items():
        P = s["paired"]
        t, f_, m, a, t0 = P["truth"], P["flip"], P["macro_f1"], P["acc"], P["truth_nei0"]
        L.append(f"| {name} | {t['diff']:+.3f} [{t['ci_claim'][0]:+.3f}, {t['ci_claim'][1]:+.3f}] · {t['p_claim']:.4f} ({t['p_item']:.4f}) | "
                 f"{100*f_['diff']:+.1f} ({f_['ours_only']}:{f_['free_only']}) [{100*f_['ci_claim'][0]:+.1f}, {100*f_['ci_claim'][1]:+.1f}] · {f_['p_claim']:.4f} ({f_['p_item']:.4f}) | "
                 f"{100*m['diff']:+.1f} [{100*m['ci_item'][0]:+.1f}, {100*m['ci_item'][1]:+.1f}] [{100*m['ci_claim'][0]:+.1f}, {100*m['ci_claim'][1]:+.1f}] · {m['p_claim']:.4f} | "
                 f"{100*a['diff']:+.1f} ({a['ours_only']}:{a['free_only']}) · {a['p_claim']:.4f} ({a['p_item']:.4f}) | "
                 f"{t0['diff']:+.3f} · {t0['p_claim']:.4f} |")

    L.append("\n**Holm 보정 (주장 단위 p, 검정별로 HealthVer·Check-COVID 두 데이터셋 간)**\n")
    L.append("| Check-COVID 집합 | ① truthfulness p (HV, CC) → Holm | ② 위험한 뒤집힘 p (HV, CC) → Holm |")
    L.append("|---|---|---|")
    for cc in ("CC229", "CC226"):
        pt = [S["HV1823"]["paired"]["truth"]["p_claim"], S[cc]["paired"]["truth"]["p_claim"]]
        pf = [S["HV1823"]["paired"]["flip"]["p_claim"], S[cc]["paired"]["flip"]["p_claim"]]
        ht, hf = res["holm_claim"][f"{cc}:truth"], res["holm_claim"][f"{cc}:flip"]
        L.append(f"| {cc} | {pt[0]:.4f}, {pt[1]:.4f} → {ht[0]:.4f}, {ht[1]:.4f} | {pf[0]:.4f}, {pf[1]:.4f} → {hf[0]:.4f}, {hf[1]:.4f} |")

    L.append("\n## 3. 226쌍과의 3건 차이\n")
    L.append("Check-COVID test 원본은 composed 180행 + extracted 49행 = 229행이다. (주장 문장, 문헌 cord_id, 라벨)이 같은 행을 하나로 세면 226쌍이 된다. "
             "중복 3행은 모두 composed에 있고, 같은 주장이 두 주장 id(602bcf116bc088b6d218dc2c, 6026e71e6bc088b6d218dc11)로 두 번 등록된 경우다. "
             "_0(REFUTE)·_1(SUPPORT)은 근거 문장 번호까지 같고, _2(NEI)는 근거로 붙은 다른 문장이 15번과 7번으로 다르다. CC226은 먼저 나온 id를 남겼다.\n")
    L.append("| 남긴 id | 뺀 id | 정답 | free 예측 (남김, 뺌) | ours 예측 (남김, 뺌) |")
    L.append("|---|---|---|---|---|")
    for d in res["duplicates"]:
        L.append(f"| {d['kept']} | {d['dropped']} | {d['gold']} | {d['free'][0]}, {d['free'][1]} | {d['ours'][0]}, {d['ours'][1]} |")

    with open(os.path.join(HERE, "preds_e1_1113", "report_e1_pooled.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
