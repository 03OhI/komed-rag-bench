"""'규칙을 뺀 조건'(norule) 테스트 — 실험1·2_재설계_확정_1116.md §2.

같은 구조화 판독을 시키되 최종 범주를 규칙이 아니라 LLM 이 정한다 (Faithful CoT 의 'No solver' 절제와 같은 설계).

    cd mesh/bench && PYTHONIOENCODING=utf-8 python -m unittest test_norule -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESH = HERE.parent
for p in (str(MESH), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import verdict  # noqa: E402
import run_verdict as rv  # noqa: E402

V1_RULES = verdict.load_rules(str(MESH / "verdict_rules.json"))
DIM1 = {"causal": 1, "advice": 1, "species": 1, "specificity": 1}
ITEM = {"id": "t1", "dataset": "healthver", "claim": "c", "gold": "SUPPORT",
        "docs": [{"doc_id": "d", "title": "", "abstract": "x y z", "sentences": ["x y z"], "pubtypes": [], "year": ""}],
        "gold_rationale": ["x y z"], "meta": {}}


class Recorder:
    label = "mock:norule"

    def __init__(self, category, stance="supports"):
        self.category, self.stance, self.calls = category, stance, []

    def call(self, system, prompt, tool):
        self.calls.append((system, prompt, tool))
        out = {"claim_coding": dict(DIM1), "explanation_ko": "",
               "evidence": [{"n": 1, "relevant": True, "stance": self.stance, "key_sentence": "x y z", "coding": dict(DIM1)}]}
        if self.category is not None:
            out["category"] = self.category
        return out


class NoRulePrompt(unittest.TestCase):
    def test_system_tells_llm_to_decide(self):
        s = rv.NORULE_SYSTEM
        self.assertNotEqual(s, verdict.SYSTEM)
        self.assertNotIn("고정 규칙이 정한다", s)
        self.assertIn("직접 정한다", s)
        for cat in ("지지", "범위 초과", "반박", "근거 없음"):
            self.assertIn(cat, s)

    def test_system_gives_no_thresholds(self):
        # 임계값·가중 규칙은 주지 않는다 (사전 기록 §2)
        for word in ("임계", "가중", "exceed_threshold", "mixed_ratio"):
            self.assertNotIn(word, rv.NORULE_SYSTEM)

    def test_tool_has_required_category(self):
        t = rv.NORULE_TOOL
        self.assertEqual(t["input_schema"]["properties"]["category"]["enum"], ["지지", "범위 초과", "반박", "근거 없음"])
        self.assertIn("category", t["input_schema"]["required"])
        self.assertNotEqual(t["name"], verdict.TOOL["name"])
        self.assertNotIn("category", verdict.TOOL["input_schema"]["properties"])   # v1 도구는 그대로


class NoRuleRunItem(unittest.TestCase):
    def test_uses_norule_prompt_and_tool(self):
        llm = Recorder("지지")
        rv.run_item(ITEM, "norule", llm, V1_RULES)
        system, prompt, tool = llm.calls[0]
        self.assertIs(system, rv.NORULE_SYSTEM)
        self.assertIs(tool, rv.NORULE_TOOL)
        self.assertIn("category", prompt.splitlines()[-1])
        self.assertNotIn("submit_reading 으로 제출하라", prompt)

    def test_llm_category_decides_pred(self):
        # 판독은 지지인데 LLM 이 근거 없음을 고르면 판정은 NEI — 규칙이 냈을 범주는 따로 기록한다
        rec = rv.run_item(ITEM, "norule", Recorder("근거 없음"), V1_RULES)
        self.assertEqual((rec["pred"], rec["category"], rec["rule_category"], rec["consistent"]),
                         ("NEI", "근거 없음", "지지", False))

    def test_consistent_when_same_as_rule(self):
        rec = rv.run_item(ITEM, "norule", Recorder("지지"), V1_RULES)
        self.assertEqual((rec["pred"], rec["rule_category"], rec["consistent"]), ("SUPPORT", "지지", True))

    def test_exceed_maps_like_ours(self):
        rec = rv.run_item(ITEM, "norule", Recorder("범위 초과"), V1_RULES)
        self.assertEqual((rec["pred"], rec["category"]), ("SUPPORT", "범위 초과"))

    def test_missing_or_invalid_category_is_error(self):
        for bad in (None, "SUPPORT", "지지함"):
            rec = rv.run_item(ITEM, "norule", Recorder(bad), V1_RULES)
            self.assertEqual(rec["pred"], "ERROR", bad)

    def test_records_reading_fields(self):
        rec = rv.run_item(ITEM, "norule", Recorder("반박", stance="refutes"), V1_RULES)
        self.assertEqual((rec["stances"], rec["n_relevant"], rec["rule_category"]), (["refutes"], 1, "반박"))
        self.assertEqual(rec["key_sentences"], ["x y z"])


class ReportMetrics(unittest.TestCase):
    """report_e1_final.py — 실험 1 최종표(직접 판정 · 규칙을 뺀 조건 · KoMed-RAG)의 지표 계산."""
    G = ["SUPPORT", "SUPPORT", "REFUTE", "NEI", "NEI"]
    P = ["SUPPORT", "NEI", "SUPPORT", "NEI", "REFUTE"]

    def test_triplet_and_truthfulness(self):
        import report_e1_final as rep
        m = rep.metrics(self.G, self.P)
        self.assertEqual((m["n"], m["correct"], m["missed"], m["wrong"]), (5, 2, 1, 2))
        self.assertEqual(m["correct"] + m["missed"] + m["wrong"], m["n"])
        self.assertAlmostEqual(m["truth"], 0.0)

    def test_answered_precision_flips_rejection(self):
        import report_e1_final as rep
        m = rep.metrics(self.G, self.P)
        self.assertEqual((m["answered"], m["flips"], m["n_sr"]), (3, 1, 3))
        self.assertAlmostEqual(m["prec"], 100 / 3)
        self.assertAlmostEqual(m["coverage"], 60.0)
        self.assertAlmostEqual(m["nei_reject"], 50.0)
        self.assertAlmostEqual(m["acc"], 40.0)

    def test_label_with_exceed_to_nei(self):
        import report_e1_final as rep
        r = {"pred": "SUPPORT", "category": "범위 초과"}
        self.assertEqual(rep.label(r), "SUPPORT")
        self.assertEqual(rep.label(r, exceed_to_nei=True), "NEI")
        self.assertEqual(rep.label({"pred": "SUPPORT"}, exceed_to_nei=True), "SUPPORT")   # 직접 판정에는 범주가 없다

    def test_penalty_breakeven(self):
        import report_e1_final as rep
        # 정답 +194, 오답 +63 → 오답 벌점 c 가 194/63 보다 작을 때 앞선다
        self.assertAlmostEqual(rep.breakeven({"correct": 1194, "wrong": 127}, {"correct": 1000, "wrong": 64}), 194 / 63)
        self.assertIsNone(rep.breakeven({"correct": 10, "wrong": 5}, {"correct": 8, "wrong": 5}))   # 오답이 안 늘면 벌점과 무관

    def test_cluster_boot_zero_for_identical(self):
        import report_e1_final as rep
        lo, hi = rep.boot_diff(self.G, self.P, self.P, ["a", "a", "b", "c", "c"], lambda m: m["truth"], n_boot=200)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_read_keeps_last_record_per_id(self):
        # 11/13 파일에는 오류 기록 뒤에 재시도 기록이 있다 — id 별 마지막 기록을 쓰고, 마지막이 오류면 멈춘다
        import json
        import os
        import report_e1_final as rep
        from test_eval_v2 import workspace_temp
        with workspace_temp() as d:
            p = os.path.join(d, "a.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x", "gold": "NEI", "pred": "ERROR"}) + "\n")
                f.write(json.dumps({"id": "x", "gold": "NEI", "pred": "NEI"}) + "\n")
            self.assertEqual(rep.read([p])["x"]["pred"], "NEI")
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": "y", "gold": "NEI", "pred": "ERROR"}) + "\n")
            with self.assertRaises(ValueError):
                rep.read([p])

    def test_consistency_rate(self):
        import report_e1_final as rep
        recs = [{"consistent": True}, {"consistent": False}, {"consistent": True}, {"consistent": True}]
        self.assertAlmostEqual(rep.consistency(recs), 75.0)


if __name__ == "__main__":
    unittest.main()
