"""compare_exagg.py 단위 시험 — 실험 3(과장 탐지) 방식별 점수와 짝지은 차이."""
import json
import os
import tempfile
import unittest

import numpy as np

import compare_exagg as C


class F1Test(unittest.TestCase):
    def test_f1_class_counts_tp_fp_fn(self):
        gold = np.array([1, 1, 0, 2])
        pred = np.array([1, 0, 0, 2])
        self.assertAlmostEqual(C.f1_class(gold, pred, 1), 2 / 3)   # tp 1 · fp 0 · fn 1

    def test_f1_class_is_zero_without_true_positive(self):
        self.assertEqual(C.f1_class(np.array([1, 0]), np.array([0, 0]), 1), 0.0)

    def test_macro_f1_is_100_when_all_correct(self):
        g = np.array([0, 1, 2, 1])
        self.assertAlmostEqual(C.macro_f1(g, g), 100.0)

    def test_exagg_f1_scores_only_the_exaggerates_class(self):
        gold = np.array([1, 1, 0, 2])
        pred = np.array([1, 0, 2, 0])                               # 다른 두 범주는 모두 틀림
        self.assertAlmostEqual(C.exagg_f1(gold, pred), 100 * 2 / 3)


class BootTest(unittest.TestCase):
    def setUp(self):
        self.gold = np.array([0, 1, 2] * 20)
        self.good = self.gold.copy()
        self.bad = np.where(self.gold == 1, 0, self.gold)           # 과장을 전부 놓침
        self.d = C.draws(len(self.gold), n_boot=200, seed=0)

    def test_draws_are_reproducible_for_a_seed(self):
        self.assertTrue((C.draws(60, n_boot=200, seed=0) == self.d).all())
        self.assertEqual(self.d.shape, (200, 60))

    def test_diff_ci_is_zero_for_identical_predictions(self):
        self.assertEqual(C.diff_ci(self.gold, self.good, self.good, C.exagg_f1, self.d), (0.0, 0.0, 0.0))

    def test_diff_ci_excludes_zero_when_one_side_misses_every_exaggeration(self):
        delta, lo, hi = C.diff_ci(self.gold, self.good, self.bad, C.exagg_f1, self.d)
        self.assertAlmostEqual(delta, 100.0)
        self.assertGreater(lo, 0.0)
        self.assertLessEqual(lo, hi)

    def test_point_ci_brackets_the_point_estimate(self):
        noisy = self.gold.copy()
        noisy[:6] = (noisy[:6] + 1) % 3
        point, lo, hi = C.point_ci(self.gold, noisy, C.macro_f1, self.d)
        self.assertLess(point, 100.0)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)


class AlignTest(unittest.TestCase):
    @staticmethod
    def rec(i, gold, pred):
        return {"id": i, "gold": gold, "pred": pred}

    def test_aligned_orders_by_id_and_encodes_labels(self):
        a = {"b": self.rec("b", "same", "exaggerates"), "a": self.rec("a", "downplays", "downplays")}
        ids, gold, pred = C.aligned({"direct": a})
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(list(gold), [2, 0])
        self.assertEqual(list(pred["direct"]), [2, 1])

    def test_aligned_rejects_modes_with_different_ids(self):
        a = {"a": self.rec("a", "same", "same")}
        b = {"z": self.rec("z", "same", "same")}
        with self.assertRaises(ValueError):
            C.aligned({"direct": a, "ours4": b})

    def test_aligned_rejects_a_prediction_outside_the_three_labels(self):
        with self.assertRaises(ValueError):
            C.aligned({"direct": {"a": self.rec("a", "same", "ERROR")}})

    def test_aligned_rejects_modes_that_disagree_on_gold(self):
        a = {"a": self.rec("a", "same", "same")}
        b = {"a": self.rec("a", "exaggerates", "same")}
        with self.assertRaises(ValueError):
            C.aligned({"direct": a, "ours4": b})


class StrengthTest(unittest.TestCase):
    def test_strength_scores_average_f1_over_levels_that_occur(self):
        recs = [{"pr_strength": 0, "s1_4": 0}, {"pr_strength": 1, "s1_4": 1}, {"pr_strength": 3, "s1_4": 1}]
        mf1, acc = C.strength_scores(recs, "pr_strength", "s1_4")
        # 단계 0: 1.0 · 단계 1: 2/3 · 단계 3: 0.0 — 단계 2 는 정답에도 판독에도 없어 평균에서 뺀다
        self.assertAlmostEqual(mf1, 100 * (1 + 2 / 3 + 0) / 3)
        self.assertAlmostEqual(acc, 100 * 2 / 3)


class AttributionTest(unittest.TestCase):
    @staticmethod
    def rec(pr_gold, ab_gold, pr_read, ab_read, pred=None):
        gold = C.rule_label(pr_gold, ab_gold)
        return {"gold": gold, "pr_strength": pr_gold, "abs_strength": ab_gold, "s1_4": pr_read, "s2_4": ab_read,
                "pred": pred if pred is not None else C.rule_label(pr_read, ab_read)}

    def test_rule_label_compares_the_two_strengths(self):
        self.assertEqual([C.rule_label(3, 1), C.rule_label(1, 3), C.rule_label(2, 2)], ["exaggerates", "downplays", "same"])

    def test_attribution_assigns_each_error_to_the_side_that_was_misread(self):
        recs = [self.rec(3, 3, 3, 3),          # 정답 — 둘 다 맞게 읽음
                self.rec(3, 3, 3, 2),          # 오답 — 초록만 잘못 읽음
                self.rec(1, 1, 2, 1),          # 오답 — 보도자료만 잘못 읽음
                self.rec(3, 1, 1, 3),          # 오답 — 둘 다 잘못 읽음
                self.rec(3, 1, 2, 0)]          # 정답 — 둘 다 잘못 읽었지만 비교 결과는 같다
        a = C.attribution(recs)
        self.assertEqual((a["n"], a["errors"]), (5, 3))
        self.assertEqual((a["err_pr_only"], a["err_abs_only"], a["err_both"], a["err_neither"]), (1, 1, 1, 0))
        self.assertEqual(a["correct_despite_misreading"], 1)
        self.assertEqual(a["both_read_correctly"], 1)

    def test_attribution_counts_predictions_that_the_stored_reading_reproduces(self):
        recs = [self.rec(3, 3, 3, 3), self.rec(3, 1, 3, 1, pred="same")]      # 둘째는 판독으로 예측이 다시 나오지 않는다
        a = C.attribution(recs)
        self.assertEqual((a["replay_match"], a["gold_from_strengths"]), (1, 2))
        self.assertEqual(a["err_neither"], 1)                                  # 둘 다 맞게 읽고도 틀린 건이 있으면 드러난다

    def test_attribution_reports_which_way_each_side_was_misread(self):
        recs = [self.rec(3, 3, 3, 2), self.rec(1, 3, 2, 2), self.rec(2, 0, 1, 1)]
        a = C.attribution(recs)
        self.assertEqual((a["pr_read_stronger"], a["pr_read_weaker"]), (1, 1))
        self.assertEqual((a["abs_read_stronger"], a["abs_read_weaker"]), (1, 2))


class RenderTest(unittest.TestCase):
    def write(self, d, mode, rows):
        with open(os.path.join(d, f"exagg_{mode}_{C.MODEL}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rows(self, with_reading):
        out = []
        for i, g in enumerate(["same", "exaggerates", "downplays"] * 4):
            r = {"id": str(i), "gold": g, "pred": g, "pr_strength": i % 4, "abs_strength": i % 4}
            if with_reading:
                r.update({"s1_4": i % 4, "s2_4": i % 4})
            out.append(r)
        return out

    def test_render_lists_every_mode_and_puts_the_registered_comparison_first(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in C.MODES:
                self.write(d, mode, self.rows(with_reading=mode.startswith("ours")))
            text = C.render(d, n_boot=50)
        for mode in C.MODES:
            self.assertIn(f"| {mode} |", text)
        self.assertLess(text.index("ours4 − direct"), text.index("ours7 − direct"))
        self.assertIn("n=12", text)

    def test_render_marks_strength_column_empty_for_modes_without_a_reading(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in C.MODES:
                self.write(d, mode, self.rows(with_reading=mode.startswith("ours")))
            text = C.render(d, n_boot=50)
        direct_row = next(l for l in text.splitlines() if l.startswith("| direct |"))
        ours_row = next(l for l in text.splitlines() if l.startswith("| ours4 |"))
        self.assertTrue(direct_row.rstrip().endswith("| - |"))
        self.assertIn("100.0 / 100.0", ours_row)

    def test_render_adds_an_attribution_row_for_each_mode_that_keeps_a_reading(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in C.MODES:
                self.write(d, mode, self.rows(with_reading=mode.startswith("ours")))
            text = C.render(d, n_boot=50)
        section = text[text.index("오답의 판독 귀속"):]
        rows = [l for l in section.splitlines() if l.startswith("| ours")]
        self.assertEqual([l.split("|")[1].strip() for l in rows], ["ours4", "ours7"])
        self.assertNotIn("| direct |", section)


if __name__ == "__main__":
    unittest.main()
