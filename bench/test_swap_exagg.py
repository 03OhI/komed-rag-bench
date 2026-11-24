"""make_exagg_swap.py · swap_exagg.py 단위 시험 — 실험 3 자리 바꾸기 검사."""
import csv
import json
import os
import tempfile
import unittest

import numpy as np

import compare_exagg as C
import make_exagg_swap as MK
import swap_exagg as S

SAME, EXAG, DOWN = 0, 1, 2          # compare_exagg.LAB 의 순서


def row(i, pr, ab, ps, as_, label):
    return {"original_file_id": i, "press_release_conclusion": pr, "press_release_strength": str(ps),
            "abstract_conclusion": ab, "abstract_strength": str(as_), "exaggeration_label": label}


class MakeSwapTest(unittest.TestCase):
    def test_swap_row_exchanges_sentences_and_strengths_and_mirrors_the_label(self):
        out = MK.swap_row(row("a", "PR text", "ABS text", 3, 1, "exaggerates"))
        self.assertEqual(out["original_file_id"], "a")
        self.assertEqual((out["press_release_conclusion"], out["abstract_conclusion"]), ("ABS text", "PR text"))
        self.assertEqual((out["press_release_strength"], out["abstract_strength"]), ("1", "3"))
        self.assertEqual(out["exaggeration_label"], "downplays")

    def test_swap_row_keeps_same_as_same(self):
        self.assertEqual(MK.swap_row(row("a", "x", "y", 2, 2, "same"))["exaggeration_label"], "same")

    def test_swapping_twice_returns_the_original_row(self):
        r = row("a", "PR text", "ABS text", 0, 2, "downplays")
        self.assertEqual(MK.swap_row(MK.swap_row(r)), r)

    def test_swap_rows_stops_when_a_label_does_not_follow_from_the_strengths(self):
        with self.assertRaises(ValueError):                       # 강도는 3 > 1 인데 라벨이 same — 뒤집은 정답을 믿을 수 없다
            MK.swap_rows([row("a", "x", "y", 3, 1, "same")])

    def test_written_file_loads_with_the_benchmark_loader(self):
        from data_loaders import load_exagg
        rows = [row("a", "PR a", "ABS a", 3, 1, "exaggerates"), row("b", "PR b", "ABS b", 1, 1, "same")]
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "src.csv"), os.path.join(d, "dst.csv")
            with open(src, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            n = MK.write_swapped(src, dst)
            items = load_exagg(dst)
        self.assertEqual(n, 2)
        self.assertEqual([(it["id"], it["pr"], it["abs"], it["pr_strength"], it["abs_strength"], it["gold"]) for it in items],
                         [("a", "ABS a", "PR a", 1, 3, "downplays"), ("b", "ABS b", "PR b", 1, 1, "same")])


class MetricTest(unittest.TestCase):
    def test_mirror_swaps_exaggerates_and_downplays_only(self):
        self.assertEqual(list(S.mirror(np.array([SAME, EXAG, DOWN]))), [SAME, DOWN, EXAG])

    def test_consistency_counts_pairs_whose_swapped_prediction_is_the_mirror(self):
        o = np.array([EXAG, EXAG, SAME, DOWN])
        s = np.array([DOWN, EXAG, SAME, SAME])                    # 1번째·3번째만 뒤집힘이 맞다
        self.assertAlmostEqual(S.consistency(o, s), 50.0)

    def test_directional_consistency_ignores_pairs_first_judged_same(self):
        o = np.array([EXAG, EXAG, SAME, DOWN])
        s = np.array([DOWN, EXAG, SAME, SAME])                    # 방향 판정 3쌍 중 1쌍만 맞게 뒤집힘
        self.assertAlmostEqual(S.consistency_directional(o, s), 100 / 3)

    def test_directional_consistency_is_nan_without_any_directional_judgment(self):
        self.assertTrue(np.isnan(S.consistency_directional(np.array([SAME, SAME]), np.array([SAME, EXAG]))))

    def test_role_bias_is_zero_for_a_judge_that_follows_content(self):
        o = np.array([EXAG, DOWN, SAME])
        self.assertAlmostEqual(S.role_bias(o, S.mirror(o)), 0.0)

    def test_role_bias_is_positive_for_a_judge_that_always_blames_the_press_release_slot(self):
        o = np.array([EXAG, EXAG, SAME, SAME])
        s = np.array([EXAG, EXAG, SAME, SAME])                    # 자리를 바꿔도 여전히 "과장"
        self.assertAlmostEqual(S.role_bias(o, s), 100 * 4 / 8)

    def test_always_same_is_fully_consistent_but_has_no_directional_judgment(self):
        o = s = np.array([SAME, SAME, SAME])
        self.assertAlmostEqual(S.consistency(o, s), 100.0)
        self.assertAlmostEqual(S.role_bias(o, s), 0.0)
        self.assertTrue(np.isnan(S.consistency_directional(o, s)))


class BootTest(unittest.TestCase):
    def setUp(self):
        self.o = np.array([EXAG, DOWN, SAME] * 20)
        self.good = S.mirror(self.o)                               # 완전히 뒤집는 판정
        self.stuck = self.o.copy()                                 # 자리를 바꿔도 같은 답 — 방향 판정은 전부 어긋남
        self.d = C.draws(len(self.o), n_boot=200, seed=0)

    def test_pair_ci_brackets_the_point_estimate(self):
        point, lo, hi = S.pair_ci(S.consistency, self.o, self.stuck, self.d)
        self.assertAlmostEqual(point, 100 / 3)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_pair_diff_excludes_zero_when_one_judge_mirrors_and_the_other_does_not(self):
        delta, lo, hi = S.pair_diff(S.consistency, (self.o, self.good), (self.o, self.stuck), self.d)
        self.assertAlmostEqual(delta, 100 - 100 / 3)
        self.assertGreater(lo, 0.0)

    def test_pair_diff_is_zero_for_identical_judges(self):
        self.assertEqual(S.pair_diff(S.consistency, (self.o, self.good), (self.o, self.good), self.d), (0.0, 0.0, 0.0))

    def test_combined_scores_resample_both_orientations_of_a_pair_together(self):
        gold = np.array([EXAG, DOWN, SAME] * 20)
        point, lo, hi = S.combined_ci(C.exagg_f1, gold, gold, S.mirror(gold), S.mirror(gold), self.d)
        self.assertAlmostEqual(point, 100.0)                       # 두 방향 모두 정답
        self.assertAlmostEqual(lo, 100.0)
        half, _, _ = S.combined_ci(C.exagg_f1, gold, gold, S.mirror(gold), gold, self.d)
        self.assertLess(half, 100.0)                               # 바꾼 방향에서 뒤집지 않으면 점수가 깎인다


class RenderTest(unittest.TestCase):
    def write(self, d, mode, rows):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"exagg_{mode}_{C.MODEL}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d_or, self.d_sw = os.path.join(self.tmp.name, "or"), os.path.join(self.tmp.name, "sw")
        golds = ["same", "exaggerates", "downplays"] * 4
        mir = {"same": "same", "exaggerates": "downplays", "downplays": "exaggerates"}
        for mode in C.MODES:
            self.write(self.d_or, mode, [{"id": str(i), "gold": g, "pred": g} for i, g in enumerate(golds)])
            stuck = mode.startswith("direct")                      # direct 계열은 자리를 바꿔도 같은 답을 낸다고 꾸민다
            self.write(self.d_sw, mode, [{"id": str(i), "gold": mir[g], "pred": (g if stuck else mir[g])} for i, g in enumerate(golds)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_render_reports_every_mode_and_the_registered_comparison_first(self):
        text = S.render(self.d_or, self.d_sw, n_boot=50)
        for mode in C.MODES:
            self.assertIn(f"| {mode} |", text)
        self.assertLess(text.index("ours4 − direct"), text.index("ours7 − direct"))
        ours = next(l for l in text.splitlines() if l.startswith("| ours4 |"))
        direct = next(l for l in text.splitlines() if l.startswith("| direct |"))
        self.assertIn("100.0", ours)
        self.assertIn("33.3", direct)

    def test_render_stops_when_swapped_gold_is_not_the_mirror_of_the_original_gold(self):
        self.write(self.d_sw, "direct", [{"id": str(i), "gold": "same", "pred": "same"} for i in range(12)])
        with self.assertRaises(ValueError):
            S.render(self.d_or, self.d_sw, n_boot=50)


if __name__ == "__main__":
    unittest.main()
