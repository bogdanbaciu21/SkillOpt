"""Paired A/B evalkit: known-answer stats, seeded null calibration, RESULTS replay."""
from __future__ import annotations

import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from skillopt_sleep.evalkit import (
    EvalkitError,
    bootstrap_delta_ci,
    compare,
    compare_aa,
    exact_mcnemar_p,
    format_markdown,
    mcnemar_from_counts,
    mcnemar_paired,
    reconstruct_paired_from_rates,
)
from skillopt_sleep.evalkit import (
    main as evalkit_main,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "evalkit")


def _load(name: str):
    with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as f:
        return json.load(f)


class TestMcNemarKnownAnswer(unittest.TestCase):
    def test_textbook_2x2_chi2_and_exact(self):
        fx = _load("mcnemar_textbook.json")
        res = mcnemar_from_counts(
            fx["both_success"], fx["a_only"], fx["b_only"], fx["both_fail"],
        )
        self.assertAlmostEqual(res.chi2, fx["chi2"], places=12)
        self.assertAlmostEqual(res.p_chi2, fx["p_chi2"], places=12)
        self.assertAlmostEqual(res.p_exact, fx["p_exact"], places=12)
        self.assertTrue(res.significant)
        self.assertEqual(res.n, 100)

    def test_zero_discordants_is_not_significant(self):
        res = mcnemar_from_counts(20, 0, 0, 5)
        self.assertEqual(res.chi2, 0.0)
        self.assertEqual(res.p_chi2, 1.0)
        self.assertEqual(res.p_exact, 1.0)
        self.assertFalse(res.significant)

    def test_paired_vectors_match_counts(self):
        a = [1, 1, 1, 0, 0]
        b = [1, 0, 1, 1, 0]
        res = mcnemar_paired(a, b)
        self.assertEqual(res.both_success, 2)
        self.assertEqual(res.a_only, 1)
        self.assertEqual(res.b_only, 1)
        self.assertEqual(res.both_fail, 1)
        self.assertAlmostEqual(res.p_exact, exact_mcnemar_p(1, 1))

    def test_exact_tail_is_stable_above_one_thousand_discordants(self):
        self.assertEqual(exact_mcnemar_p(700, 700), 1.0)
        value = exact_mcnemar_p(590, 611)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_invalid_counts_and_binary_vectors_are_refused(self):
        for args in ((-1, 0), (True, 0), (1.5, 0)):
            with self.subTest(args=args), self.assertRaises(EvalkitError):
                exact_mcnemar_p(*args)
        with self.assertRaises(EvalkitError):
            mcnemar_from_counts(1, -1, 0, 1)
        with self.assertRaises(EvalkitError):
            mcnemar_paired([], [])
        with self.assertRaises(EvalkitError):
            mcnemar_paired([0, 2], [0, 1])


class TestBootstrapCoverage(unittest.TestCase):
    def test_identical_series_ci_collapses_to_zero(self):
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        ci = bootstrap_delta_ci(a, a, n_boot=2000, seed=7)
        self.assertEqual(ci.low, 0.0)
        self.assertEqual(ci.high, 0.0)
        self.assertEqual(ci.mean, 0.0)

    def test_known_shift_ci_excludes_zero(self):
        # A always 0, B always 1: delta = 1 exactly, CI is [1, 1].
        a = [0] * 30
        b = [1] * 30
        ci = bootstrap_delta_ci(a, b, n_boot=1000, seed=1)
        self.assertEqual(ci.low, 1.0)
        self.assertEqual(ci.high, 1.0)

    def test_seed_is_deterministic(self):
        a = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        b = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        x = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        y = bootstrap_delta_ci(a, b, n_boot=500, seed=99)
        self.assertEqual((x.low, x.high, x.mean), (y.low, y.high, y.mean))

    def test_invalid_alpha_and_bootstrap_counts_are_refused(self):
        for alpha in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(alpha=alpha), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], alpha=alpha)
        for n_boot in (0, -1, 1.5, True, 1_000_001):
            with self.subTest(n_boot=n_boot), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], n_boot=n_boot)
        for seed in (True, 1.5, "7"):
            with self.subTest(seed=seed), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [1], n_boot=10, seed=seed)

    def test_malformed_direct_api_scores_are_contract_errors(self):
        for value in ("not-a-number", None, object(), float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(EvalkitError):
                bootstrap_delta_ci([0], [value], n_boot=10)


class TestAACalibration(unittest.TestCase):
    def test_aa_does_not_reject(self):
        man = _load("aa_manifest.json")
        out = _load("aa_outcomes.json")
        report = compare_aa(man["ids"], out["outcomes"], n_boot=2000, seed=42)
        self.assertEqual(report.delta, 0.0)
        self.assertIsNotNone(report.mcnemar)
        self.assertFalse(report.mcnemar.significant)
        self.assertEqual(report.mcnemar.p_exact, 1.0)
        self.assertLessEqual(report.bootstrap.low, 0.0)
        self.assertGreaterEqual(report.bootstrap.high, 0.0)

    def test_exact_test_controls_type_one_error_under_a_seeded_null(self):
        # Unlike comparing an array with itself, this exercises non-zero,
        # symmetrically distributed discordance and can catch p-value inflation.
        rng = random.Random(20260824)
        trials = 500
        rejected = 0
        for _ in range(trials):
            a = []
            b = []
            for _task in range(80):
                left = int(rng.random() < 0.5)
                right = 1 - left if rng.random() < 0.30 else left
                a.append(left)
                b.append(right)
            rejected += int(mcnemar_paired(a, b, alpha=0.05).significant)
        rate = rejected / trials
        self.assertGreater(rejected, 0)
        self.assertLessEqual(rate, 0.075)


class TestCompareContracts(unittest.TestCase):
    def test_mismatched_ids_are_refused(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 1, "t2": 0}, {"t1": 1, "t3": 0})
        self.assertIn("must equal the manifest", str(ctx.exception))

    def test_duplicate_manifest_ids_refused(self):
        with self.assertRaises(EvalkitError):
            compare(["t1", "t1"], {"t1": 1}, {"t1": 0})

    def test_empty_manifest_refused(self):
        with self.assertRaises(EvalkitError):
            compare([], {}, {})

    def test_graded_refused_without_flag(self):
        with self.assertRaises(EvalkitError) as ctx:
            compare(["t1", "t2"], {"t1": 0.4, "t2": 0.9}, {"t1": 0.5, "t2": 0.8})
        self.assertIn("allow-graded", str(ctx.exception))

    def test_graded_bootstrap_only(self):
        report = compare(
            ["t1", "t2"],
            {"t1": 0.4, "t2": 0.9},
            {"t1": 0.5, "t2": 0.8},
            allow_graded=True,
            n_boot=500,
            seed=3,
        )
        self.assertIsNone(report.mcnemar)
        self.assertTrue(any("graded" in n for n in report.notes))
        self.assertAlmostEqual(report.delta, 0.0, places=12)

    def test_multi_seed_variance_band(self):
        report = compare(
            ["t1", "t2"],
            {"t1": [1, 0, 1], "t2": [0, 0, 1]},
            {"t1": [1, 1, 1], "t2": [1, 0, 1]},
            n_boot=400,
            seed=2,
        )
        self.assertEqual(len(report.per_seed), 3)
        self.assertIsNotNone(report.seed_mean_delta)
        self.assertGreaterEqual(report.seed_sd_delta, 0.0)
        self.assertAlmostEqual(report.rate_a, (2 / 3 + 1 / 3) / 2)
        self.assertAlmostEqual(report.rate_b, (1.0 + 2 / 3) / 2)
        self.assertIsNone(report.mcnemar)
        self.assertTrue(any("task-cluster" in note for note in report.notes))

    def test_empty_nonfinite_and_out_of_range_seed_scores_are_refused(self):
        bad_values = ([], [float("nan")], [float("inf")], [-0.01], [1.01])
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": value}, {"t1": [1]}, allow_graded=True)

    def test_duplicating_seeds_within_tasks_does_not_inflate_inference(self):
        ids = ["t1", "t2", "t3", "t4"]
        a_two = {tid: [0, 0] for tid in ids}
        b_two = {tid: [1, 1] for tid in ids}
        a_many = {tid: [0] * 100 for tid in ids}
        b_many = {tid: [1] * 100 for tid in ids}
        two = compare(ids, a_two, b_two, n_boot=500, seed=8)
        many = compare(ids, a_many, b_many, n_boot=500, seed=8)
        self.assertIsNone(two.mcnemar)
        self.assertIsNone(many.mcnemar)
        self.assertEqual(two.delta, many.delta)
        self.assertEqual(two.bootstrap.to_dict(), many.bootstrap.to_dict())

    def test_invalid_compare_parameters_are_refused(self):
        for alpha in (0, 1, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": 0}, {"t1": 1}, alpha=alpha)
        for n_boot in (0, -4, True):
            with self.subTest(n_boot=n_boot), self.assertRaises(EvalkitError):
                compare(["t1"], {"t1": 0}, {"t1": 1}, n_boot=n_boot)


class TestResultsCellReplay(unittest.TestCase):
    def test_malformed_reconstruction_inputs_are_contract_errors(self):
        for rates in (("bad", 0.5), (None, 0.5), (float("nan"), 0.5), (0.5, 1.1)):
            with self.subTest(rates=rates), self.assertRaises(EvalkitError):
                reconstruct_paired_from_rates(10, *rates)
        for n in (True, 0, 1.5):
            with self.subTest(n=n), self.assertRaises(EvalkitError):
                reconstruct_paired_from_rates(n, 0.5, 0.5)

    def test_published_searchqa_nano_gated_delta(self):
        cell = _load("results_searchqa_nano_gated.json")
        a, b = reconstruct_paired_from_rates(cell["n"], cell["baseline"], cell["after"])
        self.assertEqual(len(a), cell["n"])
        self.assertAlmostEqual(sum(a) / cell["n"], cell["baseline"], places=3)
        self.assertAlmostEqual(sum(b) / cell["n"], cell["after"], places=3)
        ids = [f"q{i:04d}" for i in range(cell["n"])]
        report = compare(
            ids,
            dict(zip(ids, a)),
            dict(zip(ids, b)),
            n_boot=800,
            seed=42,
        )
        self.assertAlmostEqual(report.delta, cell["published_delta"], places=3)
        self.assertGreater(report.bootstrap.low, 0.0)
        self.assertTrue(report.mcnemar.significant)
        md = format_markdown(report)
        self.assertIn("delta (B-A)", md)
        self.assertIn("McNemar", md)


class TestCLI(unittest.TestCase):
    def test_aa_cli_exit_zero(self):
        rc = evalkit_main([
            "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
            "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
            "--aa",
            "--boot", "300",
            "--json",
        ])
        self.assertEqual(rc, 0)

    def test_mismatch_cli_exit_two(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as f:
                json.dump(["t1", "t2"], f)
            with open(a, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t2": 0}, f)
            with open(b, "w", encoding="utf-8") as f:
                json.dump({"t1": 1, "t3": 0}, f)
            rc = evalkit_main(["--manifest", man, "--a", a, "--b", b])
            self.assertEqual(rc, 2)

    def test_malformed_json_and_shapes_are_clean_contract_errors(self):
        cases = (
            ("{", '{"t1": 1}', '{"t1": 1}'),
            ('{"ids": "t1"}', '{"t1": 1}', '{"t1": 1}'),
            ('["t1"]', '{"outcomes": []}', '{"t1": 1}'),
        )
        for manifest_text, a_text, b_text in cases:
            with self.subTest(manifest=manifest_text), tempfile.TemporaryDirectory() as td:
                paths = []
                for name, content in (("m.json", manifest_text), ("a.json", a_text), ("b.json", b_text)):
                    path = os.path.join(td, name)
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(content)
                    paths.append(path)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    rc = evalkit_main(["--manifest", paths[0], "--a", paths[1], "--b", paths[2]])
                self.assertEqual(rc, 2)
                self.assertTrue(stderr.getvalue().startswith("ERR_EVALKIT "))
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_module_entrypoint(self):
        proc = subprocess.run(
            [
                sys.executable, "-m", "skillopt_sleep.evalkit",
                "--manifest", os.path.join(FIXTURE_DIR, "aa_manifest.json"),
                "--a", os.path.join(FIXTURE_DIR, "aa_outcomes.json"),
                "--aa",
                "--boot", "200",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("delta (B-A): +0.000000", proc.stdout)

    def test_nonfinite_input_is_refused_without_nonstandard_json_output(self):
        with tempfile.TemporaryDirectory() as td:
            man = os.path.join(td, "m.json")
            a = os.path.join(td, "a.json")
            b = os.path.join(td, "b.json")
            with open(man, "w", encoding="utf-8") as f:
                json.dump(["t1"], f)
            with open(a, "w", encoding="utf-8") as f:
                f.write('{"t1": NaN}')
            with open(b, "w", encoding="utf-8") as f:
                json.dump({"t1": 1}, f)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = evalkit_main([
                    "--manifest", man, "--a", a, "--b", b, "--json",
                ])
            self.assertEqual(rc, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("finite", stderr.getvalue())
            self.assertNotIn("NaN", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
