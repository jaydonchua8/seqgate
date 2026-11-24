import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

from search import Ledger, gamma
from seqgate import ACCEPT, CONTINUE, REJECT, Gate, MeanGate, read_pairs, run

HERE = os.path.dirname(os.path.abspath(__file__))


def bern(rng, p_win, size=0.3):
    return size if rng.random() < p_win else -size


class MeanGateTests(unittest.TestCase):
    def test_clear_improvement_accepts(self):
        g = run([(0.5, 0.7)] * 200, gate=MeanGate(max_n=200))
        self.assertEqual((g.decision, g.reason), (ACCEPT, "evidence"))
        self.assertGreaterEqual(g.wealth, 20.0)

    def test_clear_harm_stops_early_as_worse(self):
        g = run([(0.7, 0.5)] * 500, gate=MeanGate(max_n=500))
        self.assertEqual((g.decision, g.reason), (REJECT, "worse"))
        self.assertLess(g.n, 100)

    def test_harm_stop_can_be_turned_off(self):
        g = run([(0.7, 0.5)] * 60, gate=MeanGate(max_n=60, harm_alpha=None))
        self.assertEqual(g.decision, REJECT)
        self.assertIn(g.reason, ("budget", "futile"))

    def test_uses_size_not_just_sign(self):
        # Wins on 4 of 10 units, but the wins are large and the losses small.
        # The mean is positive, so MeanGate accepts and the sign Gate does not.
        seq = ([(0.0, 0.9)] * 4 + [(0.0, -0.05)] * 6) * 30
        self.assertEqual(run(seq, gate=MeanGate(max_n=300)).decision, ACCEPT)
        self.assertEqual(run(seq, gate=Gate(max_n=300)).decision, REJECT)

    def test_out_of_bound_difference_raises(self):
        g = MeanGate(bound=0.1)
        with self.assertRaises(ValueError):
            g.update(0.0, 0.5)
        self.assertEqual(g.n, 0)

    def test_non_finite_scores_raise_in_both_gates(self):
        for g in (Gate(), MeanGate()):
            for bad in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    g.update(0.0, bad)
                with self.assertRaises(ValueError):
                    g.update(bad, 0.0)
            self.assertEqual(g.n, 0)

    def test_failed_run_counts_as_worst_case(self):
        g = MeanGate(max_n=50)
        g.update(0.0, 0.2)
        g.update(0.0, 0.2)
        before = g.wealth
        g.update_failed()
        self.assertLess(g.wealth, before)
        self.assertEqual(g.failures, 1)
        s = Gate()
        s.update(0.0, 1.0)
        s.update(0.0, 1.0)
        s.update_failed()
        self.assertEqual(s.losses, 1)

    def test_candidate_that_wins_but_crashes_often_is_not_accepted(self):
        # +0.05 when it runs, crashes on 1 unit in 10: mean is 0.9*0.05 - 0.1 < 0.
        seq = ([(0.5, 0.55)] * 9 + [(0.5, None)]) * 100
        g = run(seq, gate=MeanGate(max_n=1000))
        self.assertEqual(g.decision, REJECT)

    def test_wealth_stays_positive_and_decision_is_final(self):
        rng = random.Random(1)
        g = MeanGate(max_n=400, lam_max=0.9, harm_alpha=None)
        for _ in range(400):
            g.update(0.0, rng.choice((-1.0, 1.0)))
            self.assertGreater(g.wealth, 0.0)
            self.assertGreater(g.harm_wealth, 0.0)
        n = g.n
        g.update(0.0, 1.0)
        self.assertEqual(g.n, n)

    def test_bound_scales_the_test(self):
        a = run([(0.0, 0.02)] * 100, gate=MeanGate(max_n=100, bound=0.02))
        b = run([(0.0, 1.0)] * 100, gate=MeanGate(max_n=100, bound=1.0))
        self.assertEqual(a.n, b.n)
        self.assertAlmostEqual(a.wealth, b.wealth)

    def test_lower_is_better(self):
        g = run([(0.7, 0.5)] * 200, gate=MeanGate(max_n=200, higher_is_better=False))
        self.assertEqual(g.decision, ACCEPT)

    def test_futility_bound_never_changes_the_answer(self):
        class NoFutility(MeanGate):
            def _step(self, x):
                d = MeanGate._step(self, x)
                if self.reason == "futile":
                    self.decision, self.reason = CONTINUE, ""
                    if self.n >= self.max_n:
                        self.decision, self.reason = REJECT, "budget"
                return self.decision
        rng = random.Random(3)
        for _ in range(1500):
            p = rng.choice((0.45, 0.5, 0.6, 0.7))
            seq = [(0.0, bern(rng, p)) for _ in range(60)]
            a = run(seq, gate=MeanGate(max_n=60, harm_alpha=None))
            c = run(seq, gate=NoFutility(max_n=60, harm_alpha=None))
            self.assertEqual(a.decision, c.decision)
            self.assertLessEqual(a.n, c.n)

    def test_false_accept_rate_at_the_boundary_is_below_alpha(self):
        # Mean exactly zero, checked after every unit for 300 units, three shapes.
        rng = random.Random(4)
        shapes = [lambda: rng.choice((-0.3, 0.3)),
                  lambda: rng.choice((-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
                  lambda: 0.9 if rng.random() < 0.1 else -0.1]
        for draw in shapes:
            accepts, trials = 0, 3000
            for _ in range(trials):
                g = MeanGate(alpha=0.05, max_n=300)
                while g.update(0.0, draw()) == CONTINUE:
                    pass
                accepts += g.decision == ACCEPT
            self.assertLess(accepts / trials, 0.05)

    def test_harm_stop_rarely_kills_a_real_improvement(self):
        rng = random.Random(5)
        worse, trials = 0, 2000
        for _ in range(trials):
            g = MeanGate(max_n=300)
            while g.update(0.0, bern(rng, 0.55)) == CONTINUE:
                pass
            worse += g.reason == "worse"
        self.assertLess(worse / trials, 0.05)

    def test_bad_arguments(self):
        for kw in ({"alpha": 0}, {"lam_max": 1.0}, {"max_n": 0}, {"bound": 0}, {"bound": float("inf")},
                   {"prior_var": 0}, {"prior_n": 0}, {"harm_alpha": 1.0}):
            with self.assertRaises(ValueError):
                MeanGate(**kw)


class LedgerTests(unittest.TestCase):
    def test_gammas_sum_to_at_most_one(self):
        for series in ("log", "square"):
            total = sum(gamma(k, series) for k in range(1, 300000))
            self.assertLess(total, 1.0)
            self.assertGreater(total, 0.9)
        # the log series leaves an infinite tail; partial sum plus integral bound
        n = 300000
        total = sum(gamma(k, "log") for k in range(1, n))
        tail_bound = 1.0 / math.log(n) / sum(1.0 / (m * math.log(m) ** 2) for m in range(2, 10001))
        self.assertLess(total + tail_bound, 1.01)

    def test_levels(self):
        fixed = Ledger(0.05, "fixed")
        spend = Ledger(0.05, "spend")
        elond = Ledger(0.05, "elond")
        for led in (fixed, spend, elond):
            a1 = led.next_alpha()
            led.record(True)
            a2 = led.next_alpha()
            led.record(False)
            self.assertLessEqual(a2, 0.05)
            self.assertGreater(a2, 0.0)
            if led is fixed:
                self.assertEqual((a1, a2), (0.05, 0.05))
        self.assertAlmostEqual(spend.alpha * gamma(1), 0.05 * gamma(1))
        # after one accept, e-LOND tests the second candidate at twice spend's level
        s, e = Ledger(0.05, "spend"), Ledger(0.05, "elond")
        for led in (s, e):
            led.next_alpha()
            led.record(True)
        self.assertAlmostEqual(e.next_alpha(), 2 * s.next_alpha())

    def test_protocol_errors(self):
        led = Ledger()
        with self.assertRaises(RuntimeError):
            led.record(True)
        led.next_alpha()
        with self.assertRaises(RuntimeError):
            led.next_alpha()
        with self.assertRaises(ValueError):
            Ledger(mode="bonferroni")
        with self.assertRaises(ValueError):
            Ledger(series="cubic")
        with self.assertRaises(ValueError):
            gamma(0)

    def _search(self, mode, rng, n_cand=40, frac_real=0.1, max_n=200):
        led = Ledger(0.1, mode)
        acc = false = 0
        for _ in range(n_cand):
            real = rng.random() < frac_real
            g = MeanGate(alpha=led.next_alpha(), max_n=max_n, harm_alpha=None)
            while g.update(0.0, bern(rng, 0.9 if real else 0.5, 0.5)) == CONTINUE:
                pass
            ok = g.decision == ACCEPT
            led.record(ok)
            acc += ok
            false += ok and not real
        return acc, false

    def test_elond_holds_fdr_where_a_fixed_level_does_not(self):
        out = {}
        for mode in ("fixed", "elond", "spend"):
            rng = random.Random(6)
            fdp = anyfalse = found = 0.0
            reps = 150
            for _ in range(reps):
                acc, false = self._search(mode, rng)
                fdp += false / acc if acc else 0.0
                anyfalse += false > 0
                found += acc - false
            out[mode] = (fdp / reps, anyfalse / reps, found / reps)
        self.assertGreater(out["fixed"][0], 0.1)     # nominal 10% per candidate is not 10% FDR
        self.assertLess(out["elond"][0], 0.1)
        self.assertLess(out["spend"][1], 0.1)        # union bound: any false accept

    def test_elond_level_is_never_below_spend(self):
        rng = random.Random(9)
        s, e = Ledger(0.05, "spend"), Ledger(0.05, "elond")
        for _ in range(500):
            a_s, a_e = s.next_alpha(), e.next_alpha()
            self.assertGreaterEqual(a_e, a_s)
            self.assertLessEqual(a_e, 0.05)
            ok = rng.random() < 0.2
            s.record(ok)
            e.record(ok)


class CliTests(unittest.TestCase):
    def _cli(self, *args):
        p = subprocess.run([sys.executable, os.path.join(HERE, "seqgate.py")] + list(args),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return p.returncode, p.stdout + p.stderr

    def _tmp(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_example_file(self):
        code, out = self._cli(os.path.join(HERE, "example_scores.csv"))
        self.assertEqual(code, 0)
        self.assertIn('"decision": "accept"', out)

    def test_mean_mode_and_failed_rows(self):
        path = self._tmp("# b,c\n" + "0.5,0.7\n" * 60 + "0.5,fail\n")
        self.assertEqual(read_pairs(path)[-1], (0.5, None))
        code, out = self._cli(path, "0.05", "200", "--mean")
        self.assertEqual(code, 0)
        self.assertIn('"units_used"', out)

    def test_malformed_input_is_reported_not_crashed(self):
        for text in ("0.5\n", "0.5,abc\n", "nan,0.5\n"):
            code, out = self._cli(self._tmp(text))
            self.assertEqual(code, 2)
            self.assertIn("line 1", out)
            self.assertNotIn("Traceback", out)
        code, out = self._cli(self._tmp("0.0,0.9\n"), "0.05", "40", "--mean", "--bound=0.1")
        self.assertEqual(code, 2)
        self.assertIn("outside the declared bound", out)
        code, out = self._cli("/nonexistent/scores.csv")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
