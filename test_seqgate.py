import random
import unittest

from seqgate import Gate, run, spend, binom_tail, ACCEPT, REJECT, CONTINUE


class GateTests(unittest.TestCase):
    def test_all_wins_accepts(self):
        g = run([(0.0, 1.0)] * 40, alpha=0.05, max_n=40)
        self.assertEqual(g.decision, ACCEPT)
        self.assertEqual(g.reason, "evidence")
        self.assertLess(g.n, 12)
        self.assertGreaterEqual(g.wealth, 20.0)

    def test_all_losses_rejected_early_as_futile(self):
        g = run([(1.0, 0.0)] * 40, alpha=0.05, max_n=40)
        self.assertEqual(g.decision, REJECT)
        self.assertEqual(g.reason, "futile")
        self.assertLess(g.n, 40)

    def test_wealth_never_negative_and_losses_never_raise_it(self):
        rng = random.Random(1)
        g = Gate(max_n=500)
        for _ in range(500):
            before = g.wealth
            won = rng.random() < 0.5
            g.update(0.0, 1.0 if won else -1.0)
            self.assertGreater(g.wealth, 0.0)
            if not won:
                self.assertLessEqual(g.wealth, before)
            if g.decision != CONTINUE:
                break

    def test_ties_do_not_move_wealth(self):
        g = Gate()
        g.update(0.0, 1.0)
        g.update(0.0, 1.0)
        w = g.wealth
        g.update(2.0, 2.0)
        self.assertEqual(g.wealth, w)
        self.assertEqual(g.ties, 1)

    def test_lower_is_better(self):
        g = run([(1.0, 0.5)] * 40, higher_is_better=False)
        self.assertEqual(g.decision, ACCEPT)

    def test_decision_is_final(self):
        g = run([(0.0, 1.0)] * 40)
        n = g.n
        g.update(1.0, 0.0)
        self.assertEqual(g.n, n)
        self.assertEqual(g.decision, ACCEPT)

    def test_budget_is_respected(self):
        rng = random.Random(2)
        for _ in range(300):
            g = run(((0.0, rng.gauss(0.2, 1)) for _ in range(100)), max_n=25)
            self.assertLessEqual(g.n, 25)
            self.assertIn(g.decision, (ACCEPT, REJECT))

    def test_out_of_data_rejects(self):
        g = run([(0.0, 1.0)] * 2)
        self.assertEqual((g.decision, g.reason), (REJECT, "out of data"))

    def test_futility_stop_never_changes_the_answer(self):
        # A gate with the futility stop must accept exactly the same sequences
        # as one that keeps going to the end of the budget.
        class NoFutility(Gate):
            def _best_case(self):
                return float("inf")
        rng = random.Random(3)
        saved = 0
        for _ in range(2000):
            p = rng.choice([0.4, 0.5, 0.6, 0.75])
            seq = [(0.0, 1.0 if rng.random() < p else -1.0) for _ in range(40)]
            a = run(seq, max_n=40)
            c = NoFutility(max_n=40)
            for x, y in seq:
                if c.update(x, y) != CONTINUE:
                    break
            self.assertEqual(a.decision, c.decision)
            self.assertLessEqual(a.n, c.n)
            saved += c.n - a.n
        self.assertGreater(saved, 0)

    def test_false_accept_rate_under_null_is_below_alpha(self):
        rng = random.Random(4)
        trials, accepts = 20000, 0
        for _ in range(trials):
            g = Gate(alpha=0.05, max_n=40)
            while g.update(0.0, 1.0 if rng.random() < 0.5 else -1.0) == CONTINUE:
                pass
            accepts += g.decision == ACCEPT
        self.assertLess(accepts / trials, 0.05)

    def test_false_accept_rate_holds_with_a_long_budget(self):
        # Peeking for 1000 seeds would break a fixed-sample test.
        rng = random.Random(5)
        trials, accepts = 3000, 0
        for _ in range(trials):
            g = Gate(alpha=0.05, max_n=1000)
            while g.update(0.0, 1.0 if rng.random() < 0.5 else -1.0) == CONTINUE:
                pass
            accepts += g.decision == ACCEPT
        self.assertLess(accepts / trials, 0.05)

    def test_spend_sums_to_alpha(self):
        total = sum(spend(0.05, k) for k in range(1, 200000))
        self.assertLess(total, 0.05)
        self.assertGreater(total, 0.0499)
        with self.assertRaises(ValueError):
            spend(0.05, 0)

    def test_binom_tail(self):
        self.assertAlmostEqual(binom_tail(0, 10), 1.0)
        self.assertAlmostEqual(binom_tail(10, 10), 1 / 1024)
        self.assertAlmostEqual(binom_tail(5, 9), 0.5)

    def test_bad_arguments(self):
        for kw in ({"alpha": 0}, {"alpha": 1}, {"lam_max": 1.0}, {"max_n": 0}):
            with self.assertRaises(ValueError):
                Gate(**kw)


if __name__ == "__main__":
    unittest.main()
