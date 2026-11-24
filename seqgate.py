"""seqgate: decide whether a candidate really beats a baseline, one paired seed at a time.

The test is a betting e-process on the sign of the paired difference. Under the
null hypothesis (the candidate is not better, so each non-tied seed is a win
with probability at most 1/2) the wealth is a nonnegative supermartingale, so by
Ville's inequality the chance it ever reaches 1/alpha is at most alpha. That
holds no matter how often you look or when you decide to stop.

Gate      uses only the sign of each paired difference (no bound needed).
MeanGate  uses the size of the difference too. It needs a known bound on the
          difference (for example 1.0 for a per-query Recall@10) and tests
          "the mean difference is at most zero".
"""
import json
import math
import sys

ACCEPT = "accept"
REJECT = "reject"
CONTINUE = "continue"


class Gate:
    """Sequential accept/reject decision for one candidate against one baseline.

    alpha    bound on the probability of accepting a candidate that is not better
    max_n    seed budget; undecided candidates are rejected when it runs out
    lam_max  cap on the fraction of wealth bet on each seed (must be < 1)
    higher_is_better  set False for losses
    """

    def __init__(self, alpha=0.05, max_n=40, lam_max=0.9, higher_is_better=True):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < lam_max < 1:
            raise ValueError("lam_max must be in (0, 1)")
        if max_n < 1:
            raise ValueError("max_n must be at least 1")
        self.alpha = alpha
        self.max_n = max_n
        self.lam_max = lam_max
        self.higher_is_better = higher_is_better
        self.threshold = 1.0 / alpha
        self.wealth = 1.0
        self.n = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.decision = CONTINUE
        self.reason = ""

    def _bet(self):
        # Chosen from past seeds only, which is what keeps the test valid.
        # Kelly bet for a coin with win rate p against 1/2 is 2p - 1; p is
        # estimated with one pseudo-win and one pseudo-loss.
        p = (self.wins + 1.0) / (self.wins + self.losses + 2.0)
        return min(self.lam_max, max(0.0, 2.0 * p - 1.0))

    def _best_case(self):
        """Largest wealth still reachable if every remaining seed is a win."""
        w, wins, losses = self.wealth, self.wins, self.losses
        for _ in range(self.max_n - self.n):
            p = (wins + 1.0) / (wins + losses + 2.0)
            w *= 1.0 + min(self.lam_max, max(0.0, 2.0 * p - 1.0))
            wins += 1
            if w >= self.threshold:
                break
        return w

    def update(self, baseline, candidate):
        """Feed the two scores from one shared seed. Returns the decision so far."""
        if self.decision != CONTINUE:
            return self.decision
        diff = _checked(candidate) - _checked(baseline)
        if not self.higher_is_better:
            diff = -diff
        return self._step(diff)

    def update_failed(self):
        """The candidate run crashed or timed out on this unit: count it as a loss."""
        if self.decision != CONTINUE:
            return self.decision
        return self._step(-1.0)

    def _step(self, diff):
        lam = self._bet()
        self.n += 1
        if diff > 0:
            self.wealth *= 1.0 + lam
            self.wins += 1
        elif diff < 0:
            self.wealth *= 1.0 - lam
            self.losses += 1
        else:
            self.ties += 1
        if self.wealth >= self.threshold:
            self.decision, self.reason = ACCEPT, "evidence"
        elif self.n >= self.max_n:
            self.decision, self.reason = REJECT, "budget"
        elif self._best_case() < self.threshold:
            # Cannot be accepted even if every remaining seed wins, so the
            # remaining seeds would be wasted compute. This costs no power.
            self.decision, self.reason = REJECT, "futile"
        return self.decision

    def summary(self):
        return {
            "decision": self.decision,
            "reason": self.reason,
            "seeds_used": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "e_value": round(self.wealth, 4),
            "alpha": self.alpha,
        }


def _checked(x):
    """Scores must be finite numbers. NaN would otherwise be silently read as a tie."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("score is not finite: %r" % x)
    return x


class MeanGate:
    """Sequential test of "mean paired difference <= 0" for bounded differences.

    Each unit (a query, a seed, a test case) gives d = candidate - baseline with
    |d| <= bound. With x = d / bound in [-1, 1] and a bet lam in [0, 1) chosen
    before seeing x, the wealth is multiplied by 1 + lam * x. If the mean of x
    is at most 0 the expected multiplier is at most 1, so the wealth is a
    nonnegative supermartingale and P(wealth ever >= 1/alpha) <= alpha.

    alpha       bound on the probability of accepting a candidate whose mean
                difference is not positive
    max_n       unit budget; undecided candidates are rejected when it runs out
    bound       largest possible |candidate - baseline| on one unit. A value
                outside the bound raises, because the guarantee would be gone.
    lam_max     cap on the bet (must be < 1 so the wealth stays positive)
    prior_var   variance the bet assumes before it has data (in units of bound^2)
    prior_n     how many pseudo-observations that prior is worth
    harm_alpha  if set, a mirrored test of "mean >= 0" runs alongside and stops
                the candidate early as "worse" once it has that much evidence of
                harm. It can only remove accepts, so it cannot raise the false
                accept rate, and it rejects a truly better candidate with
                probability at most harm_alpha.
    """

    def __init__(self, alpha=0.05, max_n=1000, bound=1.0, lam_max=0.9,
                 prior_var=0.25, prior_n=5.0, harm_alpha=0.05, higher_is_better=True):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < lam_max < 1:
            raise ValueError("lam_max must be in (0, 1)")
        if max_n < 1:
            raise ValueError("max_n must be at least 1")
        if not bound > 0 or math.isinf(bound):
            raise ValueError("bound must be a positive finite number")
        if prior_var <= 0 or prior_n <= 0:
            raise ValueError("prior_var and prior_n must be positive")
        if harm_alpha is not None and not 0 < harm_alpha < 1:
            raise ValueError("harm_alpha must be in (0, 1) or None")
        self.alpha = alpha
        self.max_n = max_n
        self.bound = float(bound)
        self.lam_max = lam_max
        self.prior_var = prior_var
        self.prior_n = prior_n
        self.harm_alpha = harm_alpha
        self.higher_is_better = higher_is_better
        self.threshold = 1.0 / alpha
        self.wealth = 1.0
        self.harm_wealth = 1.0
        self.n = 0
        self.failures = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.decision = CONTINUE
        self.reason = ""

    def _bets(self):
        # Chosen from past units only. lam = m / (v + m^2) is the approximate
        # growth-optimal bet for mean m and variance v; the prior shrinks the
        # mean towards 0 and keeps the variance away from 0 on the first units.
        k = self.n + self.prior_n
        m = self._sum / k
        second = (self._sumsq + self.prior_n * self.prior_var) / k
        lam = m / second if second > 0 else 0.0
        up = min(self.lam_max, max(0.0, lam))
        down = min(self.lam_max, max(0.0, -lam))
        return up, down

    def update(self, baseline, candidate):
        """Feed the two scores from one shared unit. Returns the decision so far."""
        if self.decision != CONTINUE:
            return self.decision
        diff = _checked(candidate) - _checked(baseline)
        if not self.higher_is_better:
            diff = -diff
        x = diff / self.bound
        if abs(x) > 1.0 + 1e-12:
            raise ValueError("difference %r is outside the declared bound %r"
                             % (diff, self.bound))
        return self._step(max(-1.0, min(1.0, x)))

    def update_failed(self):
        """The candidate run crashed or timed out on this unit: worst case, -bound."""
        if self.decision != CONTINUE:
            return self.decision
        self.failures += 1
        return self._step(-1.0)

    def _step(self, x):
        up, down = self._bets()
        self.n += 1
        self.wealth *= 1.0 + up * x
        self.harm_wealth *= 1.0 - down * x
        self._sum += x
        self._sumsq += x * x
        if self.wealth >= self.threshold:
            self.decision, self.reason = ACCEPT, "evidence"
        elif self.harm_alpha is not None and self.harm_wealth >= 1.0 / self.harm_alpha:
            self.decision, self.reason = REJECT, "worse"
        elif self.n >= self.max_n:
            self.decision, self.reason = REJECT, "budget"
        elif self.wealth * (1.0 + self.lam_max) ** (self.max_n - self.n) < self.threshold:
            # Upper bound on the reachable wealth, so this never changes an answer.
            self.decision, self.reason = REJECT, "futile"
        return self.decision

    def summary(self):
        return {
            "decision": self.decision,
            "reason": self.reason,
            "units_used": self.n,
            "mean_diff": round(self.bound * self._sum / self.n, 6) if self.n else None,
            "failures": self.failures,
            "e_value": round(self.wealth, 4),
            "harm_e_value": round(self.harm_wealth, 4),
            "alpha": self.alpha,
        }


def run(pairs, gate=None, **kwargs):
    """Run a gate over an iterable of (baseline, candidate) pairs, pulling lazily.

    A pair whose candidate score is None is treated as a failed candidate run.
    """
    if gate is None:
        gate = Gate(**kwargs)
    for b, c in pairs:
        d = gate.update_failed() if c is None else gate.update(b, c)
        if d != CONTINUE:
            break
    if gate.decision == CONTINUE:
        gate.decision, gate.reason = REJECT, "out of data"
    return gate


def spend(alpha, k):
    """Level for the k-th candidate (k = 1, 2, ...) so the levels sum to alpha.

    Using Gate(alpha=spend(0.05, k)) for the k-th candidate bounds the chance of
    any false accept over an unbounded search at 0.05 (union bound).
    """
    if k < 1:
        raise ValueError("k starts at 1")
    return alpha * 6.0 / (3.141592653589793 ** 2 * k * k)


def binom_tail(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2). Used by the baselines in bench.py."""
    total, c = 0, 1
    for i in range(n + 1):
        if i >= wins:
            total += c
        c = c * (n - i) // (i + 1)
    return total / float(2 ** n)


def read_pairs(path):
    """Read baseline,candidate lines. An empty or "fail" candidate is a failed run."""
    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 2:
                raise ValueError("%s line %d: expected baseline,candidate" % (path, lineno))
            try:
                b = _checked(cells[0])
                c = None if cells[1].lower() in ("", "fail", "failed", "error") else _checked(cells[1])
            except ValueError as e:
                raise ValueError("%s line %d: %s" % (path, lineno, e))
            pairs.append((b, c))
    return pairs


def _main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "") for a in argv[1:] if a.startswith("--"))
    if not args:
        print("usage: python seqgate.py scores.csv [alpha] [max_n] [--mean] [--bound=1.0]")
        print("scores.csv has one unit per line: baseline_score,candidate_score")
        print("--mean uses the size of each difference and needs --bound (default 1.0)")
        return 2
    alpha = float(args[1]) if len(args) > 1 else 0.05
    max_n = int(args[2]) if len(args) > 2 else 40
    try:
        pairs = read_pairs(args[0])
        if "mean" in opts:
            gate = MeanGate(alpha=alpha, max_n=max_n, bound=float(opts.get("bound", 1.0)))
        else:
            gate = Gate(alpha=alpha, max_n=max_n)
        run(pairs, gate=gate)
    except (ValueError, OSError) as e:
        print("error: %s" % e)
        return 2
    print(json.dumps(gate.summary(), indent=2))
    return 0 if gate.decision == ACCEPT else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
