"""seqgate: decide whether a candidate really beats a baseline, one paired seed at a time.

The test is a betting e-process on the sign of the paired difference. Under the
null hypothesis (the candidate is not better, so each non-tied seed is a win
with probability at most 1/2) the wealth is a nonnegative supermartingale, so by
Ville's inequality the chance it ever reaches 1/alpha is at most alpha. That
holds no matter how often you look or when you decide to stop.
"""
import json
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
        diff = candidate - baseline
        if not self.higher_is_better:
            diff = -diff
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


def run(pairs, **kwargs):
    """Run a Gate over an iterable of (baseline, candidate) pairs, pulling lazily."""
    gate = Gate(**kwargs)
    for b, c in pairs:
        if gate.update(b, c) != CONTINUE:
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


def _main(argv):
    if len(argv) < 2:
        print("usage: python seqgate.py scores.csv [alpha] [max_n]")
        print("scores.csv has one seed per line: baseline_score,candidate_score")
        return 2
    alpha = float(argv[2]) if len(argv) > 2 else 0.05
    max_n = int(argv[3]) if len(argv) > 3 else 40
    pairs = []
    with open(argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split(",")[:2]
            pairs.append((float(a), float(b)))
    gate = run(pairs, alpha=alpha, max_n=max_n)
    print(json.dumps(gate.summary(), indent=2))
    return 0 if gate.decision == ACCEPT else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
