"""Error budget for a whole search, not just one candidate.

A 5% level per candidate says nothing about the share of accepted candidates
that are wrong when most candidates are not improvements. Ledger hands out the
level for each new candidate so that a guarantee holds over the whole stream.

  fixed   every candidate gets alpha. No search-level guarantee.
  spend   candidate k gets alpha * gamma_k with sum(gamma) <= 1. Union bound:
          P(any false accept in the whole search) <= alpha.
  elond   candidate k gets alpha * gamma_k * (accepts so far + 1). This is the
          e-LOND rule of Xu and Ramdas (arXiv 2311.06412). It bounds the false
          discovery rate (expected share of accepts that are not improvements)
          at alpha, and needs only that each candidate's stopped e-value has
          expectation <= 1 when that candidate is not an improvement.

Why elond works, in one line: if candidate k is accepted then the final number
of accepts R is at least D + 1 (D = accepts before k), and accepting means
e_k * alpha * gamma_k * (D + 1) >= 1, so 1{k accepted} / R <= e_k * alpha *
gamma_k. Summing the expectation over the non-improvements gives at most alpha.
"""
import math

MODES = ("fixed", "spend", "elond")

_N = 10000
# sum over k >= 1 of 1 / ((k + 1) ln^2(k + 1)), bounded above by a partial sum
# plus the integral of the tail, so the gammas are guaranteed to sum to <= 1.
_LOG_NORM = sum(1.0 / (n * math.log(n) ** 2) for n in range(2, _N + 1)) + 1.0 / math.log(_N)


def gamma(k, series="log"):
    """Weight of the k-th candidate (k = 1, 2, ...). Sums to at most 1 over all k.

    "log" decays like 1 / (k ln^2 k), which keeps late candidates testable in a
    long search. "square" is 6 / (pi^2 k^2), the series the first version used.
    """
    if k < 1:
        raise ValueError("k starts at 1")
    if series == "log":
        return 1.0 / ((k + 1) * math.log(k + 1) ** 2) / _LOG_NORM
    if series == "square":
        return 6.0 / (math.pi ** 2 * k * k)
    raise ValueError("unknown series %r" % series)


class Ledger:
    """Tracks a search and gives the level to test the next candidate at.

        ledger = Ledger(alpha=0.05, mode="elond")
        gate = MeanGate(alpha=ledger.next_alpha(), ...)
        ... feed the gate ...
        ledger.record(gate.decision == "accept")
    """

    def __init__(self, alpha=0.05, mode="elond", series="log"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        gamma(1, series)
        self.alpha = alpha
        self.mode = mode
        self.series = series
        self.tested = 0
        self.accepted = 0
        self._open = False

    def next_alpha(self):
        if self._open:
            raise RuntimeError("record() the previous candidate before asking for the next level")
        self._open = True
        k = self.tested + 1
        if self.mode == "fixed":
            return self.alpha
        a = self.alpha * gamma(k, self.series)
        if self.mode == "elond":
            a *= self.accepted + 1
        return min(a, self.alpha)

    def record(self, accepted):
        if not self._open:
            raise RuntimeError("record() called without a matching next_alpha()")
        self._open = False
        self.tested += 1
        self.accepted += bool(accepted)
