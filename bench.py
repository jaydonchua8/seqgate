"""Compare four ways of deciding "did this candidate beat the baseline?".

Synthetic search: each candidate has a true effect (0 for most, positive for a
few). One seed gives a baseline score and a candidate score that share a
seed-level noise term, plus independent run noise. All methods see the same
seeds in the same order.
"""
import json
import random
import sys

from seqgate import Gate, ACCEPT, CONTINUE, binom_tail

ALPHA = 0.05
MAX_N = 40
FIXED_N = 40
SEED_SD = 1.0   # noise shared by both runs on a seed
RUN_SD = 1.0    # independent noise per run


def draw(rng, effect):
    shared = rng.gauss(0, SEED_SD)
    return shared + rng.gauss(0, RUN_SD), shared + effect + rng.gauss(0, RUN_SD)


def best_of_3(pairs):
    """Run 3 seeds, keep the candidate if its mean is higher."""
    b = sum(p[0] for p in pairs[:3])
    c = sum(p[1] for p in pairs[:3])
    return c > b, 3


def peeking(pairs):
    """Sign test re-run after every seed, stop the first time p < alpha."""
    wins = 0
    for i, (b, c) in enumerate(pairs[:MAX_N], 1):
        wins += c > b
        if i >= 5 and binom_tail(wins, i) < ALPHA:
            return True, i
    return False, MAX_N


def fixed(pairs):
    """Sign test once, after a fixed number of seeds."""
    wins = sum(c > b for b, c in pairs[:FIXED_N])
    return binom_tail(wins, FIXED_N) < ALPHA, FIXED_N


def gated(pairs):
    g = Gate(alpha=ALPHA, max_n=MAX_N)
    for b, c in pairs:
        if g.update(b, c) != CONTINUE:
            break
    return g.decision == ACCEPT, g.n


METHODS = [("best of 3 seeds", best_of_3), ("sign test, peek every seed", peeking),
           ("sign test, fixed 40 seeds", fixed), ("seqgate", gated)]


def trial(effect, trials, rng):
    out = {name: [0, 0] for name, _ in METHODS}
    for _ in range(trials):
        pairs = [draw(rng, effect) for _ in range(MAX_N)]
        for name, fn in METHODS:
            ok, used = fn(pairs)
            out[name][0] += ok
            out[name][1] += used
    return {k: (v[0] / trials, v[1] / trials) for k, v in out.items()}


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(7)
    effects = [0.0, 0.5, 1.0, 1.5]
    table = {e: trial(e, trials, rng) for e in effects}

    print("trials per cell: %d, alpha = %.2f, seed budget = %d" % (trials, ALPHA, MAX_N))
    print("effect is in units of the per-run noise sd\n")
    head = "| method | false accept (effect 0) | seeds |"
    for e in effects[1:]:
        head += " accept at %.1f | seeds |" % e
    print(head)
    print("|" + "---|" * (3 + 2 * (len(effects) - 1)))
    for name, _ in METHODS:
        row = "| %s |" % name
        for e in effects:
            rate, seeds = table[e][name]
            row += " %.1f%% | %.1f |" % (100 * rate, seeds)
        print(row)

    # A search: 1000 candidates, 5% of them real improvements of 1.0 sd.
    rng = random.Random(11)
    n_cand, frac_real, eff = 1000, 0.05, 1.0
    reps = 20
    print("\nsearch of %d candidates, %.0f%% real (effect %.1f), averaged over %d searches"
          % (n_cand, 100 * frac_real, eff, reps))
    print("| method | accepted | of which not real | real found | seeds per candidate |")
    print("|---|---|---|---|---|")
    search = {}
    for name, fn in METHODS:
        search[name] = [0, 0, 0, 0]
    n_real_total = 0
    for _ in range(reps):
        for i in range(n_cand):
            real = rng.random() < frac_real
            n_real_total += real
            pairs = [draw(rng, eff if real else 0.0) for _ in range(MAX_N)]
            for name, fn in METHODS:
                ok, used = fn(pairs)
                s = search[name]
                s[0] += ok
                s[1] += ok and not real
                s[2] += ok and real
                s[3] += used
    results = {}
    for name, _ in METHODS:
        acc, false, true, used = search[name]
        share = 100.0 * false / acc if acc else 0.0
        print("| %s | %.1f | %.1f%% | %.1f%% | %.1f |" % (
            name, acc / reps, share, 100.0 * true / n_real_total, used / (reps * n_cand)))
        results[name] = {"accepted_per_search": acc / reps, "pct_accepted_not_real": round(share, 2),
                         "pct_real_found": round(100.0 * true / n_real_total, 2),
                         "seeds_per_candidate": round(used / (reps * n_cand), 2)}
    with open("bench_results.json", "w") as f:
        json.dump({"trials": trials, "alpha": ALPHA, "max_n": MAX_N,
                   "by_effect": {str(e): {k: [round(v[0], 5), round(v[1], 2)] for k, v in table[e].items()}
                                 for e in effects},
                   "search": results}, f, indent=1)


if __name__ == "__main__":
    main()
