"""Benchmark for the gates. All data is SYNTHETIC (see retrieval.py).

Three parts:
  1. tune     pick MeanGate's bet settings on a simple parametric generator (dev)
  2. gate     per-candidate error and power on the retrieval task, a different
              generator the settings were not tuned on (held out)
  3. search   a hill-climb over BM25 configurations where every accept becomes
              the new baseline, run under four acceptance rules, on a dev
              corpus and on a held-out corpus (different seed and settings)

usage: python bench.py [scale]     scale 1.0 is the full run, 0.1 is a smoke run
"""
import json
import math
import random
import sys
import time

import retrieval
from search import Ledger
from seqgate import ACCEPT, CONTINUE, Gate, MeanGate

ALPHA = 0.05
MAX_N = 1000
WEAK_START = (2.8, 0.0, 0.0)   # untuned starting system for the search
DEV = dict(seed=1, style=0, query_seed=101)
HELD_OUT = dict(seed=2, style=1, query_seed=202)
TUNE_GRID = [dict(lam_max=l, prior_var=v) for l in (0.25, 0.5, 0.9) for v in (0.02, 0.25)]


# ---------------------------------------------------------------- methods
def z_peek(diffs, alpha_z=1.6449, start=20):
    """One-sided z-test re-run after every unit, stop the first time it passes."""
    s = ss = 0.0
    for n, d in enumerate(diffs, 1):
        s += d
        ss += d * d
        if n >= start:
            var = (ss - s * s / n) / (n - 1)
            if var > 0 and s / math.sqrt(var * n) > alpha_z:
                return True, n
    return False, len(diffs)


def z_fixed(diffs, alpha_z=1.6449):
    """One-sided z-test once, after the whole budget."""
    n = len(diffs)
    s = sum(diffs)
    var = (sum(d * d for d in diffs) - s * s / n) / (n - 1)
    return (var > 0 and s / math.sqrt(var * n) > alpha_z), n


def sign_gate(diffs, alpha=ALPHA):
    g = Gate(alpha=alpha, max_n=len(diffs))
    for d in diffs:
        if g.update(0.0, d) != CONTINUE:
            break
    return g.decision == ACCEPT, g.n


def mean_gate(diffs, alpha=ALPHA, **kw):
    g = MeanGate(alpha=alpha, max_n=len(diffs), bound=1.0, **kw)
    for d in diffs:
        if g.update(0.0, d) != CONTINUE:
            break
    return g.decision == ACCEPT, g.n


# ---------------------------------------------------------------- part 1
def dev_draw(rng, effect):
    """Dev generator: mostly-zero differences with a clipped Gaussian otherwise."""
    if rng.random() < 0.75:
        return 0.0
    return max(-1.0, min(1.0, rng.gauss(4.0 * effect, 0.2)))


def tune(trials, rng):
    effects = (0.0, 0.01, 0.02, 0.04)
    rows = []
    for cfg in TUNE_GRID:
        rates, used = [], []
        for e in effects:
            acc = tot = 0
            for _ in range(trials):
                ok, n = mean_gate([dev_draw(rng, e) for _ in range(MAX_N)], **cfg)
                acc += ok
                tot += n
            rates.append(acc / trials)
            used.append(tot / trials)
        rows.append(dict(cfg=cfg, false_accept=rates[0], power=sum(rates[1:]) / 3,
                         accept_by_effect=dict(zip(map(str, effects), rates)),
                         units_null=used[0]))
    valid = [r for r in rows if r["false_accept"] <= ALPHA]
    best = max(valid, key=lambda r: r["power"])
    return rows, best["cfg"]


# ---------------------------------------------------------------- part 2
def load(spec, n_queries):
    corpus = retrieval.make_corpus(spec["seed"], style=spec["style"])
    queries = retrieval.make_queries(corpus, n_queries, spec["query_seed"])
    return retrieval.score_matrix(corpus, queries, retrieval.GRID)


def mean(xs):
    return sum(xs) / float(len(xs))


BINS = [("worse than -0.01", lambda e: e < -0.01),
        ("0 to 0.01", lambda e: 0 < e <= 0.01),
        ("0.01 to 0.03", lambda e: 0.01 < e <= 0.03),
        ("above 0.03", lambda e: e > 0.03)]


def gate_table(matrix, cfg, trials, rng):
    base = matrix[WEAK_START]
    diffs = {c: [a - b for a, b in zip(v, base)] for c, v in matrix.items() if c != WEAK_START}
    eff = {c: mean(d) for c, d in diffs.items()}
    methods = [("z-test, peek every query", z_peek), ("z-test, fixed budget", z_fixed),
               ("seqgate Gate (sign)", sign_gate),
               ("seqgate MeanGate", lambda d: mean_gate(d, **cfg))]
    out = {name: {} for name, _ in methods}
    counts = {}

    def cell(label, n_trials, pool, flip):
        res = {name: [0, 0] for name, _ in methods}
        for _ in range(n_trials):
            d = diffs[rng.choice(pool)]
            sample = [d[rng.randrange(len(d))] for _ in range(MAX_N)]
            if flip:
                sample = [x if rng.random() < 0.5 else -x for x in sample]
            for name, fn in methods:
                ok, n = fn(sample)
                res[name][0] += ok
                res[name][1] += n
        for name, _ in methods:
            out[name][label] = [res[name][0] / n_trials, res[name][1] / n_trials]

    # Exact null with realistic differences: real per-query differences, each
    # with a random sign, so the mean is exactly zero.
    cell("null (sign-flipped)", 4 * trials, list(diffs), True)
    counts["null (sign-flipped)"] = len(diffs)
    for label, pred in BINS:
        pool = [c for c in diffs if pred(eff[c])]
        counts[label] = len(pool)
        if pool:
            cell(label, trials, pool, False)
    return out, counts


# ---------------------------------------------------------------- part 3
def propose(rng, current):
    if rng.random() < 0.2:
        c = rng.choice(retrieval.GRID)
        while c == current:
            c = rng.choice(retrieval.GRID)
        return c
    return rng.choice(retrieval.neighbours(current))


def one_search(matrix, means, rule, cfg, n_candidates, rng):
    """Hill-climb. Returns accepts, false accepts, units used, true final score."""
    current = WEAK_START
    ledger = Ledger(alpha=ALPHA, mode=rule if rule != "peek" else "fixed")
    accepts = false = used = 0
    nq = len(matrix[current])
    for _ in range(n_candidates):
        cand = propose(rng, current)
        a, b = matrix[cand], matrix[current]
        level = ledger.next_alpha()
        sample = []
        for _ in range(MAX_N):
            q = rng.randrange(nq)
            sample.append(a[q] - b[q])
        if rule == "peek":
            ok, n = z_peek(sample)
        else:
            ok, n = mean_gate(sample, alpha=level, **cfg)
        ledger.record(ok)
        used += n
        if ok:
            accepts += 1
            false += means[cand] <= means[current]
            current = cand
    return accepts, false, used, means[current]


RULES = [("z-test, peek every query", "peek"), ("MeanGate, 5% per candidate", "fixed"),
         ("MeanGate + spend (union bound)", "spend"), ("MeanGate + e-LOND ledger", "elond")]


def search_table(matrix, cfg, reps, n_candidates, seed):
    means = {c: mean(v) for c, v in matrix.items()}
    out = {}
    for name, rule in RULES:
        rng = random.Random(seed)
        acc = fdp = anyfalse = used = final = 0.0
        for _ in range(reps):
            a, f, u, m = one_search(matrix, means, rule, cfg, n_candidates, rng)
            acc += a
            fdp += f / float(a) if a else 0.0
            anyfalse += f > 0
            used += u
            final += m
        out[name] = dict(accepts=acc / reps, fdr=fdp / reps, any_false=anyfalse / reps,
                         units_per_candidate=used / reps / n_candidates,
                         final_recall=final / reps)
    out["_start_recall"] = means[WEAK_START]
    out["_best_recall"] = max(means.values())
    return out


# ---------------------------------------------------------------- report
def pct(x):
    return "%.1f%%" % (100 * x)


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    t0 = time.time()
    trials = max(20, int(1000 * scale))
    n_queries = max(300, int(2000 * scale))
    reps = max(5, int(100 * scale))
    n_cand = 60
    print("SYNTHETIC DATA. alpha = %.2f, budget = %d queries per candidate, scale = %s\n"
          % (ALPHA, MAX_N, scale))

    rows, cfg = tune(max(20, int(400 * scale)), random.Random(7))
    print("1. tuning MeanGate on the dev generator (%d trials per cell)" % max(20, int(400 * scale)))
    print("| lam_max | prior_var | false accept | mean accept at 0.01/0.02/0.04 | queries on a null |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %.0f |" % (r["cfg"]["lam_max"], r["cfg"]["prior_var"],
              pct(r["false_accept"]), pct(r["power"]), r["units_null"]))
    print("chosen: %s\n" % json.dumps(cfg))

    held = load(HELD_OUT, n_queries)
    table, counts = gate_table(held, cfg, trials, random.Random(8))
    labels = ["null (sign-flipped)"] + [l for l, _ in BINS if counts[l]]
    print("2. one candidate against a fixed baseline, held-out retrieval corpus")
    print("   (%d queries in the pool, %d trials per cell, %d for the null)" % (n_queries, trials, 4 * trials))
    print("   configurations per column: " + ", ".join("%s: %d" % (l, counts[l]) for l in labels))
    print("| method | " + " | ".join("%s: accepted | queries" % l for l in labels) + " |")
    print("|" + "---|" * (1 + 2 * len(labels)))
    for name in table:
        print("| %s | " % name + " | ".join("%s | %.0f" % (pct(table[name][l][0]), table[name][l][1])
                                             for l in labels) + " |")

    dev = load(DEV, n_queries)
    searches = {}
    for tag, matrix, seed in (("dev corpus", dev, 21), ("held-out corpus", held, 22)):
        s = search_table(matrix, cfg, reps, n_cand, seed)
        searches[tag] = s
        print("\n3. search on the %s: %d candidates per search, %d searches" % (tag, n_cand, reps))
        print("   Recall@10 of the starting system %.4f, of the best configuration in the grid %.4f"
              % (s["_start_recall"], s["_best_recall"]))
        print("| rule | accepts | share of accepts that were not improvements | searches with any false accept | queries per candidate | final Recall@10 |")
        print("|---|---|---|---|---|---|")
        for name, _ in RULES:
            r = s[name]
            print("| %s | %.1f | %s | %s | %.0f | %.4f |" % (name, r["accepts"], pct(r["fdr"]),
                  pct(r["any_false"]), r["units_per_candidate"], r["final_recall"]))

    if scale != 1.0:
        print("\nsmoke run finished in %.0f s; bench_results.json is only written at scale 1.0"
              % (time.time() - t0))
        return
    with open("bench_results.json", "w") as f:
        json.dump({"synthetic": True, "scale": scale, "alpha": ALPHA, "max_n": MAX_N,
                   "tuning": rows, "chosen": cfg, "gate_held_out": table,
                   "gate_held_out_configs": counts, "search": searches}, f, indent=1, sort_keys=True)
    print("\nwrote bench_results.json in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
