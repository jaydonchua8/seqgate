"""Demo: python demo.py   (about 20 seconds, SYNTHETIC data, no arguments)

Builds a small synthetic retrieval task, then runs the same hill-climb over
BM25 configurations twice: once accepting a candidate the first time a z-test
passes, and once behind MeanGate with the e-LOND ledger. Because the data is
generated, the true Recall@10 of every configuration is known, so each accept
can be marked as a real improvement or not.
"""
import random

import retrieval
from bench import ALPHA, MAX_N, WEAK_START, mean, propose, z_peek
from search import Ledger
from seqgate import ACCEPT, CONTINUE, MeanGate


def climb(matrix, means, gated, seed, n_candidates=40):
    rng = random.Random(seed)
    current = WEAK_START
    ledger = Ledger(alpha=ALPHA, mode="elond")
    used = false = accepts = 0
    for k in range(1, n_candidates + 1):
        cand = propose(rng, current)
        a, b = matrix[cand], matrix[current]
        level = ledger.next_alpha()
        sample = []
        for _ in range(MAX_N):
            q = rng.randrange(len(a))
            sample.append(a[q] - b[q])
        if gated:
            gate = MeanGate(alpha=level, max_n=MAX_N, bound=1.0)
            for d in sample:
                if gate.update(0.0, d) != CONTINUE:
                    break
            ok, n = gate.decision == ACCEPT, gate.n
        else:
            ok, n = z_peek(sample)
        ledger.record(ok)
        used += n
        if ok:
            true_gain = means[cand] - means[current]
            accepts += 1
            false += true_gain <= 0
            print("  candidate %2d  k1=%.1f b=%.2f title=%.0f  accepted after %4d queries, "
                  "true change %+.4f%s" % ((k,) + cand + (n, true_gain,
                                            "" if true_gain > 0 else "   <- not an improvement")))
            current = cand
    print("  %d accepts, %d not improvements, %d queries, true Recall@10 %.4f -> %.4f\n"
          % (accepts, false, used, means[WEAK_START], means[current]))


def main():
    print("SYNTHETIC DATA: generated corpus and queries, see retrieval.py\n")
    corpus = retrieval.make_corpus(seed=5, style=1)
    queries = retrieval.make_queries(corpus, 800, seed=6)
    matrix = retrieval.score_matrix(corpus, queries, retrieval.GRID)
    means = {c: mean(v) for c, v in matrix.items()}
    print("%d documents, %d queries, %d BM25 configurations, 40 candidates per search"
          % (len(corpus.bodies), len(queries), len(matrix)))
    print("start k1=%.1f b=%.2f title=%.0f, true Recall@10 %.4f, best in grid %.4f\n"
          % (WEAK_START + (means[WEAK_START], max(means.values()))))
    print("accept the first time a one-sided 5% z-test passes (checked after every query):")
    climb(matrix, means, gated=False, seed=3)
    print("MeanGate with the e-LOND ledger (false discovery rate 5% over the search):")
    climb(matrix, means, gated=True, seed=3)


if __name__ == "__main__":
    main()
