# seqgate

Decide whether a candidate really beats a baseline, one seed at a time, without fooling yourself by looking early.

An automated search that proposes and evaluates many candidates has a quiet failure mode. Each evaluation is noisy, the loop keeps running seeds until something looks better, and most candidates are not improvements. Under those conditions a large share of what gets accepted is noise. `seqgate` is a small accept/reject gate for that loop. You feed it the baseline score and the candidate score from one shared seed at a time, and it tells you to accept, reject, or keep going. The chance that it accepts a candidate that is not better stays under the level you set, no matter how many times you check.

It is one file, 157 lines of Python, with no dependencies. Copy `seqgate.py` into a project to use it.

## Results

`python bench.py` simulates paired evaluations where both runs on a seed share a noise term and each run adds its own noise. The effect size is in units of the per-run noise standard deviation. Every method sees the same seeds in the same order, with a budget of 40 seeds, a 5% level, and 20,000 trials per cell.

| method | false accept (effect 0) | seeds | accept at 0.5 | seeds | accept at 1.0 | seeds | accept at 1.5 | seeds |
|---|---|---|---|---|---|---|---|---|
| best of 3 seeds | 50.1% | 3.0 | 73.3% | 3.0 | 89.1% | 3.0 | 96.6% | 3.0 |
| sign test, peek every seed | 14.1% | 36.7 | 66.5% | 25.2 | 98.1% | 13.3 | 100.0% | 8.4 |
| sign test, fixed 40 seeds | 3.9% | 40.0 | 50.7% | 40.0 | 96.4% | 40.0 | 100.0% | 40.0 |
| seqgate | 1.2% | 22.9 | 20.3% | 27.6 | 76.6% | 24.0 | 98.7% | 15.8 |

The second row is the one that motivated this. Running a correct 5% sign test after every seed and stopping the first time it passes gives a 14.1% false accept rate, not 5%.

The second part of the benchmark is a search of 1000 candidates where 5% are real improvements of 1.0, averaged over 20 searches.

| method | accepted | of which not real | real found | seeds per candidate |
|---|---|---|---|---|
| best of 3 seeds | 523.3 | 91.6% | 88.9% | 3.0 |
| sign test, peek every seed | 185.8 | 74.1% | 98.0% | 35.4 |
| sign test, fixed 40 seeds | 84.2 | 43.8% | 96.2% | 40.0 |
| seqgate | 49.0 | 25.2% | 74.7% | 23.1 |

`seqgate` has the lowest share of accepted candidates that are not real and uses 23.1 seeds per candidate against 40 for the fixed test. It pays for that in power. It finds 74.7% of the real improvements where the fixed 40 seed test finds 96.2%, and at an effect of 0.5 it accepts only 20.3%. A quarter of its accepts are still not real, because a 5% level per candidate is not a guarantee about the whole search when 95% of candidates are null. `spend()` (below) is the strict answer to that and costs more power.

## How it works

The gate keeps a wealth that starts at 1. Before each seed it bets a fraction of the wealth that the candidate will win the seed. A win multiplies the wealth by `1 + bet`, a loss by `1 - bet`, and a tie leaves it alone. If the candidate is not better, each seed is at best a fair coin, so the wealth cannot grow on average. By Ville's inequality the probability that it ever reaches `1 / alpha` is at most `alpha`. That bound does not depend on when or how often you look, which is why stopping early is allowed here and not in the peeking baseline.

The bet is chosen from earlier seeds only: `2p - 1`, capped at 0.9, where `p` is the win rate so far with one pseudo-win and one pseudo-loss added. That is the Kelly bet for a coin with win rate `p` against a fair coin.

The gate also stops when acceptance has become impossible. After each seed it computes the wealth it would reach if every remaining seed were a win. If that is still under the threshold it rejects with reason `futile`. This never changes a decision (there is a test for that), it only saves the remaining seeds. In the benchmark it is why a null candidate costs 22.9 seeds on average and not 40.

```python
from seqgate import Gate, CONTINUE

gate = Gate(alpha=0.05, max_n=40)
for seed in range(40):
    if gate.update(evaluate(baseline, seed), evaluate(candidate, seed)) != CONTINUE:
        break
print(gate.summary())   # decision, reason, seeds_used, wins, losses, e_value
```

For a long search, `Gate(alpha=spend(0.05, k))` for the k-th candidate bounds the chance of any false accept over the whole search at 5%.

There is also a CLI: `python seqgate.py example_scores.csv` reads one `baseline,candidate` pair per line and prints the decision as JSON.

## What I tried that did not help much

I tried starting the bet with an optimistic prior (1.5, 2 or 3 pseudo-wins against 1 pseudo-loss) so the first seeds are not wasted. In a 4,000 trial run the accept rate at an effect of 1.0 went from 76.4% to between 78.1% and 79.4%, and the false accept rate went from 1.1% to between 1.6% and 2.2%. That was too small a gain to justify another parameter, so I kept the symmetric prior.

## Limits

- All data is synthetic Gaussian noise. I have not run it on scores from real experiments or wired it into any search system.
- It only uses whether the candidate won each seed, not by how much. A test that uses the size of the difference would have more power. That is the main reason it trails the fixed test at small effects, and it is what I would do next.
- "Not better" here means the candidate wins at most half of the non-tied seeds. A candidate that wins rarely but by a lot can have a higher mean and still be rejected.
- Seeds have to be independent and the pairing has to be honest: same seed, same data, same budget for both runs.
- `spend()` is a union bound, so it is conservative for long searches.

## Run it

```
python test_seqgate.py
python bench.py
python seqgate.py example_scores.csv
```

Tested on Python 3.13. Nothing to install. The tests take about 30 seconds because two of them simulate the false accept rate.
