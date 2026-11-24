"""A small retrieval task to evaluate the gates on something that is not Gaussian noise.

SYNTHETIC DATA. The corpus, queries and relevance labels are generated here from
a seed. Nothing in this file comes from a real dataset or a real system.

The task: BM25 search over a generated corpus, scored by Recall@10 per query.
A "candidate" is a BM25 configuration (k1, b, title weight) and the baseline is
the usual default (1.2, 0.75, 1). The per-query Recall@10 difference between a
candidate and the baseline is bounded in [-1, 1], is zero on many queries, and
is far from normal, which is the kind of paired score a gate has to cope with.
"""
import heapq
import math
import random

K1_GRID = (0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.8)
B_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TITLE_GRID = (0.0, 1.0, 2.0, 4.0)
BASELINE = (1.2, 0.75, 1.0)
GRID = tuple((k1, b, tw) for k1 in K1_GRID for b in B_GRID for tw in TITLE_GRID)


class Corpus:
    def __init__(self, titles, bodies, subtopic_of, sub_words, topic_words, common, sub_topic):
        self.titles = titles            # list of word lists
        self.bodies = bodies
        self.subtopic_of = subtopic_of  # doc id -> subtopic id
        self.sub_words = sub_words      # subtopic id -> words
        self.topic_words = topic_words  # topic id -> words
        self.common = common
        self.sub_topic = sub_topic      # subtopic id -> topic id
        self.relevant = {}
        for d, s in enumerate(subtopic_of):
            self.relevant.setdefault(s, set()).add(d)


def _zipf_pick(rng, words):
    # index ~ floor(len * u^2) favours the front of the list
    return words[int(len(words) * rng.random() ** 2)]


def make_corpus(seed, n_topics=30, subs_per_topic=5, style=0):
    """Generate a corpus. style changes the generator settings, not just the seed:
    style 1 has longer, noisier documents and weaker titles than style 0."""
    rng = random.Random(seed)
    med_len, sigma, distract, good_title = [(60, 0.7, 0.04, 0.7), (90, 0.9, 0.07, 0.5)][style]
    common = ["c%d" % i for i in range(400)]
    topic_words = [["t%d_%d" % (t, i) for i in range(50)] for t in range(n_topics)]
    sub_words, sub_topic = [], []
    for t in range(n_topics):
        for s in range(subs_per_topic):
            sub_words.append(["s%d_%d_%d" % (t, s, i) for i in range(12)])
            sub_topic.append(t)
    titles, bodies, subtopic_of = [], [], []
    for s in range(len(sub_words)):
        t = sub_topic[s]
        siblings = [x for x in range(len(sub_words)) if sub_topic[x] == t and x != s]
        for _ in range(rng.randint(3, 8)):
            length = int(max(15, min(400, rng.lognormvariate(math.log(med_len), sigma))))
            # A document says a roughly fixed amount about its subtopic. Extra
            # length is filler plus stray mentions of sibling subtopics.
            n_sub = max(2, int(rng.gauss(7, 2.5)))
            body = [_zipf_pick(rng, sub_words[s]) for _ in range(min(n_sub, length))]
            while len(body) < length:
                r = rng.random()
                if r < distract:
                    body.append(_zipf_pick(rng, sub_words[rng.choice(siblings)]))
                elif r < 0.35:
                    body.append(_zipf_pick(rng, topic_words[t]))
                else:
                    body.append(_zipf_pick(rng, common))
            if rng.random() < good_title:
                title = [_zipf_pick(rng, sub_words[s]) for _ in range(rng.randint(1, 3))]
                title += [_zipf_pick(rng, topic_words[t]) for _ in range(rng.randint(1, 3))]
            else:
                # A title that names a sibling subtopic or says nothing useful.
                other = rng.choice(siblings)
                title = [_zipf_pick(rng, sub_words[other]) for _ in range(rng.randint(0, 2))]
                title += [_zipf_pick(rng, common) for _ in range(rng.randint(2, 4))]
            titles.append(title)
            bodies.append(body)
            subtopic_of.append(s)
    return Corpus(titles, bodies, subtopic_of, sub_words, topic_words, common, sub_topic)


def make_queries(corpus, n, seed):
    """n queries as (terms, relevant doc ids). Short, and not always on target."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = rng.randrange(len(corpus.sub_words))
        terms = []
        for _ in range(rng.randint(2, 4)):
            r = rng.random()
            if r < 0.5:
                terms.append(_zipf_pick(rng, corpus.sub_words[s]))
            elif r < 0.85:
                terms.append(_zipf_pick(rng, corpus.topic_words[corpus.sub_topic[s]]))
            else:
                terms.append(_zipf_pick(rng, corpus.common))
        out.append((terms, corpus.relevant[s]))
    return out


class Index:
    """BM25 with a weighted title field, for one configuration."""

    def __init__(self, corpus, config):
        k1, b, tw = config
        if k1 < 0 or not 0 <= b <= 1 or tw < 0:
            raise ValueError("bad BM25 configuration %r" % (config,))
        n = len(corpus.bodies)
        tfs, lens = [], []
        df = {}
        for d in range(n):
            tf = {}
            for w in corpus.bodies[d]:
                tf[w] = tf.get(w, 0.0) + 1.0
            if tw:
                for w in corpus.titles[d]:
                    tf[w] = tf.get(w, 0.0) + tw
            tfs.append(tf)
            lens.append(len(corpus.bodies[d]) + tw * len(corpus.titles[d]))
            for w in tf:
                df[w] = df.get(w, 0) + 1
        avg = sum(lens) / float(n)
        self.postings = {}
        for d in range(n):
            norm = k1 * (1.0 - b + b * lens[d] / avg)
            for w, f in tfs[d].items():
                idf = math.log(1.0 + (n - df[w] + 0.5) / (df[w] + 0.5))
                self.postings.setdefault(w, []).append((d, idf * f * (k1 + 1.0) / (f + norm)))

    def top(self, terms, k=10):
        scores = {}
        for w in terms:
            for d, wt in self.postings.get(w, ()):
                scores[d] = scores.get(d, 0.0) + wt
        # ties broken by doc id so results do not depend on dict order
        return [d for _, d in heapq.nsmallest(k, ((-s, d) for d, s in scores.items()))]

    def recall_at_10(self, terms, relevant):
        if not relevant:
            raise ValueError("query has no relevant documents")
        hits = sum(1 for d in self.top(terms, 10) if d in relevant)
        return hits / float(min(10, len(relevant)))


def score_matrix(corpus, queries, configs):
    """{config: [Recall@10 for each query]} for every configuration."""
    out = {}
    for cfg in configs:
        idx = Index(corpus, cfg)
        out[cfg] = [idx.recall_at_10(t, rel) for t, rel in queries]
    return out


def neighbours(config):
    """Configurations one grid step away in exactly one parameter."""
    out = []
    for axis, grid in enumerate((K1_GRID, B_GRID, TITLE_GRID)):
        i = grid.index(config[axis])
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                c = list(config)
                c[axis] = grid[j]
                out.append(tuple(c))
    return out
