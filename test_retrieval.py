import random
import unittest

import retrieval
from retrieval import BASELINE, GRID, Index, make_corpus, make_queries, neighbours, score_matrix


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = make_corpus(3, n_topics=8)
        cls.queries = make_queries(cls.corpus, 150, 4)

    def test_generation_is_deterministic(self):
        again = make_corpus(3, n_topics=8)
        self.assertEqual(again.bodies, self.corpus.bodies)
        self.assertEqual(again.titles, self.corpus.titles)
        self.assertEqual(make_queries(again, 150, 4)[10][0], self.queries[10][0])
        self.assertNotEqual(make_corpus(5, n_topics=8).bodies, self.corpus.bodies)

    def test_styles_differ(self):
        a = make_corpus(3, n_topics=8, style=0)
        b = make_corpus(3, n_topics=8, style=1)
        mean_len = lambda c: sum(map(len, c.bodies)) / len(c.bodies)
        self.assertGreater(mean_len(b), mean_len(a))

    def test_scores_are_bounded_and_ranking_is_stable(self):
        idx = Index(self.corpus, BASELINE)
        for terms, rel in self.queries:
            r = idx.recall_at_10(terms, rel)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            top = idx.top(terms)
            self.assertEqual(top, idx.top(terms))
            self.assertEqual(len(top), len(set(top)))
            self.assertLessEqual(len(top), 10)

    def test_retrieval_beats_chance_by_a_wide_margin(self):
        idx = Index(self.corpus, BASELINE)
        mean = sum(idx.recall_at_10(t, r) for t, r in self.queries) / len(self.queries)
        self.assertGreater(mean, 0.4)

    def test_exact_match_document_ranks_first(self):
        idx = Index(self.corpus, BASELINE)
        d = 17
        terms = list(set(self.corpus.bodies[d]))
        self.assertEqual(idx.top(terms, 1), [d])

    def test_unknown_terms_and_empty_query(self):
        idx = Index(self.corpus, BASELINE)
        self.assertEqual(idx.top(["no_such_word"]), [])
        self.assertEqual(idx.top([]), [])
        self.assertEqual(idx.recall_at_10(["no_such_word"], {1, 2}), 0.0)
        with self.assertRaises(ValueError):
            idx.recall_at_10(["c1"], set())

    def test_bad_configuration(self):
        for cfg in ((-1.0, 0.5, 1.0), (1.2, 1.5, 1.0), (1.2, 0.5, -1.0)):
            with self.assertRaises(ValueError):
                Index(self.corpus, cfg)

    def test_title_weight_zero_ignores_titles(self):
        idx = Index(self.corpus, (1.2, 0.75, 0.0))
        only_in_title = None
        for d, title in enumerate(self.corpus.titles):
            for w in title:
                if all(w not in body for body in self.corpus.bodies):
                    only_in_title = w
        if only_in_title is not None:
            self.assertEqual(idx.top([only_in_title]), [])

    def test_matrix_and_grid(self):
        cfgs = [BASELINE, (2.8, 0.0, 0.0)]
        m = score_matrix(self.corpus, self.queries, cfgs)
        self.assertEqual(set(m), set(cfgs))
        self.assertTrue(all(len(v) == len(self.queries) for v in m.values()))
        self.assertNotEqual(m[cfgs[0]], m[cfgs[1]])
        self.assertEqual(len(GRID), len(set(GRID)))
        self.assertIn(BASELINE, GRID)

    def test_neighbours(self):
        for cfg in GRID:
            ns = neighbours(cfg)
            self.assertTrue(1 <= len(ns) <= 6)
            for n in ns:
                self.assertIn(n, GRID)
                self.assertEqual(sum(a != b for a, b in zip(n, cfg)), 1)
        self.assertEqual(len(neighbours((0.3, 0.0, 0.0))), 3)


if __name__ == "__main__":
    unittest.main()
