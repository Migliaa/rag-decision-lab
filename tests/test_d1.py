from unittest import TestCase

from scripts.run_d1_pilot import ndcg_at, recall_at
from src.retrieval import Passage, lexical_tokens


class RetrievalChoicesTest(TestCase):
    def test_lexical_tokens_keep_financial_identifiers_and_numbers(self) -> None:
        self.assertEqual(
            lexical_tokens("ZX-401 costs 1.5% in an IRA."),
            ["zx-401", "costs", "1.5", "in", "an", "ira"],
        )

    def test_indexed_text_exposes_title_and_passage(self) -> None:
        passage = Passage("p1", "IRA rules", "The contribution limit applies.")
        self.assertEqual(passage.indexed_text, "IRA rules\nThe contribution limit applies.")

    def test_recall_counts_relevant_passages_within_cutoff(self) -> None:
        self.assertEqual(recall_at(["a", "x", "b"], {"a", "b", "c"}, 2), 1 / 3)
        self.assertEqual(recall_at(["a", "x", "b"], {"a", "b", "c"}, 3), 2 / 3)

    def test_ndcg_rewards_relevant_passages_near_the_top(self) -> None:
        relevant = {"a", "b"}
        self.assertGreater(ndcg_at(["a", "x", "b"], relevant, 3), ndcg_at(["x", "a", "b"], relevant, 3))

