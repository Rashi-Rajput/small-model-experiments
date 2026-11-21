import unittest
from unittest.mock import Mock, patch

import numpy as np

from task2_retrieval.retrieve import answer, page_spans, search, supporting_span


class EvidenceTests(unittest.TestCase):
    def test_overlapping_windows_keep_boundaries_and_share_reader_choices(self):
        pages = ["Intro. This rule applies to red boxes. Keep the lid closed. Other notes."]
        page = pages[0]
        seeds = []
        for word in ("red", "boxes"):
            start = page.index(word)
            seeds.append(dict(page=1, start=start, end=start + len(word), text=word))
        spans = [page_spans(page)]
        hits = search(seeds, np.ones((2, 1)), np.ones(1), pages, spans)
        self.assertEqual(hits[0]["text"], "Intro. This rule applies to red boxes. Keep the lid closed.")
        self.assertEqual(page[hits[1]["seed_start"]:hits[1]["seed_end"]], "boxes")
        response = {"choices": [{"message": {"content": "Keep the lid closed."}}]}
        with patch("task2_retrieval.retrieve.chat", return_value=response), \
                patch("task2_retrieval.retrieve.supporting_span", return_value=2):
            result = answer("", None, "What should happen to the lid?", pages, spans, hits)
        self.assertEqual(len(result["choices"]), 3)
        self.assertEqual((result["page"], result["answer"]), (1, "Keep the lid closed."))

    def test_empty_evidence_abstains(self):
        self.assertEqual(page_spans(""), [])
        with patch("task2_retrieval.retrieve.chat") as chat:
            result = answer("", None, "What is the deadline?", [], [], [])
        chat.assert_not_called()
        self.assertIsNone(result["answer"])

    def test_unknown_draft_abstains(self):
        page = "The package is red."
        hits = [dict(page=1, start=0, end=len(page))]
        completion = {"choices": [{"message": {"content": "Unknown."}}]}
        with patch("task2_retrieval.retrieve.chat", return_value=completion):
            result = answer("", None, "When was it shipped?", [page], [page_spans(page)], hits)
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["page"])

    def test_draft_terms_resolve_a_semantic_tie(self):
        model = Mock()
        model.tokenizer.return_value = {"input_ids": [[1, 2], [1, 3], [1, 2]]}
        choices = [(1, "First passage"), (2, "Second passage")]
        with patch("task2_retrieval.retrieve.embed", return_value=np.ones((4, 1))):
            self.assertEqual(supporting_span(model, "question", "draft", choices), 1)


if __name__ == "__main__":
    unittest.main()
