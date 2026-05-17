import unittest
from unittest.mock import patch, MagicMock


class TestReranker(unittest.TestCase):
    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_ranks_results(self, mock_ce):
        from voice_control.rag.retrieval import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_ce.return_value = mock_model

        reranker = Reranker()
        results = ["bad result", "good result", "ok result"]
        sorted_results, scores = reranker("test query", results)

        self.assertEqual(sorted_results, ["good result", "ok result", "bad result"])
        self.assertEqual(scores, [0.9, 0.5, 0.1])

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_empty_results(self, mock_ce):
        from voice_control.rag.retrieval import Reranker

        reranker = Reranker()
        sorted_results, scores = reranker("test query", [])

        self.assertEqual(sorted_results, [])
        self.assertEqual(scores, [])

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_custom_model(self, mock_ce):
        from voice_control.rag.retrieval import Reranker

        reranker = Reranker(model_name="custom-model")
        self.assertEqual(reranker.model_name, "custom-model")
        mock_ce.assert_called_with("custom-model")

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_single_result(self, mock_ce):
        from voice_control.rag.retrieval import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7]
        mock_ce.return_value = mock_model

        reranker = Reranker()
        sorted_results, scores = reranker("test query", ["only result"])

        self.assertEqual(sorted_results, ["only result"])
        self.assertEqual(scores, [0.7])


class TestComposer(unittest.TestCase):
    def setUp(self):
        from voice_control.rag.generation import Composer

        self.mock_session = MagicMock()
        self.composer = Composer(session=self.mock_session)

    def test_generate_answer(self):
        self._mock_session_response("test answer")
        answer = self.composer.generate_answer(
            query="test query", context="test context"
        )
        self.assertEqual(answer, "test answer")

    def test_generate_answer_with_critique(self):
        self._mock_session_response("improved answer")
        answer = self.composer.generate_answer(
            query="test query",
            context="test context",
            critique="needs more detail",
        )
        self.assertEqual(answer, "improved answer")

    def test_summarize_context(self):
        self._mock_session_response("summary")
        summary = self.composer.summarize_context(
            query="test query", context="test context"
        )
        self.assertEqual(summary, "summary")

    def _mock_session_response(self, text: str):
        self.mock_session.return_value = iter([text])

    def test_critique_answer_valid_json(self):
        self._mock_session_response(
            '{"explanation": "looks good", "is_correct": true}'
        )
        explanation, is_correct = self.composer.critique_answer(
            query="test query", context="context", answer="answer"
        )
        self.assertEqual(explanation, "looks good")
        self.assertTrue(is_correct)

    def test_critique_answer_invalid_json(self):
        self._mock_session_response("not json")
        explanation, is_correct = self.composer.critique_answer(
            query="test query", context="context", answer="answer"
        )
        self.assertEqual(explanation, "")
        self.assertFalse(is_correct)

    def test_critique_answer_markdown_fenced(self):
        self._mock_session_response(
            '```json\n{"explanation": "correct", "is_correct": true}\n```'
        )
        explanation, is_correct = self.composer.critique_answer(
            query="test query", context="context", answer="answer"
        )
        self.assertEqual(explanation, "correct")
        self.assertTrue(is_correct)

    def test_extract_new_triplets_empty(self):
        self._mock_session_response("[]")
        triplets = self.composer.extract_new_triplets(
            answer="answer", context="context"
        )
        self.assertEqual(triplets, [])

    def test_extract_new_triplets_with_data(self):
        import json

        expected = [
            {
                "subject": "Einstein",
                "predicate": "DEVELOPED",
                "object": "relativity",
            }
        ]
        self._mock_session_response(json.dumps(expected))
        triplets = self.composer.extract_new_triplets(
            answer="answer", context="context"
        )
        self.assertEqual(triplets, expected)


class TestIterativeCorrection(unittest.TestCase):
    def setUp(self):
        from voice_control.rag.generation import Composer

        self.mock_session = MagicMock()
        self.composer = Composer(session=self.mock_session)

    def test_composer_iterates_correctly(self):
        self.composer.summarize_context = MagicMock(return_value="test summary")

        self.composer.generate_answer = MagicMock(
            side_effect=[
                "first draft",
                "second draft",
                "final version",
            ]
        )
        self.composer.critique_answer = MagicMock(
            side_effect=[
                ("needs more detail", False),
                ("still incomplete", False),
            ]
        )

        result = self.composer.__call__(
            query="test query", context="test context", n_iter=3
        )

        self.assertEqual(self.composer.generate_answer.call_count, 3)
        self.assertEqual(self.composer.critique_answer.call_count, 2)
        self.assertEqual(result, "final version")

    def test_composer_breaks_early_on_correct(self):
        self.composer.summarize_context = MagicMock(return_value="test summary")

        self.composer.generate_answer = MagicMock(
            side_effect=["good answer", "final version"]
        )
        self.composer.critique_answer = MagicMock(
            return_value=("looks good", True)
        )

        result = self.composer.__call__(
            query="test query", context="test context", n_iter=3
        )

        self.assertEqual(self.composer.generate_answer.call_count, 2)
        self.assertEqual(self.composer.critique_answer.call_count, 1)
        self.assertEqual(result, "final version")


if __name__ == "__main__":
    unittest.main()
