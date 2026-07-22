import time
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

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_deduplicates_and_caches(self, mock_ce):
        from voice_control.rag.retrieval import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8]
        mock_ce.return_value = mock_model
        reranker = Reranker()

        first = reranker("query", ["alpha", "alpha", "bravo"])
        second = reranker("query", ["alpha", "alpha", "bravo"])

        self.assertEqual(first, second)
        self.assertEqual(first[0], ["bravo", "alpha"])
        mock_model.predict.assert_called_once()


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
        self.composer._context_needs_summary = MagicMock(return_value=True)
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
        self.composer._context_needs_summary = MagicMock(return_value=True)
        self.composer.summarize_context = MagicMock(return_value="test summary")

        self.composer.generate_answer = MagicMock(
            return_value="good answer"
        )
        self.composer.critique_answer = MagicMock(
            return_value=("looks good", True)
        )

        result = self.composer.__call__(
            query="test query", context="test context", n_iter=3
        )

        self.assertEqual(self.composer.generate_answer.call_count, 1)
        self.assertEqual(self.composer.critique_answer.call_count, 1)
        self.assertEqual(result, "good answer")

    def test_small_context_is_not_lossily_summarized(self):
        self.composer.summarize_context = MagicMock()
        self.composer.generate_answer = MagicMock(return_value="answer")

        result = self.composer(
            query="query",
            context="short authoritative evidence",
            n_iter=1,
        )

        self.assertEqual(result, "answer")
        self.composer.summarize_context.assert_not_called()


class TestParetoRetrieval(unittest.TestCase):
    def test_exact_graph_entities_avoid_keyword_llm_call(self):
        from voice_control.rag.retrieval import SmartGraphRetriever

        session = MagicMock()
        graph = MagicMock()
        graph.exact_label_search.return_value = {
            "captain price": {
                "id": "price",
                "label": "Captain Price",
                "description": "",
            },
            "soap": {"id": "soap", "label": "Soap", "description": ""},
        }
        primary = MagicMock()
        primary.backend = graph
        primary.search.return_value = []
        primary.format_results.return_value = []
        retriever = SmartGraphRetriever(session, primary)

        retriever("How is Captain Price connected to Soap?")

        keywords = primary.search.call_args.args[0]
        self.assertEqual(keywords, ["Captain Price", "Soap"])
        session.assert_not_called()

    def test_bounded_graph_retrieval_skips_keyword_llm_fallback(self):
        from voice_control.rag.retrieval import SmartGraphRetriever

        session = MagicMock()
        primary = MagicMock()
        primary.backend.exact_label_search.return_value = {}
        primary.search.return_value = []
        primary.format_results.return_value = []
        retriever = SmartGraphRetriever(session, primary)
        retriever._extract_keywords_ner = MagicMock(return_value=None)

        retriever("unknown entity", deadline=time.monotonic() + 1)

        primary.search.assert_called_once()
        self.assertEqual(primary.search.call_args.args[0], ["unknown entity"])
        session.assert_not_called()

    @patch("sentence_transformers.CrossEncoder")
    def test_graph_first_deduplicates_and_caches_before_rerank(self, mock_ce):
        from voice_control.rag.model import BaseRAG

        class DummyRAG(BaseRAG):
            def _attach_graph_retriever(self, backend):
                pass

            def __call__(self, query: str, **kwargs) -> str:
                return self.retrieve_context(query, kwargs.get("top_k"))

        session = MagicMock()
        rag = DummyRAG(session=session, web_search=True)
        graph_retriever = MagicMock(return_value=["alpha", "alpha", "bravo"])
        rag.retrievers = [graph_retriever]
        rag.web_retriever = MagicMock(return_value=["web result"])
        rag.reranker = MagicMock(return_value=(["bravo", "alpha"], [0.9, 0.8]))

        first = rag.retrieve_context("query", top_k=2)
        second = rag.retrieve_context("query", top_k=2)

        self.assertEqual(first, "[graph] bravo\n[graph] alpha")
        self.assertEqual(second, first)
        rag.reranker.assert_called_once_with(
            "query", results=["alpha", "bravo"]
        )
        graph_retriever.assert_called_once()
        rag.web_retriever.assert_not_called()

    @patch("sentence_transformers.CrossEncoder")
    def test_web_is_only_used_when_graph_has_no_evidence(self, mock_ce):
        from voice_control.rag.model import BaseRAG

        class DummyRAG(BaseRAG):
            def _attach_graph_retriever(self, backend):
                pass

            def __call__(self, query: str, **kwargs) -> str:
                return self.retrieve_context(query, kwargs.get("top_k"))

        rag = DummyRAG(session=MagicMock(), web_search=True)
        rag.retrievers = [MagicMock(return_value=[])]
        rag.web_retriever = MagicMock(return_value=["fallback"])
        rag.reranker = MagicMock(return_value=(["fallback"], [0.9]))

        result = rag.retrieve_context("query", top_k=1)

        self.assertEqual(result, "[web] fallback")
        rag.web_retriever.assert_called_once()

    def test_pipeline_registers_evidence_callback_not_answer_callback(self):
        from voxpipe.llm.conversation import Conversation
        from voice_control.pipeline import Pipeline

        class FakeRAG:
            def retrieve_context(self, query: str, top_k: int = 5) -> str:
                return f"evidence for {query}"

            def __call__(self, query: str) -> str:
                return f"answer for {query}"

        pipeline = Pipeline.__new__(Pipeline)
        pipeline._rag = FakeRAG()
        pipeline.session = MagicMock()
        pipeline.session.conversation = Conversation()

        pipeline._configure_session()

        callback = pipeline.session.conversation.tools["retrieve"].callback
        self.assertEqual(callback.__name__, "retrieve_context")

    def test_verified_spath_draft_is_not_regenerated(self):
        from voice_control.rag.model import SPathRAG

        rag = SPathRAG.__new__(SPathRAG)
        rag._retrieve_with_optional_draft = MagicMock(
            return_value=("[graph] evidence", "approved answer")
        )
        rag.composer = MagicMock()
        rag._backend = None
        rag._graph = None
        rag._web_search_enabled = False

        result = rag("query", top_k=5, max_iterations=3)

        self.assertEqual(result, "approved answer")
        rag.composer.assert_not_called()

    def test_spath_retrieve_context_is_strictly_evidence_only(self):
        from voice_control.rag.model import SPathRAG

        rag = SPathRAG.__new__(SPathRAG)
        rag.runtime = MagicMock(top_k=4)
        rag._retrieve_with_optional_draft = MagicMock(
            return_value=("[graph] evidence", None)
        )

        result = rag.retrieve_context("query")

        self.assertEqual(result, "[graph] evidence")
        rag._retrieve_with_optional_draft.assert_called_once_with(
            "query", 4, 1
        )

    def test_expired_web_deadline_performs_no_network_request(self):
        from voice_control.rag.retrieval import WebRetriever

        retriever = WebRetriever(MagicMock())
        retriever._get_ddgs = MagicMock()
        result = retriever.search(
            "query",
            deadline=time.monotonic() - 1,
        )

        self.assertEqual(result, [])
        retriever._get_ddgs.assert_not_called()

    def test_web_search_pins_ddgs_to_duckduckgo_backend(self):
        from voice_control.rag.retrieval import WebRetriever

        retriever = WebRetriever(MagicMock())
        ddgs = MagicMock()
        ddgs.text.return_value = [{"title": "result"}]
        retriever._get_ddgs = MagicMock(return_value=ddgs)
        retriever._rate_limit = MagicMock(return_value=True)

        result = retriever._search_ddgs(
            "query", 1, time.monotonic() + 5
        )

        self.assertEqual(result, [{"title": "result"}])
        ddgs.text.assert_called_once_with(
            "query", max_results=1, backend="duckduckgo"
        )


class TestKnowledgeGraphQueries(unittest.TestCase):
    def _graph_with_session(self):
        from voice_control.rag.knowledge import KnowledgeGraph

        graph = KnowledgeGraph.__new__(KnowledgeGraph)
        graph._database = "neo4j"
        graph._query_timeout = 5.0
        graph._driver = MagicMock()
        session = MagicMock()
        graph._driver.session.return_value.__enter__.return_value = session
        graph._execute_with_retry = lambda fn: fn()
        return graph, session

    def test_shortest_path_uses_native_bounded_selector(self):
        graph, session = self._graph_with_session()
        record = MagicMock()
        record.data.return_value = {"nodes": [], "relations": [], "weight": 1}
        session.run.return_value = [record]

        graph.k_shortest_paths_batch([("alpha", "bravo")], k=3)

        query = session.run.call_args.args[0].text
        self.assertIn("SHORTEST 3", query)
        self.assertIn("CALL (source, target)", query)
        self.assertNotIn("collect(path)", query)

    def test_exact_label_uses_normalized_index_with_legacy_fallback(self):
        graph, session = self._graph_with_session()
        session.run.side_effect = [
            [
                {
                    "query_label": "alpha",
                    "id": "1",
                    "label": "Alpha",
                    "description": "indexed",
                }
            ],
            [
                {
                    "query_label": "bravo",
                    "id": "2",
                    "label": "Bravo",
                    "description": "legacy",
                }
            ],
        ]

        result = graph.exact_label_search(["Alpha", "Bravo"])

        self.assertEqual(set(result), {"alpha", "bravo"})
        indexed_query = session.run.call_args_list[0].args[0].text
        legacy_query = session.run.call_args_list[1].args[0].text
        self.assertIn("normalized_label", indexed_query)
        self.assertIn("normalized_label IS NULL", legacy_query)

    def test_remote_plaintext_neo4j_uri_is_rejected(self):
        from voice_control.rag.knowledge import KnowledgeGraph

        from voxpipe.core.exceptions import StorageError
        with self.assertRaises(StorageError):
            KnowledgeGraph._validate_uri("bolt://192.0.2.10:7687")
        KnowledgeGraph._validate_uri("bolt://127.0.0.1:7687")
        KnowledgeGraph._validate_uri("neo4j+s://db.example.com:7687")

    def test_shortest_path_embeds_only_unresolved_entities(self):
        from voice_control.rag.retrieval import ShortestPathStrategy

        backend = MagicMock()
        backend.exact_label_search.return_value = {
            "alpha": {"id": "a", "label": "Alpha", "description": ""}
        }
        backend.vector_search.return_value = [
            [{"id": "b", "label": "Bravo", "description": ""}]
        ]
        backend.k_shortest_paths_batch.return_value = []
        embedder = MagicMock()
        embedder.encode.return_value = [[0.1]]
        strategy = ShortestPathStrategy(backend, embedder=embedder)

        strategy.search(["Alpha", "unknown Bravo"])

        embedder.encode.assert_called_once_with(["unknown Bravo"])
        backend.k_shortest_paths_batch.assert_called_once_with(
            [("a", "b")], k=3
        )

    def test_path_format_preserves_relationship_direction(self):
        from voice_control.rag.retrieval import ShortestPathStrategy

        strategy = ShortestPathStrategy(MagicMock())
        result = strategy.format_results(
            [
                {
                    "nodes": [
                        {"id": "a", "label": "Alpha"},
                        {"id": "b", "label": "Bravo"},
                    ],
                    "relations": [
                        {"source": "b", "target": "a", "type": "KNOWS"}
                    ],
                }
            ]
        )

        self.assertEqual(result, ["Bravo is Knows Alpha."])

    def test_import_embeddings_include_descriptions_and_normalized_labels(self):
        from voice_control.rag.data import Neo4jImporter

        importer = Neo4jImporter.__new__(Neo4jImporter)
        importer._embedding_model = MagicMock()
        vector = MagicMock()
        vector.tolist.return_value = [0.1, 0.2]
        importer._embedding_model.encode.return_value = [vector]

        result = importer._generate_entity_embeddings(
            {"1": {"label": "Alpha", "description": "Squad leader"}}
        )

        importer._embedding_model.encode.assert_called_once_with(
            ["Alpha. Squad leader"],
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.assertEqual(result[0]["normalized_label"], "alpha")
        self.assertEqual(result[0]["embedding"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
