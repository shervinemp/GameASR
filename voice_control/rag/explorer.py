import json
import random
from typing import List, Dict, Any

from ..llm.model import LLM
from ..llm.session import Session
from collections import OrderedDict
from typing import Any, Dict, List

from .graph import KnowledgeGraph
from ..common.utils import get_logger
from ..llm.model import LLM
from ..llm.session import Session


class Exploration:

    class Frontier:
        def __init__(self, maxlen: int | None = None):
            self._d = OrderedDict()
            self._maxlen = maxlen

        def append(self, x):
            k = x["id"]
            if k in self._d:
                self._d.move_to_end(k)
            self._d[k] = x
            if self._maxlen and len(self) > self._maxlen:
                _ = self.pop()

        def extend(self, X):
            for x in X:
                self.append(x)

        def pop(self):
            return self._d.popitem(last=False)

        def clear(self):
            self._d.clear()

        def __iter__(self):
            return iter(self._d.values())

        def __repr__(self):
            return f"Frontier({self._d})"

        def __str__(self):
            return self._d

        def __len__(self):
            return len(self._d)

    def __init__(self, graph: KnowledgeGraph, frontier_size: int = 12):
        self.graph = graph
        self.frontier = self.Frontier(frontier_size)
        self.ancestry = dict()

    def start(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        subgraph = self.graph.subgraph(node_ids)
        self._add(subgraph["nodes"], subgraph["relations"])
        return subgraph

    def expand(
        self,
        candidates: List[Dict[str, Dict[str, Any]]],
        flush_frontier: bool = False,
    ) -> None:
        if flush_frontier:
            self.frontier.clear()
        self._add(candidates["nodes"], candidates["relations"])

    @property
    def candidates(self) -> List[Dict[str, Dict[str, Any]]]:
        frontier_ids = [n["id"] for n in self.frontier]
        excluded_ids = list(self.ancestry.keys())
        candidates = self.graph.expansion(
            frontier_ids=frontier_ids, excluded_ids=excluded_ids
        )
        return candidates

    def _add(self, nodes: List[Dict[str, Any]], relations: List[Dict] = None) -> None:
        self.frontier.extend(nodes)

        for node in nodes:
            nid = node["id"]
            if nid not in self.ancestry:
                self.ancestry[nid] = nid

        if relations is not None:
            for relation in relations:
                a, b = relation["head"], relation["tail"]
                if relation["head"] not in self.ancestry:
                    a, b = b, a
                self.ancestry[b] = self.ancestry[a]

class ExplorationEngine:
    def __init__(self, graph: KnowledgeGraph, llm: LLM, max_iterations: int = 5, max_retries: int = 3):
        self.logger = get_logger(__file__)
        self.graph = graph
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    def explore(self, query: str, initial_nodes: List[str], initial_report: Dict, generation_service: "GenerationService"):
        state = Exploration(self.graph)
        state.start(initial_nodes)

        report = initial_report
        final_answer = None
        is_verified = False

        i = 0
        errors = 0
        while i < self.max_iterations:
            self.logger.info(f"\n--- Iteration {i + 1} ---")
            if not state.frontier:
                self.logger.info("Frontier is empty. Halting exploration.")
                break

            # Step 1: Score candidates
            scoring_prompt = self._build_scoring_prompt(query, state)
            scoring_response = "".join(self.session(scoring_prompt))
            self.logger.debug(f"Scoring Response: {scoring_response}")

            try:
                scored_candidates = json.loads(scoring_response)
                promising_ids = [c["id"] for c in scored_candidates if c.get("score", 0) > 5]
            except (json.JSONDecodeError, TypeError):
                self.logger.warning(f"Failed to decode scoring JSON: {scoring_response}")
                errors += 1
                if errors >= self.max_retries:
                    break
                i += 1
                continue

            nodes_to_expand = [
                n for kword_arr in state.candidates for c in kword_arr if (n := c["node"])["id"] in promising_ids
            ]

            # Step 2: Generate report and answer from promising nodes
            if nodes_to_expand:
                final_answer, report, is_verified = generation_service.generate(
                    query, report, nodes_to_expand
                )
            else:
                final_answer = None
                is_verified = False

            if is_verified:
                self.logger.info("Final answer found.")
                break

            if not nodes_to_expand:
                self.logger.info("No nodes to expand. Halting exploration.")
                break

            state.expand({"nodes": nodes_to_expand, "relations": None})
            i += 1
            errors = 0

        return final_answer, report

    def _build_scoring_prompt(self, query: str, state: Exploration, max_neighbors: int = 20) -> str:
        candidates = [
            item for kword_arr in state.candidates for item in random.sample(kword_arr, min(max_neighbors, len(kword_arr)))
        ]
        candidate_descs = [
            f"ID: {item['node']['id']}, Label: {item['node']['label']}, Description: {item['node'].get('description', 'N/A')}"
            for item in candidates
        ]
        candidates_str = "\n".join(candidate_descs)
        return (
            "Task: Score the following candidate nodes based on their relevance to the user's query. "
            "A score of 10 is highly relevant, and 1 is not relevant at all. "
            "Return a JSON array of objects, where each object has 'id' and 'score' keys.\n"
            f" * User Query: '{query}'\n"
            f" * Candidate Nodes:\n{candidates_str}\n"
            "Please provide the JSON array of scores."
        )
