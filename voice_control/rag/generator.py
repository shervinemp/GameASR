import json
from typing import Dict, List, Optional

from ..llm.model import LLM
from ..llm.session import Session
from ..common.utils import get_logger
from .knowledge_base import KnowledgeGraph


class GenerationService:
    """Handles the final answer generation process."""
    def __init__(self, llm: LLM, graph: KnowledgeGraph, max_iterations: int = 2):
        self.logger = get_logger(__file__)
        self.session = Session(llm)
        self.graph = graph
        self.session.conversation._cutoff_idx = -1
        self.max_iterations = max_iterations

    def _summarize_context(
        self,
        query: str,
        context_nodes: List[Dict],
        web_context: Optional[str] = None,
    ) -> str:
        """Summarizes the combined context, highlighting contradictions."""
        self.logger.info("Summarizing context for the generator...")
        if not context_nodes and not web_context:
            self.logger.warning("No context provided for summarization.")
            return ""

        graph_context_str = "\n".join(
            [f"- {node.get('label', 'N/A')}: {node.get('description', 'N/A')} (Score: {node.get('relevance_score', 0):.2f})" for node in context_nodes]
        )
        full_context = ""
        if graph_context_str:
            full_context += f"**Knowledge Graph Context:**\n{graph_context_str}\n\n"
        if web_context:
            full_context += f"**Web Search Context:**\n{web_context}\n\n"

        prompt = (
            "You are a helpful assistant. Your job is to synthesize and summarize the provided context "
            "to help answer a user's query. Focus only on the information that is directly relevant to the query.\n\n"
            "**IMPORTANT:** If you find conflicting information between the 'Knowledge Graph Context' and the 'Web Search Context', "
            "you must highlight the contradiction in your summary.\n\n"
            f"**User Query:**\n{query}\n\n"
            f"**Context to Summarize:**\n{full_context}\n\n"
            "**Concise Summary (including any contradictions):**"
        )
        try:
            summary = "".join(self.session(prompt)).strip()
            self.logger.debug(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"Failed to summarize context: {e}", exc_info=True)
            return "Could not summarize the provided context."

    def _generate_standalone_answer(self, query: str, context: str, critique: Optional[str] = None) -> str:
        """Generates a single, standalone answer based on the context and an optional critique."""
        critique_section = ""
        if critique:
            critique_section = f"\n\nPlease improve the previous answer based on the following critique:\n**Critique:** {critique}\n"
        prompt = (
            "You are a helpful assistant. Based on the context below, provide the best possible answer to the user's query.\n\n"
            f"**Context:**\n{context}\n\n"
            f"**Query:**\n{query}\n"
            f"{critique_section}\n\n"
            "**Answer:**"
        )
        return "".join(self.session(prompt)).strip()

    def _critique_answer(self, query: str, context: str, answer: str) -> Dict:
        """
        Critiques a given answer and provides a structured JSON response.

        Returns:
            A dictionary with 'is_correct' (bool) and 'explanation' (str).
        """
        prompt = (
            "You are a fact-checker. Your task is to critique the 'Proposed Answer' based on the 'Evidence'.\n"
            "Please return a JSON object with two keys:\n"
            "1. `is_correct`: A boolean (true/false) indicating if the answer is fully supported by the evidence.\n"
            "2. `explanation`: A string containing your reasoning. If the answer is incorrect, explain what is wrong or missing. If it is correct, this can be a simple confirmation (e.g., 'The answer is fully supported.').\n\n"
            f"**Evidence:**\n{context}\n\n"
            f"**Query:**\n{query}\n\n"
            f"**Proposed Answer:**\n{answer}\n\n"
            "**JSON Response:**"
        )
        response_str = "".join(self.session(prompt)).strip()
        try:
            return json.loads(response_str)
        except (json.JSONDecodeError, TypeError):
            self.logger.error(f"Could not parse critique JSON: {response_str}")
            return {"is_correct": False, "explanation": "Failed to get a valid critique from the model."}

    def _extract_new_triplets(self, query: str, context: str, answer: str) -> List[Dict[str, str]]:
        """Extracts new, high-confidence facts from the web context that are supported by the final answer."""
        self.logger.info("Extracting new triplets from the answer and context...")
        prompt = (
            "You are a knowledge graph expert. Your task is to extract new, high-confidence facts from the 'Web Search Context' that are supported by the final 'Verified Answer'. "
            "Do not extract facts that are already present in the 'Knowledge Graph Context'.\n"
            "Format the new facts as a JSON list of triplets, where each triplet has 'subject', 'predicate', and 'object' keys. "
            "The predicate should be a concise action phrase (e.g., 'IS A', 'HAS MEMBER', 'WAS BORN IN').\n"
            "If no new facts can be confidently extracted, return an empty JSON list.\n\n"
            f"**Context:**\n{context}\n\n"
            f"**Verified Answer:**\n{answer}\n\n"
            "**New Triplets (JSON List):**"
        )
        try:
            response = "".join(self.session(prompt))
            triplets = json.loads(response)
            if triplets:
                self.logger.info(f"Extracted {len(triplets)} new triplets to be added to the graph.")
            return triplets
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Failed to extract triplets from LLM response: {e}")
            return []

    def generate_answer(
        self,
        query: str,
        context_nodes: List[Dict],
        web_context: Optional[str] = None,
        write_to_graph: bool = False,
    ) -> str:
        """Generates a final answer using a summarize -> self-correct -> write pipeline."""
        summarized_context = self._summarize_context(query, context_nodes, web_context)
        if not summarized_context or "Could not summarize" in summarized_context:
             return "I could not find enough information to formulate an answer."

        self.logger.info("Starting iterative self-correction for answer generation.")
        last_answer = ""
        critique_explanation = None
        for i in range(self.max_iterations):
            self.logger.info(f"Self-correction iteration {i + 1}/{self.max_iterations}")
            answer = self._generate_standalone_answer(query, summarized_context, critique_explanation)
            self.logger.debug(f"Iteration {i+1} Answer: {answer}")

            if i == self.max_iterations - 1:
                last_answer = answer
                break

            critique_result = self._critique_answer(query, summarized_context, answer)
            self.logger.debug(f"Iteration {i+1} Critique: {critique_result}")

            if critique_result.get("is_correct", False):
                self.logger.info("Answer deemed correct by critique. Halting iterations.")
                last_answer = answer
                break

            critique_explanation = critique_result.get("explanation", "")
            last_answer = answer

        self.logger.info(f"Final Answer after self-correction: {last_answer}")

        # Optional: Write new knowledge back to the graph
        if write_to_graph:
            new_triplets = self._extract_new_triplets(query, summarized_context, last_answer)
            if new_triplets:
                self.graph.add_triplets(new_triplets)

        return last_answer
