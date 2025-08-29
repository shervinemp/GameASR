import json
from typing import Dict, List, Optional

from ..llm.model import LLM
from ..llm.session import Session
from ..common.utils import get_logger


class GenerationService:
    def __init__(self, llm: LLM, max_iterations: int = 2):
        self.logger = get_logger(__file__)
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1
        self.max_iterations = max_iterations

    def _summarize_context(
        self,
        query: str,
        context_nodes: List[Dict],
        web_context: Optional[str] = None,
    ) -> str:
        """
        Summarizes the combined context from the graph and web search
        with a focus on the user's query.
        """
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
            f"**User Query:**\n{query}\n\n"
            f"**Context to Summarize:**\n{full_context}\n\n"
            "**Concise Summary:**"
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
            critique_section = (
                "\n\nPlease improve the previous answer based on the following critique:\n"
                f"**Critique:** {critique}\n"
            )

        prompt = (
            "You are a helpful assistant. Based on the context below, provide the best possible answer to the user's query.\n\n"
            f"**Context:**\n{context}\n\n"
            f"**Query:**\n{query}\n"
            f"{critique_section}\n\n"
            "**Answer:**"
        )
        return "".join(self.session(prompt)).strip()

    def _critique_answer(self, query: str, context: str, answer: str) -> str:
        """Critiques a given answer based on the context, checking for factual accuracy and completeness."""
        prompt = (
            "You are a fact-checker. Your task is to determine if the 'Proposed Answer' is fully supported by the 'Evidence'. "
            "If it is not, explain what is wrong or missing. If it is correct, simply respond with 'The answer is correct'.\n\n"
            f"**Evidence:**\n{context}\n\n"
            f"**Query:**\n{query}\n\n"
            f"**Proposed Answer:**\n{answer}\n\n"
            "**Critique:**"
        )
        return "".join(self.session(prompt)).strip()

    def generate_answer(
        self,
        query: str,
        context_nodes: List[Dict],
        web_context: Optional[str] = None,
    ) -> str:
        """
        Generates a final answer by summarizing context and then iteratively
        generating and critiquing the answer.
        """
        summarized_context = self._summarize_context(query, context_nodes, web_context)
        if not summarized_context or "Could not summarize" in summarized_context:
             return "I could not find enough information to formulate an answer."

        self.logger.info("Starting iterative self-correction for answer generation.")

        last_answer = ""
        critique = None
        for i in range(self.max_iterations):
            self.logger.info(f"Self-correction iteration {i + 1}/{self.max_iterations}")

            answer = self._generate_standalone_answer(query, summarized_context, critique)
            self.logger.debug(f"Iteration {i+1} Answer: {answer}")

            # If it's the last iteration, we don't need to critique the final answer.
            if i == self.max_iterations - 1:
                last_answer = answer
                break

            critique = self._critique_answer(query, summarized_context, answer)
            self.logger.debug(f"Iteration {i+1} Critique: {critique}")

            # If the answer is deemed correct, we can break early.
            if "the answer is correct" in critique.lower():
                self.logger.info("Answer deemed correct by critique. Halting iterations.")
                last_answer = answer
                break

            last_answer = answer # Keep the last generated answer in case the loop finishes

        self.logger.info(f"Final Answer after {self.max_iterations} iterations: {last_answer}")
        return last_answer
