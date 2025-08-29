import json
from typing import Dict, List, Optional

from ..llm.model import LLM
from ..llm.session import Session
from ..common.utils import get_logger


class GenerationService:
    def __init__(self, llm: LLM):
        self.logger = get_logger(__file__)
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1

    def generate_answer(
        self,
        query: str,
        context_nodes: List[Dict],
        web_context: Optional[str] = None,
    ) -> str:
        """
        Generates a final answer based on the provided context,
        using an in-context reasoning prompt.
        """
        prompt = self._build_reasoning_prompt(query, context_nodes, web_context)
        response_str = "".join(self.session(prompt))
        self.logger.debug(f"LLM Reasoning Response: {response_str}")

        try:
            # The response is expected to be a JSON object with a 'reasoning'
            # and 'final_answer' field. We only need the final answer.
            response_json = json.loads(response_str)
            final_answer = response_json.get("final_answer", "I could not find a confident answer.")
            self.logger.info(f"Generated Answer: {final_answer}")
            return final_answer
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Failed to decode LLM response: {e}")
            # Fallback in case of malformed JSON
            return "I encountered an issue while formulating the response."

    def _build_reasoning_prompt(
        self,
        query: str,
        nodes: List[Dict],
        web_context: Optional[str] = None,
    ) -> str:
        """
        Builds a prompt that encourages the LLM to reason before answering,
        using context from both the knowledge graph and a web search.
        """
        if not nodes and not web_context:
            self.logger.warning("No context provided for generation.")
            return "No context provided to answer the query."

        # Format the knowledge graph context
        graph_context_str = "\n".join(
            [
                f"- Node ID: {node.get('id', 'N/A')}, Label: {node.get('label', 'N/A')}, Description: {node.get('description', 'N/A')}"
                for node in nodes
            ]
        )

        # Combine contexts
        full_context = ""
        if graph_context_str:
            full_context += f"**Knowledge Graph Context:**\n{graph_context_str}\n\n"
        if web_context:
            full_context += f"**Web Search Context:**\n{web_context}\n\n"

        prompt = (
            "Based on the following context, please perform two steps:\n"
            "1. First, provide a brief 'reasoning' of how the combined context can be used to answer the user's query.\n"
            "2. Second, based on your reasoning, provide a 'final_answer' to the query.\n\n"
            "Your response MUST be a JSON object with two keys: 'reasoning' and 'final_answer'.\n\n"
            f"{full_context}"
            f"**Query:**\n{query}\n\n"
            "**JSON Response:**"
        )
        return prompt
