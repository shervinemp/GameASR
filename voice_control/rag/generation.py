import json
from typing import Dict, List, Optional, Tuple

from ..llm.session import Session
from ..common.utils import get_logger


class Composer:
    def __init__(
        self,
        session: Session,
    ):
        self.logger = get_logger(__file__)
        self.session = session

    def __call__(self, query: str, context: str, n_iter: int = 3) -> str:
        summary = self.summarize_context(query=query, context=context)
        if not summary:
            return "Insufficient information to formulate an answer."

        self.logger.info(
            "Starting iterative self-correction for answer generation."
        )
        answer = summary
        is_correct = False
        critique = None
        for i in range(n_iter - 1):
            answer = self.generate_answer(
                query=query, context=summary, critique=critique
            )

            critique, is_correct = self.critique_answer(query, summary, answer)

            self.logger.debug(
                f"Iter {i+1}:\nAnswer: {answer}\nCritique: {critique}"
            )

            if is_correct:
                break

        answer = self.generate_answer(
            query=query, context=summary, critique=None
        )
        self.logger.debug(f"Final answer: {answer}")

        return answer

    def summarize_context(self, query: str, context: str) -> str:
        prompt = (
            "Your job is to summarize the provided context into clear and concise bulletpoints to help answer a user's query.\n"
            "Focus only on the information that is directly relevant to the query. Avoid making new assumptions or deductions.\n"
            "**IMPORTANT:** If you find conflicting information, you must highlight the contradictions in your summary.\n\n"
            f"**User Query:**\n{query}\n"
            f"**Context to Summarize:**\n{context}\n\n"
            "**Concise Summary (including contradictions):**"
        )
        summary = "".join(self.session(prompt)).strip()
        return summary

    def generate_answer(
        self, query: str, context: str, critique: Optional[str] = None
    ) -> str:
        critique_section = ""
        if critique:
            critique_section = f"Please softly improve the previous answer based on the following critique:\n**Critique:** {critique}\n"
        prompt = (
            "Based on the context below, provide the best possible isolated human-readable answer to the user's query.\n"
            # "**Do not, in any way, mention or allude to the scores.**\n"
            f"**Context:**\n{context}\n\n"
            f"**Query:**\n{query}\n"
            f"{critique_section}"
            "\n**Answer:**"
        )
        answer = "".join(self.session(prompt)).strip()
        return answer

    def critique_answer(
        self, query: str, context: str, answer: str
    ) -> Tuple[str, bool]:
        """Critiques a given answer and provides a structured JSON response."""
        prompt = (
            "You are a fact-checker. Your task is to critique the 'Proposed Answer' based on the 'Evidence'.\n"
            "Please return a JSON object with two keys:\n"
            "`explanation`: A string containing your reasoning. If incorrect, explain what is wrong or missing; Otherwise, this can be a simple confirmation (e.g., 'The answer is fully supported.').\n"
            "`is_correct`: A boolean indicating if the answer is fully supported by the evidence.\n"
            f"**Evidence:**\n{context}\n\n"
            f"**Query:**\n{query}\n\n"
            f"**Proposed Answer:**\n{answer}\n\n"
            "**JSON Response:**"
        )
        try:
            response_str = "".join(self.session(prompt)).strip()
            d = json.loads(response_str)
        except Exception as e:
            self.logger.error(
                f"Could not parse critique JSON or critique call failed: {e}",
                exc_info=True,
            )
            d = {"explanation": "", "is_correct": False}

        return d["explanation"], d["is_correct"]

    def extract_new_triplets(
        self, answer: str, context: str
    ) -> List[Dict[str, str]]:
        """Extracts new, high-confidence facts from the context that are supported by the final answer."""
        self.logger.info(
            "Extracting new triplets from the answer and context..."
        )
        prompt = (
            "You are a knowledge graph expert. Your task is to extract new, high-confidence facts from the 'Context' that are supported by the final 'Verified Answer'. "
            "Do not extract facts that are already directly present in 'Context'.\n"
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
            return triplets
        except Exception as e:
            self.logger.error(
                f"Failed to extract triplets from LLM response: {e}",
                exc_info=True,
            )
            return []
