from typing import Dict, List, Optional, Tuple

from ..llm.session import Session
from ..common.utils import get_logger, safe_json_loads


class Composer:
    def __init__(
        self,
        session: Session,
    ):
        self.logger = get_logger(__file__)
        self.session = session

    def __call__(self, query: str, context: str, n_iter: int = 3) -> str:
        # Skip LLM summarization when context is already structured graph output
        # (sentences like "Entity is Relation Entity.")
        if context.count(" is ") >= 2:
            summary = context
        else:
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

            # Critique against ORIGINAL context, not the lossy summary
            critique, is_correct = self.critique_answer(query, context, answer)

            self.logger.debug(
                f"Iter {i+1}:\nAnswer: {answer}\nCritique: {critique}"
            )

            if is_correct:
                break

        answer = self.generate_answer(
            query=query, context=summary, critique=critique
        )
        self.logger.debug(f"Final answer: {answer}")

        return answer

    def summarize_context(self, query: str, context: str) -> str:
        if not context.strip():
            return ""

        prompt = (
            "Summarize the context below to answer the query.\n"
            "Rules:\n"
            "- Short bullet points\n"
            "- Use only facts from the context\n"
            "- No new information\n"
            "- If facts conflict, list both sides\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            "Summary:"
        )
        summary = "".join(self.session(prompt)).strip()
        return summary

    def generate_answer(
        self, query: str, context: str, critique: Optional[str] = None
    ) -> str:
        critique_section = ""
        if critique:
            critique_section = f"Improve the previous answer. Fix: {critique}\n"
        prompt = (
            "Read the context and answer the query.\n"
            "Rules:\n"
            "- Use only facts from the context\n"
            "- Short, direct answer\n"
            f"{critique_section}"
            f"Context: {context}\n"
            f"Query: {query}\n"
            "Answer:"
        )
        answer = "".join(self.session(prompt)).strip()
        return answer

    def critique_answer(
        self, query: str, context: str, answer: str
    ) -> Tuple[str, bool]:
        """Critiques a given answer and provides a structured JSON response."""
        prompt = (
            "Check if the Answer matches the Evidence.\n"
            "Return JSON with exactly two keys:\n"
            '- "explanation": describe what is right or wrong\n'
            '- "is_correct": true if answer uses only facts from evidence, else false\n\n'
            f"Evidence: {context}\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            '{"explanation": "...", "is_correct": true/false}'
        )
        try:
            response_str = "".join(self.session(prompt)).strip()
            d = safe_json_loads(response_str, fallback={"explanation": "", "is_correct": False})
        except (ValueError, TypeError, RuntimeError) as e:
            self.logger.error(
                f"LLM call during critique failed: {e}", exc_info=True
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
            "Extract new facts from the answer that are not in the context.\n"
            "Rules:\n"
            "- Skip facts already in the context\n"
            "- Each fact is a triplet: {\"subject\": \"X\", \"predicate\": \"REL\", \"object\": \"Y\"}\n"
            "- Predicate examples: IS_A, HAS_MEMBER, WAS_BORN_IN, LOCATED_IN, DEVELOPED\n"
            "- If no new facts, return []\n\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            "New triplets:"
        )
        try:
            response = "".join(self.session(prompt))
            triplets = safe_json_loads(response)
            return triplets
        except Exception as e:
            self.logger.error(
                f"Failed to extract triplets from LLM response: {e}",
                exc_info=True,
            )
            return []
