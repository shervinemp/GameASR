from functools import cache
import os
from typing import Dict, List, Optional, Tuple

from ..llm.session import Session
from ..common.utils import get_logger, safe_json_loads


@cache
def _load_prompts() -> Dict[str, str]:
    """Loads prompts from prompts.yaml with fallback to built-in defaults."""
    defaults = {
        "summarize_context": (
            "Summarize the context below to answer the query.\n"
            "Rules:\n- Short bullet points\n- Use only facts from the context\n"
            "- No new information\n- If facts conflict, list both sides"
        ),
        "generate_answer": (
            "Read the context and answer the query.\n"
            "Rules:\n- Use only facts from the context\n- Short, direct answer"
        ),
        "generate_answer_with_critique": (
            "Read the context and answer the query.\n"
            "Rules:\n- Use only facts from the context\n- Short, direct answer\n"
            "Improve the previous answer"
        ),
        "critique_answer": (
            "Check if the Answer matches the Evidence.\n"
            "Return JSON with exactly two keys:\n"
            '- "explanation": describe what is right or wrong\n'
            '- "is_correct": true if answer uses only facts from evidence, else false'
        ),
        "extract_triplets": (
            "Extract new facts from the answer that are not in the context.\n"
            "Rules:\n- Skip facts already in the context\n"
            'Each fact is a triplet: {"subject": "X", "predicate": "REL", "object": "Y"}\n'
            "- Predicate examples: IS_A, HAS_MEMBER, WAS_BORN_IN, LOCATED_IN, DEVELOPED\n"
            "- If no new facts, return []"
        ),
    }
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (
        os.path.join(base, "prompts.yaml"),
        os.path.join(os.getcwd(), "prompts.yaml"),
    ):
        if os.path.exists(path):
            import yaml
            with open(path, encoding="utf-8") as f:
                user = yaml.safe_load(f)
                if user:
                    defaults.update(user)
            break
    return defaults


class Composer:
    def __init__(self, session: Session):
        self.logger = get_logger(__file__)
        self.session = session

    def __call__(self, query: str, context: str, n_iter: int = 3) -> str:
        if context.count(" is ") >= 2:
            summary = context
        else:
            summary = self.summarize_context(query=query, context=context)

        if not summary:
            return "Insufficient information to formulate an answer."

        self.logger.info("Starting iterative self-correction for answer generation.")
        answer = summary
        is_correct = False
        critique = None
        for i in range(n_iter - 1):
            answer = self.generate_answer(query=query, context=summary, critique=critique)
            critique, is_correct = self.critique_answer(query, context, answer)
            self.logger.debug(f"Iter {i+1}:\nAnswer: {answer}\nCritique: {critique}")
            if is_correct:
                break

        answer = self.generate_answer(query=query, context=summary, critique=critique)
        self.logger.debug(f"Final answer: {answer}")
        return answer

    def summarize_context(self, query: str, context: str) -> str:
        if not context.strip():
            return ""
        prompts = _load_prompts()
        prompt = f"{prompts['summarize_context']}\n\nQuery: {query}\nContext: {context}\nSummary:"
        summary = "".join(self.session(prompt)).strip()
        return summary

    def generate_answer(self, query: str, context: str, critique: Optional[str] = None) -> str:
        prompts = _load_prompts()
        if critique:
            prompt = f"{prompts['generate_answer_with_critique']}. Fix: {critique}\n"
        else:
            prompt = f"{prompts['generate_answer']}\n"
        prompt += f"Context: {context}\nQuery: {query}\nAnswer:"
        answer = "".join(self.session(prompt)).strip()
        return answer

    def critique_answer(self, query: str, context: str, answer: str) -> Tuple[str, bool]:
        prompts = _load_prompts()
        prompt = f"{prompts['critique_answer']}\n\nEvidence: {context}\nQuery: {query}\nAnswer: {answer}\n"
        prompt += '{"explanation": "...", "is_correct": true/false}'
        try:
            response_str = "".join(self.session(prompt)).strip()
            d = safe_json_loads(response_str, fallback={"explanation": "", "is_correct": False})
        except (ValueError, TypeError, RuntimeError) as e:
            self.logger.error(f"LLM call during critique failed: {e}", exc_info=True)
            d = {"explanation": "", "is_correct": False}
        return d["explanation"], d["is_correct"]

    def extract_new_triplets(self, answer: str, context: str) -> List[Dict[str, str]]:
        self.logger.info("Extracting new triplets from the answer and context...")
        prompts = _load_prompts()
        prompt = f"{prompts['extract_triplets']}\n\nContext: {context}\nAnswer: {answer}\nNew triplets:"
        try:
            response = "".join(self.session(prompt))
            triplets = safe_json_loads(response)
            return triplets
        except Exception as e:
            self.logger.error(f"Failed to extract triplets from LLM response: {e}", exc_info=True)
            return []
