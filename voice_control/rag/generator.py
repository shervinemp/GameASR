import json
from typing import Dict, List, Tuple

from ..llm.model import LLM
from ..llm.session import Session
from ..common.utils import get_logger


class GenerationService:
    def __init__(self, llm: LLM):
        self.logger = get_logger(__file__)
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1

    def generate(self, query: str, report: Dict, nodes: List[Dict]) -> Tuple[str, Dict, bool]:
        prompt = self._build_generation_prompt(query, report, nodes)
        response = "".join(self.session(prompt))
        self.logger.debug(f"Generation Response: {response}")

        try:
            generation_data = json.loads(response)
            new_report = generation_data.get("report", report)
            final_answer = generation_data.get("answer", None)
            is_verified = generation_data.get("is_verified", False)
            return final_answer, new_report, is_verified
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Failed to decode generation JSON: {response}")
            return None, report, False

    def verify(self, answer: str, report: Dict) -> str:
        if not answer:
            return None
        if not self._verify_answer(answer, report):
            return "I found some relevant information, but could not form a confident answer based on the facts."
        return answer

    def _build_generation_prompt(self, query: str, report: Dict, nodes: List[Dict]) -> str:
        nodes_info = [f"- {node['label']} ({node['id']}): {node.get('description', 'N/A')}" for node in nodes]
        nodes_str = "\n".join(nodes_info)
        return (
            "Task: Based on the provided evidence, update the report and provide the best possible answer to the user's query. "
            "Return a JSON object with three keys:\n"
            "1. 'report': A dictionary compiling the most relevant evidence found so far.\n"
            "2. 'answer': The best-guess, human-readable answer. Can be null if the answer is not yet known.\n"
            "3. 'is_verified': A boolean that is true only if the answer is completely verified by the evidence.\n"
            f" * User Query: '{query}'\n"
            f" * Current Report: {report}\n"
            f" * New Evidence (Promising Nodes):\n{nodes_str}\n"
            "Please provide the updated JSON object."
        )

    def _verify_answer(self, answer: str, report: Dict) -> bool:
        prompt = (
            "You are a fact-checker. Your task is to determine if the provided 'Answer' is fully supported by the 'Evidence'. "
            "Respond with only 'true' or 'false'.\n"
            f" * Evidence: {json.dumps(report, indent=2)}\n"
            f" * Answer: {answer}\n"
            "Is the answer fully and directly supported by the evidence? (true/false)"
        )
        response = "".join(self.session(prompt)).strip().lower()
        self.logger.debug(f"Verification response: {response}")
        return response == 'true'
