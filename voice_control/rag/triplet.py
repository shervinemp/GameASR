import json
import sys
from typing import Dict, Any

from ..llm.session import Session

from ..common.utils import setup_logging


class KnowledgeGraphTripletExtractor:
    def __init__(self):
        """
        Initializes the extractor with an LLM model instance.
        """
        self.session = Session()

    def _generate_triplet_prompt(self, text: str, existing_schema: Dict = None) -> str:
        """
        Generates the detailed prompt for the LLM to extract knowledge graph triplets.
        The prompt explicitly instructs the LLM on the desired output format,
        including how to represent adjectives, adverbs, and contextual information as properties.
        """
        prompt_parts = [
            "Your task is to extract all factual knowledge from the following text in the form of Subject-Predicate-Object (S-P-O) triplets.",
            "Each triplet should be a concise, accurate, and comprehensive representation of information.",
            "The output MUST be a JSON array of objects, where each object represents a triplet.",
            "Each triplet object must have 'subject', 'predicate', and 'object' keys.",
            "Subjects and Objects should be specific entities or concepts. They can include an optional 'properties' key (a dictionary) for attributes or descriptive adjectives (e.g., 'brilliant', 'red').",
            "Predicates should be concise phrases describing relationships. They can include an optional 'properties' key (a dictionary) for adverbs or contextual details like 'manner', 'time', 'condition', or 'reason'.",
            "Example of desired JSON output format:",
            "```json",
            "[",
            '  {"subject": {"name": "Albert Einstein", "properties": {"nationality": "German-born", "type": "theoretical physicist", "attribute": "brilliant"}}, "predicate": {"name": "developed"}, "object": {"name": "theory of relativity", "properties": {"significance": "groundbreaking"}}},',
            '  {"subject": {"name": "The car"}, "predicate": {"name": "drove down", "properties": {"manner": "quickly", "condition": "if the light was green"}}, "object": {"name": "the street"}}',
            "]",
            "```",
            "Guidelines for extraction:",
            "- Replace pronouns with actual entities.",
            "- Break down complex sentences into multiple simple triplets.",
            "- Capture all information verifiable from the text, including nuances as properties.",
            "- Ensure the triplets are logically consistent and coherent.",
            "- Use clear and concise language for predicates and properties.",
            "- Avoid redundant or overlapping triplets.",
            "- If a value for a field is not found, omit.",
        ]

        if existing_schema:
            prompt_parts.append("\nConsider the following predefined schema elements:")
            if existing_schema.get("entity_types"):
                prompt_parts.append(
                    f"- **Entity Types (guidance):** {', '.join(existing_schema['entity_types'])}"
                )
            if existing_schema.get("relation_types"):
                prompt_parts.append(
                    f"- **Relation Types (guidance):** {', '.join(existing_schema['relation_types'])}"
                )
            if existing_schema.get("entity_properties"):
                props = ", ".join(
                    [
                        f"{k}: {', '.join(v)}"
                        for k, v in existing_schema["entity_properties"].items()
                    ]
                )
                prompt_parts.append(
                    f"- **Common Entity Properties (guidance):** {props}"
                )
            if existing_schema.get("relation_properties"):
                props = ", ".join(
                    [
                        f"{k}: {', '.join(v)}"
                        for k, v in existing_schema["relation_properties"].items()
                    ]
                )
                prompt_parts.append(
                    f"- **Common Relation Properties (guidance):** {props}"
                )

        prompt_parts.append("\nHere is the text to analyze:")
        prompt_parts.append(f"```text\n{text}\n```")
        prompt_parts.append(
            "\nPlease provide the JSON array of triplets, conforming strictly to the specified schema."
        )

        return "\n".join(prompt_parts)

    def extract_triplets(
        self, text: str, existing_schema: Dict = None
    ) -> Dict[str, Any]:
        """
        Extracts knowledge graph triplets from the given text using the LLM,
        formatted as an OpenAI tool call.

        In a real system, this method would interact with the LLM API.
        For this implementation, it simulates the LLM's tool call response.
        """
        prompt = self._generate_triplet_prompt(text, existing_schema)
        response = "".join(self.session(prompt, tool_choice="knowledge_graph_triplets"))
        return response


def main():
    setup_logging("DEBUG", stream=sys.stdout)

    extractor = KnowledgeGraphTripletExtractor()

    my_schema = {
        "entity_types": [
            "Person",
            "Organization",
            "Location",
            "Event",
            "Concept",
            "Object",
        ],
        "relation_types": [
            "born_in",
            "developed",
            "awarded",
            "located_in",
            "designed_by",
            "created_by",
            "flows_through",
            "causes",
            "modified_by",
            "occurs_at",
            "has_condition",
            "has_reason",
        ],
        "entity_properties": {
            "Person": ["nationality", "occupation", "attribute"],
            "Event": ["date", "time", "location", "status", "reason"],
            "Object": ["color", "attribute"],
        },
        "relation_properties": {
            "developed": ["time_period", "context"],
            "drove down": ["manner", "condition"],
            "was postponed": ["reason"],
        },
    }

    text_to_process = "Muse is an English rock band formed in 1994 in Cambridge, England. The band was formed by Matthew Bellamy and Christopher Wolstenholme. The band's lead vocalist is Matthew Bellamy, who is known for his distinctive voice and songwriting. Muse has released several albums, including 'Absolution' and 'Black Holes and Revelations'. The band's music is known for its complex compositions and innovative sound."

    output = extractor.extract_triplets(text_to_process, existing_schema=my_schema)

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))


if __name__ == "__main__":
    main()
