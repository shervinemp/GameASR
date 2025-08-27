import json
import sys
from typing import Dict, Any

from ..llm.model import LLM
from ..llm.session import Session

from ..common.utils import setup_logging


class KnowledgeExtractor:
    prompt: str = [
        "Your task: Extract all factual knowledge from the text as "
        "Subject-Predicate-Object (S-P-O) triplets.",
        "Output MUST be a JSON array of objects, each representing an atomic triplet.",
        "Each triplet object requires 'subject', 'predicate', and 'object' keys.",
        "**Subject/Object structure:** {'name': 'core_entity', 'properties': "
        "{'attribute': 'value'}}",
        "- 'name' is the core, singular entity/concept.",
        "- 'properties' (optional dict) holds **distinct attributes** "
        "(e.g., 'color', 'type', 'nationality', 'attribute'). "
        "Do NOT re-pack main facts or redundant info here.",
        "**Predicate structure:** {'name': 'relationship_phrase', 'properties': "
        "{'context': 'value'}}",
        "- 'name' is a concise, atomic relationship phrase (ideally 1-3 words).",
        "- 'properties' (optional dict) holds **contextual details** "
        "(e.g., 'manner', 'time', 'condition', 'reason') modifying the relationship.",
        "**Example Output (Strictly adhere):**",
        "```json",
        "[",
        '  {"subject": {"name": "Albert Einstein", "properties": {"nationality": '
        '"German-born", "type": "theoretical physicist"}}, "predicate": '
        '{"name": "developed"}, "object": {"name": "theory of relativity", '
        '"properties": {"significance": "groundbreaking"}}},',
        '  {"subject": {"name": "The car"}, "predicate": {"name": "drove down", '
        '"properties": {"manner": "quickly"}}, "object": {"name": "the street"}}',
        "]",
        "```",
        "**Extraction Guidelines:**",
        "- Replace pronouns with actual entities.",
        "- **Strictly** extract as separate, atomic triplets. Break down and "
        "correctly categorize/label fields. **Avoid over-packing.**",
        "- Capture all verifiable info. Properties are for **additional attributes "
        "only**, not the core S-P-O fact.",
        "- Ensure logical consistency and coherence; facts must be verifiable from text.",
        "- Use clear, concise, and atomic language for predicates and properties. "
        "Avoid multi-verb predicates.",
        "- Avoid redundant/overlapping triplets. Each triplet must be new info. "
        "Do NOT repeat S/P/O names in properties.",
        "- If optional 'properties' or any key within is empty/irrelevant, "
        "**omit that field entirely**.",
        "- If schema guidance follows, prioritize those types/keys. Only create "
        "new ones if no suitable option exists.",
        "- Observe all complex interactions and relationships for accuracy and completeness.",
    ]

    def __init__(self, llm: LLM = None):
        """
        Initializes the extractor with an LLM model instance.
        """
        self.session = Session(llm)
        self.session.conversation._cutoff_idx = -1

    def _generate_triplet_prompt(
        self, text: str, existing_schema: Dict = None, retrieval: bool = False
    ) -> str:
        """
        Generates the detailed prompt for the LLM to extract knowledge graph triplets.
        The prompt explicitly instructs the LLM on the desired output format, including how to represent adjectives, adverbs, and contextual information as properties.
        """
        prompt_parts = self.prompt + [
            (
                '- Put the "?" wildcard for unknown entities as an indication to '
                "look up the answer."
                if retrieval
                else ""
            )
        ]
        if existing_schema:
            prompt_parts.append(
                "\nConsider the following predefined schema elements:"
            )
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
                        for k, v in existing_schema[
                            "entity_properties"
                        ].items()
                    ]
                )
                prompt_parts.append(
                    f"- **Common Entity Properties (guidance):** {props}"
                )
            if existing_schema.get("relation_properties"):
                props = ", ".join(
                    [
                        f"{k}: {', '.join(v)}"
                        for k, v in existing_schema[
                            "relation_properties"
                        ].items()
                    ]
                )
                prompt_parts.append(
                    f"- **Common Relation Properties (guidance):** {props}"
                )

        prompt_parts.append("\nHere is the text to analyze:")
        prompt_parts.append(f"```text\n{text}\n```")
        prompt_parts.append(
            "\nPlease provide the JSON array of triplets, "
            "conforming strictly to the specified schema."
        )

        return "\n".join(prompt_parts)

    def extract_triplets(
        self,
        text: str,
        existing_schema: Dict = None,
        retrieval: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts knowledge graph triplets from the given text using the LLM.
        """
        prompt = self._generate_triplet_prompt(
            text, existing_schema, retrieval
        )
        self.session.conversation.messages.clear()
        response = "".join(
            self.session(prompt, tool_choice="knowledge_graph_triplets")
        )
        return response


def main():
    setup_logging("DEBUG", stream=sys.stdout)

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

    extractor = KnowledgeExtractor()

    text_to_process = (
        "Muse is an English rock band formed in 1994 in Cambridge, England. "
        "The band was formed by Matthew Bellamy and Christopher Wolstenholme. "
        "The band's lead vocalist is Matthew Bellamy, who is known for his "
        "distinctive voice and songwriting. Muse has released several albums, "
        "including 'Absolution' and 'Black Holes and Revelations'. The band's "
        "music is known for its complex compositions and innovative sound."
    )

    output = extractor.extract_triplets(
        text_to_process, existing_schema=my_schema
    )

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))

    text_to_process = (
        "Do you know the Uncle of that lead actor in the movie 'The Matrix'? "
        "He was named after him. What was his name and address?"
    )

    output = extractor.extract_triplets(
        text_to_process, existing_schema=my_schema, retrieval=True
    )

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))


if __name__ == "__main__":
    main()
