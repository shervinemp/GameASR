import json
import sys
from typing import Dict, Any

from ..llm.session import Session

from ..common.utils import setup_logging


class KnowledgeGraphTripletExtractor:
    prompt: str = [
        "Your task: Extract all factual knowledge from the text as Subject-Predicate-Object (S-P-O) triplets.",
        "Output MUST be a JSON array of objects, each representing an atomic triplet.",
        "Each triplet object requires 'subject', 'predicate', and 'object' keys.",
        "**Subject/Object structure:** {'name': 'core_entity', 'properties': {'attribute': 'value'}}",
        "- 'name' is the core, singular entity/concept.",
        "- 'properties' (optional dict) holds **distinct attributes** (e.g., 'color', 'type', 'nationality', 'attribute'). Do NOT re-pack main facts or redundant info here.",
        "**Predicate structure:** {'name': 'relationship_phrase', 'properties': {'context': 'value'}}",
        "- 'name' is a concise, atomic relationship phrase (ideally 1-3 words).",
        "- 'properties' (optional dict) holds **contextual details** (e.g., 'manner', 'time', 'condition', 'reason') modifying the relationship.",
        "**Example Output (Strictly adhere):**",
        "```json",
        "[",
        '  {"subject": {"name": "Albert Einstein", "properties": {"nationality": "German-born", "type": "theoretical physicist"}}, "predicate": {"name": "developed"}, "object": {"name": "theory of relativity", "properties": {"significance": "groundbreaking"}}},',
        '  {"subject": {"name": "The car"}, "predicate": {"name": "drove down", "properties": {"manner": "quickly"}}, "object": {"name": "the street"}}',
        "]",
        "```",
        "**Extraction Guidelines:**",
        "- Replace pronouns with actual entities.",
        "- **Strictly** extract as separate, atomic triplets. Break down and correctly categorize/label fields. **Avoid over-packing.**",
        "- Capture all verifiable info. Properties are for **additional attributes only**, not the core S-P-O fact.",
        "- Ensure logical consistency and coherence; facts must be verifiable from text.",
        "- Use clear, concise, and atomic language for predicates and properties. Avoid multi-verb predicates.",
        "- Avoid redundant/overlapping triplets. Each triplet must be new info. Do NOT repeat S/P/O names in properties.",
        "- If optional 'properties' or any key within is empty/irrelevant, **omit that field entirely**.",
        "- If schema guidance follows, prioritize those types/keys. Only create new ones if no suitable option exists.",
        "- Observe all complex interactions and relationships for accuracy and completeness.",
    ]

    def __init__(self):
        """
        Initializes the extractor with an LLM model instance.
        """
        self.session = Session()

    def _generate_triplet_prompt(
        self, text: str, existing_schema: Dict = None, retrieval: bool = False
    ) -> str:
        """
        Generates the detailed prompt for the LLM to extract knowledge graph triplets.
        The prompt explicitly instructs the LLM on the desired output format,
        including how to represent adjectives, adverbs, and contextual information as properties.
        """
        prompt_parts = self.prompt + [
            (
                '- Put the "?" wildcard for unknown entities as an indication to look up the answer.'
                if retrieval
                else ""
            )
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
        self,
        text: str,
        existing_schema: Dict = None,
        retrieval: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts knowledge graph triplets from the given text using the LLM
        """
        prompt = self._generate_triplet_prompt(text, existing_schema, retrieval)
        self.session.conversation.messages.clear()
        response = "".join(self.session(prompt, tool_choice="knowledge_graph_triplets"))
        return response


class KnowledgeGraphEntityExtractor(KnowledgeGraphTripletExtractor):
    prompt: str = [
        "Your task is to analyze a user's question/query to extract critical information for querying a knowledge graph.",
        "The goal is to identify all relevant entities, relations, properties, and keywords present in the query.",
        "Be comprehensive and precise in your extraction, capturing all elements that could contribute to a strong match.",
        "The arguments for this tool MUST be a JSON object with the following structure:",
        "**Tool Arguments Schema:**",
        "```json",
        "{",
        '  "query_entities": ["list of specific entities mentioned in the query"],',
        '  "query_relations": ["list of relationships or predicates implied or explicitly stated in the query"],',
        '  "query_properties": {"property_key": "property_value", "condition": "value"},',
        '  "keywords": ["list of other important concepts or terms"]',
        "}",
        "```",
        "**Guidelines for Extraction:**",
        "- Extract information **ONLY** from the provided user query.",
        "- For `query_entities`: Identify all specific people, organizations, locations, events, concepts, or objects the user is asking about.",
        "- For `query_relations`: Identify verbs or phrases that imply connections between entities (e.g., 'founded by', 'located in', 'is a', 'works at').",
        "- For `query_properties`: Extract key-value pairs representing specific attributes or conditions (e.g., 'year': '1994', 'status': 'active', 'color': 'red'). These modify entities or relations.",
        "- For `keywords`: Include any other significant terms or concepts that don't fit into entities, relations, or properties but are crucial for understanding the query's context.",
        "- Ensure all extracted elements are concise and directly relevant to the query.",
        "- Avoid redundancy across fields. If a concept is best represented as an entity, don't also put it in keywords unless it serves a distinct purpose.",
        "- If a field (e.g., `query_properties`) has no relevant information, provide an empty dictionary `{}` or empty list `[]` as appropriate, but the keys themselves must always be present.",
        "Task: Analyze the user's question/query to extract all critical, factual information for querying a knowledge graph.",
        "The goal is to identify all relevant entities, relations, properties, and keywords present in the query.",
        "This extracted information is crucial for an algorithm to find the most relevant subgraph in a knowledge graph, by maximizing matching score with these elements. Therefore, precision and comprehensiveness are paramount.",
        "The response MUST be a JSON object with the following structure:",
        "```json",
        "{",
        '  "query_entities": [""list of specific named entities or clearly identifiable roles/concepts from the query"],',
        '  "query_relations": ["list of relationships or predicates explicitly stated or strongly implied by the query\'s verbs/phrases"],',
        '  "query_properties": {"property_key": "property_value"},',
        '  "keywords": ["list of other significant concepts or terms that **DO NOT** fit into entities, relations, or properties, and are crucial for context"]',
        "}",
        "```",
        "Note: The `query_properties` object should contain key-value pairs where the key is the desired property type and the value is what's being queried or specified.",
        "**Guidelines for Extraction:**",
        "- Extract information **ONLY** from the provided user query. Do NOT add external knowledge.",
        "- **Resolve pronouns and indefinite references (e.g., 'that', 'he', 'it') to their most specific possible entity or role from the query text.** If a specific name cannot be inferred, provide the descriptive role.",
        "- For `query_entities`: Identify all concrete named entities, or the most precise descriptive phrases for specific entities being discussed.",
        "- For `query_relations`: Focus on explicit or strongly implied verbs/phrases indicating relationships. **Include all distinct relationships.**",
        "- For `query_properties`: Extract explicit attributes or conditions. For specific information requested, represent them as properties.",
        "- For `keywords`: Include broad conceptual terms that aid search but are *not* directly entities, relations, or specific properties already extracted. **Avoid redundancy: if an item is an entity, relation, or property, do NOT also list it as a keyword.**",
        "- Ensure all extracted elements are concise, unique, and directly relevant to the query's core intent.",
        "- **Crucial:** If any field (`query_entities`, `query_relations`, `query_properties`, `keywords`) has no relevant information, provide it as an empty list `[]` or empty dictionary `{}` as appropriate. **All four keys must always be present in the final JSON object.**",
    ]


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

    extractor = KnowledgeGraphTripletExtractor()

    text_to_process = "Muse is an English rock band formed in 1994 in Cambridge, England. The band was formed by Matthew Bellamy and Christopher Wolstenholme. The band's lead vocalist is Matthew Bellamy, who is known for his distinctive voice and songwriting. Muse has released several albums, including 'Absolution' and 'Black Holes and Revelations'. The band's music is known for its complex compositions and innovative sound."

    output = extractor.extract_triplets(text_to_process, existing_schema=my_schema)

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))

    text_to_process = "Do you know the Uncle of that lead actor in the movie 'The Matrix'? He had a business at some point. What was its name and address?"

    output = extractor.extract_triplets(
        text_to_process, existing_schema=my_schema, retrieval=True
    )

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))

    entity_extractor = KnowledgeGraphEntityExtractor()

    output = entity_extractor.extract_triplets(
        text_to_process, existing_schema=my_schema
    )

    extracted_triplets = json.loads(output)
    print(json.dumps(extracted_triplets, indent=2))


if __name__ == "__main__":
    main()
