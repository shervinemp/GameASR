"""
Offline Knowledge Extraction Script.

This script uses Google's langextract library to extract structured information
from unstructured text files. It is designed to be used as a developer tool
for pre-populating a knowledge graph with high-quality data using powerful
language models.

Usage:
    python -m scripts.extract_knowledge --input-file <path_to_text> --output-file <path_to_jsonl> --model-id <model_name>

Example:
    python -m scripts.extract_knowledge \\
        --input-file data/sample_text.txt \\
        --output-file data/extracted_triplets.jsonl \\
        --model-id "gemini-1.5-pro" \\
        --api-key-env "GEMINI_API_KEY"
"""

import argparse
import os
import textwrap
from dotenv import load_dotenv
import langextract as lx


def main():
    """Main function to run the knowledge extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract structured knowledge from text using langextract."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gemini-1.5-flash",
        help="The ID of the language model to use (e.g., 'gemini-1.5-pro', 'gpt-4o').",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="LANGEXTRACT_API_KEY",
        help="The name of the environment variable holding the API key.",
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key not found. Please set the {args.api_key_env} environment variable."
        )

    print(f"Starting knowledge extraction from '{args.input_file}'...")
    print(f"Using model: {args.model_id}")

    # 1. Define the prompt and extraction rules
    # This prompt is designed to extract S-P-O like triplets.
    prompt = textwrap.dedent("""\
        Extract all factual knowledge from the text as Subject-Predicate-Object triplets.
        The 'extraction_class' should be one of 'subject', 'predicate', or 'object'.
        Provide meaningful attributes for each entity to add context.
        Ensure that each logical fact is represented as a distinct set of extractions that can be linked.
        """)

    # 2. Provide high-quality examples to guide the model
    # This example shows how to structure the output for S-P-O triplets.
    examples = [
        lx.data.ExampleData(
            text="Albert Einstein developed the theory of relativity in his Annus Mirabilis papers.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="subject",
                    extraction_text="Albert Einstein",
                    attributes={"type": "person"},
                ),
                lx.data.Extraction(
                    extraction_class="predicate",
                    extraction_text="developed",
                    attributes={"context": "in his Annus Mirabilis papers"},
                ),
                lx.data.Extraction(
                    extraction_class="object",
                    extraction_text="the theory of relativity",
                    attributes={"type": "concept"},
                ),
            ],
        )
    ]

    # 3. Read the input text
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_text = f.read()

    # 4. Run the extraction
    print("Running extraction with langextract...")
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=args.model_id,
        api_key=api_key,
        max_workers=10,  # Use parallel processing for speed
    )

    # 5. Save the results
    output_dir = os.path.dirname(args.output_file)
    output_name = os.path.basename(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving results to '{args.output_file}'...")
    lx.io.save_annotated_documents(
        [result], output_name=output_name, output_dir=output_dir
    )

    print("Extraction complete.")
    print(
        f"To visualize the results, you can add the following to a Python script:\n"
        f"import langextract as lx\n"
        f"html_content = lx.visualize('{args.output_file}')\n"
        f"with open('visualization.html', 'w') as f:\n"
        f"    f.write(html_content)\n"
    )


if __name__ == "__main__":
    main()
