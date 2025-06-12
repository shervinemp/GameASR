"""
Template tool call definitions for LLM.

This module provides template tool calls that can be used in prompts.
"""


def create_tool_call_template(tools_list):
    """
    Create a tool call template for use in prompts.

    Args:
        tools_list (list): List of tool definitions to include in the template.

    Returns:
        str: Template string with tool definitions.
    """
    template = "<tool> "
    for tool in tools_list:
        template += f'{{"name": "{tool["name"]}", "description": "{tool["description"]}", "parameters": {str(tool["parameters"])}}} </tool>'
    return template


# Example usage
if __name__ == "__main__":
    example_tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "format"],
            },
        },
        {
            "name": "get_stock_price",
            "description": "Get the current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock symbol"}
                },
                "required": ["symbol"],
            },
        },
    ]

    print(create_tool_call_template(example_tools))
