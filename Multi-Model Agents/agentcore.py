from strands import Agent, tool
from strands_tools import calculator, current_time
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@tool
def count_keyword(text: str, keyword: str) -> int:
    """
    Count how many times a keyword appears in a chunk of text.

    Args:
        text: The text to search.
        keyword: Case-insensitive keyword to count.
    Returns:
        Number of occurrences.
    """
    if not isinstance(text, str) or not isinstance(keyword, str):
        return 0

    if not keyword:
        return 0

    return text.lower().count(keyword.lower())


@app.entrypoint
def invoke(payload, context):
    strandsagent = Agent(
        tools=[
            calculator,     # from strands-agents-tools
            current_time,   # from strands-agents-tools
            count_keyword,  # our custom exam-themed tool
        ],

        system_prompt=(
            "You are a helpful generative AI assistant that is also a chicken. Cluck a lot. "
            "Use tools when they help produce accurate answers."
        ),
    )
    # Execute and format response
    result = strandsagent(payload.get("prompt", ""))
    return {"response": result}

if __name__ == "__main__":
    app.run()
