from strands import Agent, tool
from strands_tools import calculator, current_time


# 1. Define a custom tool
@tool*
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


# 2. Create the agent using built-in and custom tools
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


if __name__ == "__main__":
    # 3. Ask the agent a question that uses the tools
    message = """
You have three tasks:

1. What time is it right now?
2. Calculate (3111696 / 74088).
3. In the text below, how many times does the word 'prompt' appear?

Text:
"In this course, we practice prompt engineering.
Good prompts lead to better model behavior, and prompt patterns
are a key part of generative AI development."
"""
    result = strandsagent(message)

    # result is an AgentResult; print the final answer
    print("\n=== AGENT RESPONSE ===")
    print(result)
