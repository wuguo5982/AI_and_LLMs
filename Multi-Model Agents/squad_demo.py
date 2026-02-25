import asyncio

# 1. Import your Strands SDK agent from agent.py
from strandsagent import strandsagent as strands_sdk_agent  # the chicken calculator/time/keyword agent

# 2. Agent Squad imports
from agent_squad.orchestrator import AgentSquad
from agent_squad.agents import (
    Agent,
    AgentOptions,
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
)
from agent_squad.types import ConversationMessage, ParticipantRole


# 3. Custom adapter that makes your Strands agent look like an Agent Squad agent
class StrandsAdapterOptions(AgentOptions):
    """No extra fields for now; we just reuse AgentOptions."""
    pass


class StrandsAdapterAgent(Agent):
    def __init__(self, options: StrandsAdapterOptions):
        super().__init__(options)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history,          # list[ConversationMessage]
        additional_params=None,
    ) -> ConversationMessage:
        """
        Call the Strands agent synchronously and wrap its result
        into Agent Squad's ConversationMessage format.
        """
        # Call your Strands SDK agent (synchronous call)
        result = strands_sdk_agent(input_text)

        # AgentResult has a readable __str__()
        final_text = str(result)

        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": final_text}],
        )


# 4. Instantiate the adapter for your Strands agent
strands_agent = StrandsAdapterAgent(
    StrandsAdapterOptions(
        name="Calculator/Time/Keyword Chicken Agent",
        description=(
            "Uses calculator, current time, and keyword counting tools. "
            "The assistant is a chicken and clucks a lot."
        ),
    )
)


# 5. General-purpose Bedrock LLM agent for non-tool-ish questions
bedrock_llm = BedrockLLMAgent(
    BedrockLLMAgentOptions(
        name="General Bedrock Assistant",
        description=(
            "Handles all general questions that do not require calculator, "
            "current time, or keyword counting."
        ),
        # NOTE: models come and go; you may need to update this to a newer
        # one if it throws an error. The latest Amazon Nova model would 
        # probably be the safest bet.
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        # IMPORTANT: turn off streaming for this demo
        streaming=False,
    )
)


# 6. Agent Squad orchestrator
orchestrator = AgentSquad()
orchestrator.add_agent(strands_agent)
orchestrator.add_agent(bedrock_llm)


def extract_text_from_message(msg: ConversationMessage) -> str:
    """
    Safely extract human-readable text from a ConversationMessage.
    """
    if isinstance(msg, ConversationMessage):
        pieces = []
        for block in msg.content or []:
            if isinstance(block, dict) and "text" in block:
                pieces.append(block["text"])
        if pieces:
            return "".join(pieces)
        return str(msg)
    return str(msg)


# 7. Simple REPL to demo routing
async def chat_loop():
    print("Agent Squad demo (Strands via adapter + Bedrock). Type 'quit' to exit.\n")
    user_id = "demo-user"
    session_id = "demo-session"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in {"quit", "exit"}:
            break

        # NOTE: no streaming kwarg here; we use simple, non-streaming responses
        response = await orchestrator.route_request(
            user_input,
            user_id,
            session_id,
        )

        print("\n[ROUTING]")
        print(f"  Chosen agent: {response.metadata.agent_name}")

        # response.output is expected to be a ConversationMessage
        text = extract_text_from_message(response.output)
        print("\nAssistant:", text)


if __name__ == "__main__":
    asyncio.run(chat_loop())
