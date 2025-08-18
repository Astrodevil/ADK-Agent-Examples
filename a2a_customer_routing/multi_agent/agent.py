import uuid
import logging
import os
import asyncio
from typing import Any
import httpx 
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill, Task, TransportProtocol
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm

from .tools import resolve_query_fn, classify_fn, escalate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Agent and Model Definitions ---
def create_llm_model(model_name: str):
    api_key = os.getenv("NEBIUS_API_KEY")
    return LiteLlm(model=model_name, api_key=api_key, temperature=0.1)

llama_8b = create_llm_model("nebius/meta-llama/Meta-Llama-3.1-8B-Instruct")
qwen = create_llm_model("nebius/Qwen/Qwen3-30B-A3B")

intake_agent = LlmAgent(name="intake_agent", model=llama_8b, description="Classifies sentiment", instruction="Use the classify_fn tool. Return ONLY the classification result (positive, neutral, or negative).", tools=[classify_fn])
resolution_agent = LlmAgent(name="resolution_agent", model=qwen, description="Answers questions from a KB", instruction="Use resolve_query_fn. If the tool returns 'KB_ANSWER:', return only the text after it. Otherwise, say you don't have information.", tools=[resolve_query_fn])
escalation_agent = LlmAgent(name="escalation_agent", model=llama_8b, description="Escalates to humans", instruction="Use escalate_fn to forward the user's message to human support and return the tool's confirmation message.", tools=[escalate_fn])


# --- A2A Server Infrastructure ---
def create_agent_a2a_server(agent: LlmAgent, agent_card: AgentCard):
    runner = Runner(
        app_name=agent.name, agent=agent, artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(), memory_service=InMemoryMemoryService()
    )
    config = A2aAgentExecutorConfig()
    executor = A2aAgentExecutor(runner=runner, config=config)
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

def create_intake_agent_server(host="127.0.0.1", port=10020):
    card = AgentCard(name="Sentiment Agent", description="Analyzes message sentiment.", url=f"http://{host}:{port}", version="1.0.0", defaultInputModes=["text"], defaultOutputModes=["text"], capabilities=AgentCapabilities(streaming=True), skills=[AgentSkill(id="classify_sentiment", name="Classify Sentiment", description="Determines message sentiment.", tags=["sentiment"])], preferred_transport=TransportProtocol.jsonrpc)
    return create_agent_a2a_server(agent=intake_agent, agent_card=card)

def create_resolution_agent_server(host="127.0.0.1", port=10021):
    card = AgentCard(name="KB Agent", description="Answers questions using a knowledge base.", url=f"http://{host}:{port}", version="1.0.0", defaultInputModes=["text"], defaultOutputModes=["text"], capabilities=AgentCapabilities(streaming=True), skills=[AgentSkill(id="resolve_question", name="Resolve Question", description="Searches KB for answers.", tags=["knowledge", "support"])], preferred_transport=TransportProtocol.jsonrpc)
    return create_agent_a2a_server(agent=resolution_agent, agent_card=card)

def create_escalation_agent_server(host="127.0.0.1", port=10022):
    card = AgentCard(name="Escalation Agent", description="Escalates issues to human support.", url=f"http://{host}:{port}", version="1.0.0", defaultInputModes=["text"], defaultOutputModes=["text"], capabilities=AgentCapabilities(streaming=True), skills=[AgentSkill(id="escalate_issue", name="Escalate Issue", description="Forwards issues to humans.", tags=["escalation", "human"])], preferred_transport=TransportProtocol.jsonrpc)
    return create_agent_a2a_server(agent=escalation_agent, agent_card=card)


# --- Coordinator Agent & Client ---
class A2AToolClient:
    def __init__(self, default_timeout: float = 120.0):
        self._agent_info_cache: dict[str, Any | None] = {}
        self.default_timeout = default_timeout

    async def create_task(self, agent_url: str, message: str) -> str:
        timeout_config = httpx.Timeout(self.default_timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
            agent_card_response = await httpx_client.get(f"{agent_url}{AGENT_CARD_WELL_KNOWN_PATH}")
            agent_card_response.raise_for_status()
            agent_card = AgentCard(**agent_card_response.json())

            config = ClientConfig(httpx_client=httpx_client)
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            message_obj = create_text_message_object(content=message)
            final_response = "Agent did not return a valid response."

            async for response in client.send_message(message_obj):
                if isinstance(response, tuple) and len(response) > 0:
                    task: Task = response[0]
                    if task.artifacts:
                        try:
                            text_response = task.artifacts[0].parts[0].root.text
                            if text_response:
                                final_response = text_response.strip()
                                break
                        except (AttributeError, IndexError):
                            logger.warning(f"Could not extract text from task artifact for {agent_url}")
                            final_response = f"Agent at {agent_url} returned an unreadable response."
                else:
                    logger.warning(f"Received an unexpected response format from {agent_url}: {response}")
            
            return final_response

coordinator_a2a_client = A2AToolClient()

def create_coordinator_agent_with_registered_agents():
    return LlmAgent(
        name="support_coordinator", model=llama_8b, description="Routes user messages to other agents.",
        instruction="""You are an expert support coordinator. Your job is to orchestrate other agents to resolve a user's request.

Follow this exact workflow step-by-step:
1.  **Analyze Sentiment:** Use the `create_task` tool to call the Intake Agent (at http://127.0.0.1:10020) with the user's original message. The tool will return a sentiment classification.
2.  **Route Request:**
    *   If the result from the Intake Agent contains the word "negative", use the `create_task` tool to call the Escalation Agent (at http://127.0.0.1:10022) with the user's original message.
    *   Otherwise (for "positive" or "neutral"), use the `create_task` tool to call the Resolution Agent (at http://127.0.0.1:10021) with the user's original message.
3.  **Finalize and Respond:** The tool used in the previous step will return the final answer. Your final job is to output that exact text as your own final answer. Do not add any of your own commentary, summaries, or phrases like "The final answer is:". Just return the text you received.
""",
        tools=[coordinator_a2a_client.create_task]
    )

coordinator_agent = None

def create_coordinator_agent_server(host="127.0.0.1", port=10023):
    global coordinator_agent
    if coordinator_agent is None: raise ValueError("Coordinator agent not initialized.")
    card = AgentCard(name="Support Coordinator", description="Orchestrates customer support.", url=f"http://{host}:{port}", version="1.0.0", defaultInputModes=["text"], defaultOutputModes=["text"], capabilities=AgentCapabilities(streaming=True), skills=[AgentSkill(id="coordinate_support", name="Coordinate Support", description="Routes customer message to the right agent.", tags=["routing", "sentiment"])], preferred_transport=TransportProtocol.jsonrpc)
    return create_agent_a2a_server(agent=coordinator_agent, agent_card=card)