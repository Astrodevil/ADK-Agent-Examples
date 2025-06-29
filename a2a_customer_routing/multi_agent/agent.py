import uuid
import logging
import os

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill, MessageSendParams, Part,
    TaskState, TextPart, SendMessageRequest, Message
)
from a2a.utils import new_agent_text_message, new_task
from a2a.client import A2AClient

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
import httpx


from .tools import resolve_query_fn, classify_fn, escalate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Agent and Model Definitions ---
def create_llm_model(model_name: str):
    """Factory function to create LLM models with consistent configuration."""
    api_key = os.getenv("NEBIUS_API_KEY")
    return LiteLlm(model=model_name, api_key=api_key, temperature=0.1)

llama_8b = create_llm_model("nebius/meta-llama/Meta-Llama-3.1-8B-Instruct")
qwen = create_llm_model("nebius/Qwen/Qwen3-235B-A22B")

intake_agent = LlmAgent(name="intake_agent", model=llama_8b, description="Classifies sentiment", instruction="Use the classify_fn tool. Return ONLY the classification result (positive, neutral, or negative).", tools=[classify_fn])
resolution_agent = LlmAgent(name="resolution_agent", model=qwen, description="Answers questions from a KB", instruction="Use resolve_query_fn. If the tool returns 'KB_ANSWER:', return only the text after it. Otherwise, say you don't have information.", tools=[resolve_query_fn])
escalation_agent = LlmAgent(name="escalation_agent", model=llama_8b, description="Escalates to humans", instruction="Use escalate_fn to forward the user's message to human support and return the tool's confirmation message.", tools=[escalate_fn])


# --- A2A Server Infrastructure ---
class ADKAgentExecutor(AgentExecutor):
    def __init__(self, agent, status_message="Processing...", artifact_name="response"):
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name
        self.runner = Runner(
            app_name=agent.name, agent=agent, artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(), memory_service=InMemoryMemoryService()
        )

    async def cancel(self, task_id: str) -> None: pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)
        try:
            await updater.update_status(TaskState.working, new_agent_text_message(self.status_message, task.contextId, task.id))
            session = await self.runner.session_service.create_session(app_name=self.agent.name, user_id="a2a_user", session_id=task.contextId)
            next_message = types.Content(role='user', parts=[types.Part.from_text(text=query)])
            final_answer = "The agent could not produce a final answer."
            max_turns = 5
            for i in range(max_turns):
                logger.info(f"Agent '{self.agent.name}' Turn {i+1}/{max_turns}")
                response_content = None
                async for event in self.runner.run_async(user_id="a2a_user", session_id=session.id, new_message=next_message):
                    if event.is_final_response() and event.content:
                        response_content = event.content
                        break
                if not response_content or not response_content.parts:
                    final_answer = "Agent produced an empty response."; break
                part = response_content.parts[0]
                if part.function_response:
                    tool_output = f"Tool '{part.function_response.name}' returned: {part.function_response.response}"
                    logger.info(f"Agent '{self.agent.name}' Tool call result: {tool_output[:200]}...")
                    next_message = types.Content(role='user', parts=[types.Part.from_text(text=tool_output)])
                    continue
                elif part.text is not None:
                    final_answer = part.text.strip()
                    logger.info(f"Agent '{self.agent.name}' final answer received."); break
                else:
                    final_answer = "Agent produced an unexpected response type."; break
            await updater.add_artifact([Part(root=TextPart(text=final_answer))], name=self.artifact_name)
            await updater.complete()
        except Exception as e:
            logger.error(f"Error in ADKAgentExecutor for agent {self.agent.name}: {e}", exc_info=True)
            await updater.update_status(TaskState.failed, new_agent_text_message(f"Error: {e!s}", task.contextId, task.id), final=True)

# --- A2A Agent Server Creation ---
def create_agent_a2a_server(agent, name, description, skills, host, port, status_message, artifact_name):
    agent_card = AgentCard(
        name=name, description=description, url=f"http://{host}:{port}", version="1.0.0",
        defaultInputModes=["text"], defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True), skills=skills
    )
    executor = ADKAgentExecutor(agent=agent, status_message=status_message, artifact_name=artifact_name)
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

# --- Individual Agent Servers ---
def create_intake_agent_server(host="127.0.0.1", port=10020):
    return create_agent_a2a_server(agent=intake_agent, name="Sentiment Agent", description="Analyzes message sentiment.", skills=[AgentSkill(id="classify_sentiment", name="Classify Sentiment", description="Determines message sentiment.", tags=["sentiment"])], host=host, port=port, status_message="Analyzing sentiment...", artifact_name="sentiment_result")
def create_resolution_agent_server(host="127.0.0.1", port=10021):
    return create_agent_a2a_server(agent=resolution_agent, name="KB Agent", description="Answers questions using a knowledge base.", skills=[AgentSkill(id="resolve_question", name="Resolve Question", description="Searches KB for answers.", tags=["knowledge", "support"])], host=host, port=port, status_message="Searching knowledge base...", artifact_name="kb_answer")
def create_escalation_agent_server(host="127.0.0.1", port=10022):
    return create_agent_a2a_server(agent=escalation_agent, name="Escalation Agent", description="Escalates issues to human support.", skills=[AgentSkill(id="escalate_issue", name="Escalate Issue", description="Forwards issues to humans.", tags=["escalation", "human"])], host=host, port=port, status_message="Escalating to human support...", artifact_name="escalation_result")


# --- Coordinator Agent & Client ---
class A2AToolClient:
    def __init__(self, default_timeout: float = 120.0):
        self._agent_info_cache: dict[str, dict[str, any] | None] = {}
        self.default_timeout = default_timeout
    def add_remote_agent(self, agent_url: str):
        normalized_url = agent_url.rstrip('/')
        if normalized_url not in self._agent_info_cache: self._agent_info_cache[normalized_url] = None
    async def create_task(self, agent_url: str, message: str) -> str:
        timeout_config = httpx.Timeout(self.default_timeout)
        async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
            agent_card_response = await httpx_client.get(f"{agent_url}/.well-known/agent.json")
            agent_card_response.raise_for_status()
            agent_card = AgentCard(**agent_card_response.json())
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            
            message_payload = Message(
                messageId=f"msg_{uuid.uuid4()}",
                role='user',
                parts=[TextPart(text=message)]
            )
            send_params = MessageSendParams(message=message_payload)
            request = SendMessageRequest(
                id=f"req_{uuid.uuid4()}",  
                params=send_params
            )

            response = await client.send_message(request)
            response_dict = response.model_dump(mode='json', exclude_none=True)
            if 'result' in response_dict and 'artifacts' in response_dict['result']:
                for artifact in response_dict['result']['artifacts']:
                    for part in artifact.get('parts', []):
                        if 'text' in part and part['text'].strip(): return part['text'].strip()
            return "Agent did not return a text artifact."

coordinator_a2a_client = A2AToolClient()

def create_coordinator_agent_with_registered_agents():
    return LlmAgent(
        name="support_coordinator", model=llama_8b, description="Routes user messages to other agents.",
        instruction="""You are an expert support coordinator. Your job is to orchestrate other agents to resolve a user's request.

Follow this exact workflow:
1.  **Analyze Sentiment:** Use the `create_task` tool to call the Intake Agent (at http://127.0.0.1:10020) with the user's original message. This tool will return a sentiment classification.
2.  **Route Request:**
    *   **If the result from the Intake Agent contains the word "negative"**, use `create_task` to call the Escalation Agent (at http://127.0.0.1:10022) with the user's original message.
    *   Otherwise (for "positive" or "neutral"), use `create_task` to call the Resolution Agent (at http://127.0.0.1:10021) with the user's original message.
3.  **Return Final Answer:** Your final answer must be ONLY the text returned by the chosen agent (Resolution or Escalation). Do not add any of your own commentary, summaries, or phrases like "The final answer is:".
""",
        tools=[coordinator_a2a_client.create_task]
    )


coordinator_agent = None

def create_coordinator_agent_server(host="127.0.0.1", port=10023):
    global coordinator_agent
    if coordinator_agent is None: raise ValueError("Coordinator agent not initialized.")
    return create_agent_a2a_server(agent=coordinator_agent, name="Support Coordinator", description="Orchestrates customer support.", skills=[AgentSkill(id="coordinate_support", name="Coordinate Support", description="Routes customer message to the right agent.", tags=["routing", "sentiment"])], host=host, port=port, status_message="Coordinating request...", artifact_name="support_response")