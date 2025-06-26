import os
import json
import asyncio
import threading
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
import logging

# A2A protocol imports for agent-to-agent communication
import uvicorn
import httpx
import requests
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessageSendParams,
    Part,
    TaskState,
    TextPart,
    SendMessageRequest,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.client import A2AClient

# LlamaIndex imports for knowledge base functionality
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM as LlamaIndexLiteLLM_LLM

# Google ADK imports for agent creation
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from litellm import completion
from google.genai import types

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Knowledge Base class for handling customer support queries
class KB:
    def __init__(self):
        # Load knowledge base from JSON file
        try:
            path = Path("a2a_customer_routing/knowledge_base/swiftcart_kb.json")
            data = json.loads(path.read_text())
        except FileNotFoundError:
            logger.error("knowledge_base/swiftcart_kb.json not found. Please create it.")
            data = {}

        # Convert JSON data to LlamaIndex documents
        docs = [
            Document(
                text=f"Q: {faq['question']}\nA: {faq['answer']}",
                metadata={"category": cat}
            )
            for cat, faqs in data.items()
            for faq in faqs
            if isinstance(faq, dict) and 'question' in faq and 'answer' in faq
        ]

        # Add placeholder if no documents found
        if not docs:
            logger.warning("No documents loaded into Knowledge Base. Queries will likely fail.")
            docs.append(Document(text="Placeholder document."))

        # Split documents into chunks for better retrieval
        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(docs)

        # Get API credentials for LLM services
        api_base = os.getenv("NEBIUS_API_BASE")
        api_key = os.getenv("NEBIUS_API_KEY")

        # Create vector index for semantic search
        self.index = VectorStoreIndex(
            nodes,
            embed_model=LiteLLMEmbedding(
                model_name="nebius/BAAI/bge-multilingual-gemma2",
                api_base=api_base,
                api_key=api_key
            )
        )

        # Create query engine for answering questions
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize",
            similarity_top_k=3,
            llm=LlamaIndexLiteLLM_LLM(
                model="nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
                api_base=api_base,
                api_key=api_key
            )
        )
        logger.info(f"Knowledge base initialized with {len(docs)} documents")

# Initialize global knowledge base instance
kb = KB()

# Tool functions for agent capabilities
def resolve_query_fn(question: str) -> str:
    """Resolves question from the KB."""
    resp = kb.query_engine.query(question.strip())
    
    # Check if we got a meaningful answer
    has_answer = (
        resp.source_nodes and 
        len(resp.source_nodes) > 0 and 
        resp.response and 
        resp.response.strip() and
        "don't know" not in resp.response.lower() and
        "cannot" not in resp.response.lower()
    )
    
    if has_answer:
        return f"KB_ANSWER: {resp.response.strip()}"
    else:
        return "NO_KB_INFO: No information found in knowledge base for this question"

def classify_fn(message: str) -> str:
    """Classifies sentiment (positive/neutral/negative)."""
    prompt = """Classify the sentiment of the following message into one of [positive, neutral, negative].

Examples:
- "I love this product! It's amazing." → positive
- "It's okay, neither good nor bad." → neutral  
- "This is terrible, I'm really unhappy." → negative

Message: "%s"
Sentiment:""" % message.strip()

    # Use LLM to classify sentiment
    resp = completion(
        model="nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        api_base=os.getenv("NEBIUS_API_BASE"),
        api_key=os.getenv("NEBIUS_API_KEY"),
        max_tokens=10,
        temperature=0.1
    )
    
    sentiment = resp.choices[0].message.content.strip().lower()
    
    # Validate and fallback to keyword matching if needed
    valid_sentiments = ["positive", "neutral", "negative"]
    if sentiment not in valid_sentiments:
        if any(word in sentiment for word in ["good", "great", "happy", "love"]):
            sentiment = "positive"
        elif any(word in sentiment for word in ["bad", "terrible", "angry", "hate"]):
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
    return sentiment

def escalate_fn(message: str) -> str:
    """Escalates message to human support."""
    print(f"[ESCALATION] Forwarding to human support: {message.strip()}")
    logger.info(f"Escalated message: {message.strip()}")
    
    return "Your message has been escalated to human support. We will contact you shortly via phone call."

# Create LLM models for different agents
def create_llm_model(model_name: str, name: str):
    """Factory function to create LLM models with consistent configuration."""
    api_base = os.getenv("NEBIUS_API_BASE")
    api_key = os.getenv("NEBIUS_API_KEY")
    return LiteLlm(model=model_name, api_base=api_base, api_key=api_key, temprature=0.1)

# Initialize LLM models
llama_8b = create_llm_model("nebius/meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama 8B")
qwen = create_llm_model("nebius/Qwen/Qwen3-235B-A22B", "Qwen")

# Create specialized agents for different tasks
intake_agent = LlmAgent(name="intake_agent", model=llama_8b, description="Classifies sentiment of user messages", instruction="You are the intake agent responsible for sentiment classification.\n\nYour task:\n1. Use the classify_fn tool to determine the sentiment of the user's message\n2. Return ONLY the classification result (positive, neutral, or negative)\n\nAlways call the classify_fn tool with the user's message and return only the sentiment result.", tools=[classify_fn], output_key="sentiment_result")
resolution_agent = LlmAgent(name="resolution_agent", model=qwen, description="Searches knowledge base to answer user questions", instruction="You are a knowledge base assistant.\n\nINSTRUCTIONS:\n1. Call resolve_query_fn with the user's question\n2. Based on the tool response:\n   - If response starts with \"KB_ANSWER:\", return only the text AFTER \"KB_ANSWER: \"\n   - If response starts with \"NO_KB_INFO:\", return \"I don't have information about that topic\"\n   - If response starts with \"ERROR:\", return \"I'm experiencing a technical issue, please try again\"\n\nDo not add explanations, reasoning, or additional text. Return only the final answer.", tools=[resolve_query_fn], output_key="kb_answer")
escalation_agent = LlmAgent(name="escalation_agent", model=llama_8b, description="Handles escalation of negative sentiment cases to human support", instruction="You are the escalation agent that handles negative sentiment cases.\n\nYour task:\n1. Use escalate_fn to forward the user's message to human support\n2. Return the escalation confirmation message\n\nAlways be empathetic and professional.", tools=[escalate_fn], output_key="escalation_result")

# Custom A2A executor to wrap ADK agents for A2A protocol
class ADKAgentExecutor(AgentExecutor):
    def __init__(self, agent, status_message="Processing request...", artifact_name="response"):
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name
        # Initialize ADK runner with required services
        self.runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def cancel(self, task_id: str) -> None:
        # Placeholder for task cancellation logic
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute agent task and handle tool calls properly."""
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Update task status to show processing
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(self.status_message, task.contextId, task.id)
            )

            # Create agent session for conversation
            session = await self.runner.session_service.create_session(
                app_name=self.agent.name,
                user_id="a2a_user",
                state={},
                session_id=task.contextId,
            )

            # Start conversation with user query
            next_message = types.Content(role='user', parts=[types.Part.from_text(text=query)])
            final_answer = "The agent could not produce a final answer."
            max_turns = 5  # Prevent infinite loops
            
            # Handle multi-turn conversation for tool calls
            for i in range(max_turns):
                logger.info(f"Agent '{self.agent.name}' Turn {i+1}/{max_turns} | Input: {next_message.parts[0].text[:200]}...")
                
                response_content = None
                # Execute agent and get response
                async for event in self.runner.run_async(
                    user_id="a2a_user", session_id=session.id, new_message=next_message
                ):
                    if event.is_final_response() and event.content:
                        response_content = event.content
                        break
                
                if not response_content or not response_content.parts:
                    final_answer = "Agent produced an empty response."
                    break

                part = response_content.parts[0]
                
                # Handle tool function responses
                if part.function_response:
                    tool_output = f"Tool '{part.function_response.name}' returned the following result: {part.function_response.response}"
                    logger.info(f"Agent '{self.agent.name}' Turn {i+1}: Tool call completed. Result: {tool_output[:200]}...")
                    next_message = types.Content(role='user', parts=[types.Part.from_text(text=tool_output)])
                    continue
                
                # Handle final text response
                elif part.text is not None:
                    final_answer = part.text.strip()
                    logger.info(f"Agent '{self.agent.name}' Turn {i+1}: Final text answer received. Breaking loop.")
                    break
                
                else:
                    logger.warning(f"Agent '{self.agent.name}' produced an unexpected response part. Breaking loop.")
                    final_answer = "Agent produced an unexpected response type."
                    break

            # Send final response as artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=final_answer))], name=self.artifact_name
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"Error in ADKAgentExecutor for agent {self.agent.name}: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.contextId, task.id),
                final=True
            )

# Factory function to create A2A servers for any agent
def create_agent_a2a_server(agent, name, description, skills, host="127.0.0.1", port=10020, 
                          status_message="Processing request...", artifact_name="response"):
    """Create an A2A server wrapper for any ADK agent."""
    # Define agent capabilities
    capabilities = AgentCapabilities(streaming=True)
    
    # Create agent card with metadata
    agent_card = AgentCard(name=name, description=description, url=f"http://{host}:{port}", version="1.0.0", defaultInputModes=["text", "text/plain"], defaultOutputModes=["text", "text/plain"], capabilities=capabilities, skills=skills)
    
    # Create executor and request handler
    executor = ADKAgentExecutor(agent=agent, status_message=status_message, artifact_name=artifact_name)
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    
    # Return A2A application
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

# Server creation functions for each specialized agent
def create_intake_agent_server(host="127.0.0.1", port=10020):
    """Create A2A server for sentiment classification agent."""
    return create_agent_a2a_server(agent=intake_agent, name="Sentiment Classification Agent", description="Analyzes sentiment of user messages (positive/neutral/negative)", skills=[AgentSkill(id="classify_sentiment", name="Classify Sentiment", description="Determines if a message has positive, neutral, or negative sentiment", tags=["sentiment", "classification", "analysis"], examples=["I love this product!", "This is terrible service", "It's okay, nothing special"])], host=host, port=port, status_message="Analyzing sentiment...", artifact_name="sentiment_result")

def create_resolution_agent_server(host="127.0.0.1", port=10021):
    """Create A2A server for knowledge base resolution agent."""
    return create_agent_a2a_server(agent=resolution_agent, name="Knowledge Base Resolution Agent", description="Searches knowledge base to answer customer questions", skills=[AgentSkill(id="resolve_question", name="Resolve Question", description="Searches knowledge base for answers to customer questions", tags=["knowledge", "search", "support", "answers"], examples=["How do I return a product?", "What's your shipping policy?", "How do I cancel my order?"])], host=host, port=port, status_message="Searching knowledge base...", artifact_name="kb_answer")

def create_escalation_agent_server(host="127.0.0.1", port=10022):
    """Create A2A server for escalation agent."""
    return create_agent_a2a_server(agent=escalation_agent, name="Escalation Agent", description="Escalates negative sentiment cases to human support", skills=[AgentSkill(id="escalate_issue", name="Escalate Issue", description="Forwards customer issues to human support team", tags=["escalation", "support", "human"], examples=["I'm very unhappy with this service", "This product is completely broken", "I want to speak to a manager"])], host=host, port=port, status_message="Escalating to human support...", artifact_name="escalation_result")

# A2A client for communicating with remote agents
class A2AToolClient:
    def __init__(self, default_timeout: float = 120.0):
        # Cache for agent metadata to avoid repeated requests
        self._agent_info_cache: dict[str, dict[str, any] | None] = {}
        self.default_timeout = default_timeout

    def add_remote_agent(self, agent_url: str):
        """Register a remote agent URL for communication."""
        normalized_url = agent_url.rstrip('/')
        if normalized_url not in self._agent_info_cache:
            self._agent_info_cache[normalized_url] = None

    def list_remote_agents(self) -> dict[str, any]:
        """Get information about all registered remote agents."""
        if not self._agent_info_cache: return {}
        remote_agents_info = {}
        
        # Fetch agent info for each registered agent
        for remote_connection in self._agent_info_cache:
            if self._agent_info_cache[remote_connection] is not None:
                # Use cached data
                remote_agents_info[remote_connection] = self._agent_info_cache[remote_connection]
            else:
                try:
                    # Fetch and cache agent metadata
                    agent_info = requests.get(f"{remote_connection}/.well-known/agent.json", timeout=10)
                    agent_info.raise_for_status()
                    agent_data = agent_info.json()
                    self._agent_info_cache[remote_connection] = agent_data
                    remote_agents_info[remote_connection] = agent_data
                except Exception as e:
                    logger.warning(f"Failed to fetch agent info from {remote_connection}: {e}")
        return remote_agents_info

    async def create_task(self, agent_url: str, message: str) -> str:
        """Send a message to a remote agent and get response."""
        # Configure HTTP client with timeout
        timeout_config = httpx.Timeout(timeout=self.default_timeout, connect=10.0, read=self.default_timeout, write=10.0, pool=5.0)
        
        async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
            # Get or fetch agent card
            if agent_url in self._agent_info_cache and self._agent_info_cache[agent_url] is not None:
                agent_card_data = self._agent_info_cache[agent_url]
            else:
                agent_card_response = await httpx_client.get(f"{agent_url}/.well-known/agent.json")
                agent_card_response.raise_for_status()
                agent_card_data = agent_card_response.json()
                self._agent_info_cache[agent_url] = agent_card_data
            
            # Create A2A client and send message
            agent_card = AgentCard(**agent_card_data)
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Build message payload
            send_message_payload = {'message': {'role': 'user', 'parts': [{'kind': 'text', 'text': message}], 'messageId': uuid.uuid4().hex}}
            request = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(**send_message_payload))
            
            # Send request and parse response
            response = await client.send_message(request)
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            # Extract text from artifacts
            if 'result' in response_dict and 'artifacts' in response_dict['result']:
                for artifact in response_dict['result']['artifacts']:
                    if 'parts' in artifact:
                        for part in artifact['parts']:
                            if 'text' in part and part['text'].strip(): return part['text'].strip()
            
            # Fallback to status message
            if 'result' in response_dict and 'status' in response_dict['result'] and 'message' in response_dict['result']['status'] and 'parts' in response_dict['result']['status']['message']:
                for part in response_dict['result']['status']['message']['parts']:
                    if 'text' in part and part['text'].strip(): return part['text'].strip()
            
            # Return raw response if no text found
            return json.dumps(response_dict, indent=2)

# Initialize A2A clients for different purposes
coordinator_a2a_client = A2AToolClient()  # For coordinator agent to communicate with others
test_a2a_client = A2AToolClient()  # For testing the system

def create_coordinator_agent_with_registered_agents():
    """Create the main coordinator agent that orchestrates the support workflow."""
    return LlmAgent(
        name="support_coordinator",
        model=llama_8b,
        description="Main coordinator that routes user messages using A2A protocol",
        instruction="""You are an expert support coordinator. Your job is to orchestrate other agents to resolve a user's request.

You have access to the following remote agents:
- Intake Agent (for sentiment analysis): "http://127.0.0.1:10020"
- Resolution Agent (for answering questions): "http://127.0.0.1:10021"
- Escalation Agent (for handling angry users): "http://127.0.0.1:10022"

Follow this exact workflow:
1.  **Analyze Sentiment:** Use the `create_task` tool to call the Intake Agent (at http://127.0.0.1:10020) with the user's original message. The tool will return "positive", "neutral", or "negative".
2.  **Route Request:**
    *   If the sentiment is "negative", use `create_task` to call the Escalation Agent (at http://127.0.0.1:10022) with the user's original message.
    *   If the sentiment is "positive" or "neutral", use `create_task` to call the Resolution Agent (at http://127.0.0.1:10021) with the user's original message.
3.  **Return Final Answer:** The final response from the chosen agent (Resolution or Escalation) is your final answer. Return ONLY that text. Do not add any of your own commentary, summaries, or explanations.
""",
        tools=[coordinator_a2a_client.create_task]
    )

# Global coordinator agent variable (initialized after other agents start)
global coordinator_agent
coordinator_agent = None

def create_coordinator_agent_server(host="127.0.0.1", port=10023):
    """Create A2A server for the coordinator agent."""
    global coordinator_agent
    if coordinator_agent is None:
        raise ValueError("Coordinator agent not initialized. Call start_all_agents() first.")
    return create_agent_a2a_server(agent=coordinator_agent, name="Support Coordinator", description="Orchestrates customer support using sentiment analysis and routing", skills=[AgentSkill(id="coordinate_support", name="Coordinate Support", description="Analyzes customer message sentiment and routes to appropriate support agent", tags=["coordination", "support", "routing", "sentiment"], examples=["I need help with my order", "This product is terrible!", "How do I use this feature?"])], host=host, port=port, status_message="Coordinating support request...", artifact_name="support_response")

# Server management utilities
import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops in Jupyter notebooks
servers = []  # Keep track of running servers

async def run_server_notebook(create_agent_function, port):
    """Run a server in async context."""
    print(f"🚀 Starting agent on port {port}...")
    app = create_agent_function()
    config = uvicorn.Config(app.build(), host="127.0.0.1", port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    servers.append(server)
    await server.serve()

def run_agent_in_background(create_agent_function, port, name):
    """Run an agent server in a background thread."""
    def run() -> None:
        # Create new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_server_notebook(create_agent_function, port))
        except Exception as e:
            logger.error(f"{name} server error: {e}", exc_info=True)
    
    # Start daemon thread
    thread = threading.Thread(target=run, daemon=True, name=f"{name}-Thread")
    thread.start()
    return thread

def wait_for_agents(agent_urls, timeout=45):
    """Wait for all agents to be ready by checking their health endpoints."""
    start_time = time.monotonic()
    ready_agents = set()
    print(f"Waiting for {len(agent_urls)} agents to be ready...")
    
    while time.monotonic() - start_time < timeout:
        for url in agent_urls:
            if url in ready_agents:
                continue
            try:
                # Check agent health via metadata endpoint
                response = requests.get(f"{url}/.well-known/agent.json", timeout=1)
                if response.status_code == 200:
                    print(f"  ✅ Agent at {url} is ready.")
                    ready_agents.add(url)
            except requests.ConnectionError:
                pass  # Agent not ready yet
            except Exception as e:
                logger.warning(f"  ⚠️ Error checking agent at {url}: {e}")
        
        # Check if all agents are ready
        if len(ready_agents) == len(agent_urls):
            print("🎉 All support agents are ready!")
            return True
        time.sleep(1)
    
    print(f"❌ Timed out waiting for agents. Ready: {list(ready_agents)}")
    return False

def start_all_agents():
    """Start all support agents and the coordinator in the correct order."""
    global coordinator_agent
    
    print("Starting support agent servers...\n")
    
    # Define support agents to start first
    support_agents_to_start = {
        "Intake Agent": (create_intake_agent_server, 10020),
        "Resolution Agent": (create_resolution_agent_server, 10021),
        "Escalation Agent": (create_escalation_agent_server, 10022),
    }
    
    # Start support agent servers
    threads = {}
    for name, (create_fn, port) in support_agents_to_start.items():
        threads[name] = run_agent_in_background(create_fn, port, name)

    # Wait for support agents to be ready
    support_agent_urls = [f"http://127.0.0.1:{port}" for _, port in support_agents_to_start.values()]
    if not wait_for_agents(support_agent_urls):
        raise RuntimeError("Not all support agents started successfully. Check logs for errors.")

    # Register support agents with A2A clients
    print("\nRegistering agents with clients...")
    for url in support_agent_urls:
        coordinator_a2a_client.add_remote_agent(url)
        test_a2a_client.add_remote_agent(url)

    # Create and start coordinator agent
    print("Creating coordinator agent...")
    coordinator_agent = create_coordinator_agent_with_registered_agents()
    
    print("Starting coordinator agent server...")
    threads["Coordinator Agent"] = run_agent_in_background(create_coordinator_agent_server, 10023, "Coordinator Agent")
    
    # Wait for coordinator to be ready
    if not wait_for_agents(["http://127.0.0.1:10023"]):
        raise RuntimeError("Coordinator agent failed to start.")

    # Register coordinator with test client
    test_a2a_client.add_remote_agent("http://127.0.0.1:10023")
    
    print("\n✅ All A2A agents are running!")
    print("   - Intake Agent: http://127.0.0.1:10020")
    print("   - Resolution Agent: http://127.0.0.1:10021")
    print("   - Escalation Agent: http://127.0.0.1:10022")
    print("   - Coordinator Agent: http://127.0.0.1:10023")
    
    return threads

async def test_coordinator():
    """Test the complete coordinator workflow with different sentiment scenarios."""
    print("\n\n=== TESTING COORDINATOR AGENT WORKFLOW ===")
    
    # Test positive/neutral sentiment routing
    print("\n--- Test 1: Positive/Neutral Sentiment (should route to Resolution Agent) ---")
    positive_query = "What payment methods do you accept?"
    print(f"USER: {positive_query}")
    result1 = await test_a2a_client.create_task("http://127.0.0.1:10023", positive_query)
    print(f"COORDINATOR RESPONSE: {result1}")
    
    # Test negative sentiment routing  
    print("\n--- Test 2: Negative Sentiment (should route to Escalation Agent) ---")
    negative_query = "I am experiencing a very bad time using your product! I want a refund now!"
    print(f"USER: {negative_query}")
    result2 = await test_a2a_client.create_task("http://127.0.0.1:10023", negative_query)
    print(f"COORDINATOR RESPONSE: {result2}")

# Main execution entry point
if __name__ == "__main__":
    # Start all agents and run tests
    threads = start_all_agents()
    asyncio.run(test_coordinator())