from dotenv import load_dotenv
import os

# Load keys
load_dotenv()
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.models.lite_llm import LiteLlm

from exa_py import Exa
from tavily import TavilyClient

# Init Nebius model
nebius_model = LiteLlm(
    model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_base=os.getenv("NEBIUS_API_BASE"),
    api_key=os.getenv("NEBIUS_API_KEY")
)

# Tool 1: Exa
def exa_search_ipl(_: str) -> dict:
    results = Exa(api_key=os.getenv("EXA_API_KEY")).search_and_contents(
        query="IPL 2025 latest updates April 2025",
        num_results=5,
        type="auto",
        highlights={"highlights_per_url": 2},
        text={"max_characters": 500}
    )
    return {
        "type": "exa",
        "results": [r.__dict__ for r in results.results]
    }

# Tool 2: Tavily
def tavily_search_hn(_: str) -> dict:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query="AI news from Hacker News",
        search_depth="basic",
        time_range="day",
        include_domains=["https://news.ycombinator.com/"]
    )
    return {
        "type": "tavily",
        "results": response.get("results", [])
    }

# Agents 1
exa_agent = Agent(
    name="ExaAgent",
    model=nebius_model,
    description="Fetches IPL 2025 news using Exa.",
    instruction="Use the exa_search_ipl tool to fetch the latest IPL 2025 news from April 2025.",
    tools=[exa_search_ipl],
    output_key="exa_news"
)

# Agents 2
tavily_agent = Agent(
    name="TavilyAgent",
    model=nebius_model,
    description="Fetches AI news from Hacker News using Tavily.",
    instruction="Use the tavily_search_hn tool to retrieve AI-related news from Hacker News posted in the last 24 hours.",
    tools=[tavily_search_hn],
    output_key="tavily_news"
)

# Agents 3
summary_agent = Agent(
    name="SummaryAgent",
    model=nebius_model,
    description="Summarizes Exa and Tavily results in a fun, structured format.",
    instruction="""
You are a summarizer. Create a clean and fun summary using info from:
- 'exa_news': IPL 2025 updates
- 'tavily_news': AI news from Hacker News

Show a clean, structured summary (can use tables or emojis 🏏🤖📢). Make it easy to scan and engaging.
""",
    tools=[],
    output_key="final_summary"
)

# Agent chain
pipeline = SequentialAgent(
    name="NewsPipelineAgent",
    sub_agents=[exa_agent, tavily_agent, summary_agent]
)

# Runner setup
APP_NAME = "multi_news_pipeline"
USER_ID = "vscode_user"
SESSION_ID = "vscode_session"

session_service = InMemorySessionService()
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

# Run the pipeline
def run_news_pipeline():
    content = types.Content(role="user", parts=[types.Part(text="Start the news roundup")])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event.is_final_response():
            print("\n📢 Final Summary:\n")
            print(event.content.parts[0].text)

if __name__ == "__main__":
    run_news_pipeline()

root_agent = pipeline