from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from exa_py import Exa
from dotenv import load_dotenv
import os

load_dotenv()
api_base = os.getenv("NEBIUS_API_BASE")
api_key = os.getenv("NEBIUS_API_KEY")


def exa_search(query: str, num_results: int = 10) -> dict:
    """
    Performs a web search using the Exa API and retrieves detailed contents.

    Args:
        query (str): The search query.
        num_results (int): Number of search results to return.

    Returns:
        dict: A dictionary containing the search results or an error message.
    """
    try:
        # Ensure num_results is an integer
        num_results = int(num_results)

        results = Exa(api_key=os.getenv("EXA_API_KEY")).search_and_contents(
            query=query,
            num_results=num_results,
            text={"max_characters": 200},
            highlights={"highlights_per_url": 5, "num_sentences": 5},
            use_autoprompt=True
        )
        return {
            "status": "success",
            "results": [result.__dict__ for result in results.results]
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}



# ExaSearchAgent: Fetches search results
exa_search_agent = Agent(
    name="exa_search_agent",
    model=LiteLlm(model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",  # or your specific model
        api_base=api_base,
        api_key=api_key
    ),
    description="Agent that performs web searches using the Exa API.",
    instruction="Use the exa_search tool to perform web searches.",
    tools=[exa_search]
)


# RootAgent: Orchestrates the workflow
root_agent = Agent(
    name="root_agent",
    model=LiteLlm(model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",  # or your specific model
        api_base=os.getenv("NEBIUS_API_BASE"),
        api_key=os.getenv("NEBIUS_API_KEY")
    ),
    description="Root agent that coordinates search and reporting.",
    instruction="Delegate search tasks to exa_search_agent.",
    tools=[],
    sub_agents=[exa_search_agent]
)

