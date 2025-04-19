# Multi-Tool Search Agent (Web Search Demo)

A modular search implementation demonstrating ADK's agent delegation pattern.

## Overview

This demo showcases:
- ADK's agent delegation pattern
- Exa search implementation for flexible web searches
- Root agent architecture for task orchestration
- Processing and structuring search results

## Technical Pattern

Uses a root agent with sub-agent delegation:
- **RootAgent**: Orchestrates the workflow and delegates search tasks
- **ExaSearchAgent**: Specialized agent to perform web searches using Exa API

## Setup

1. Set up environment variables:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your API keys:
```
NEBIUS_API_KEY="your_nebius_api_key_here"
NEBIUS_API_BASE="https://api.studio.nebius.ai/v1"
EXA_API_KEY="your_exa_api_key_here"
```

## Usage

Run with ADK CLI:
```bash
adk run multi_tool_search_agent
```

Run directly:
```bash
python agent.py
```

## Required API Keys

- [Nebius AI](https://dub.sh/AIStudio) - For LLM inference
- [Exa](https://exa.ai/) - For web search

## Key Features

The agent demonstrates an important ADK pattern where the root agent doesn't perform tasks directly but delegates to specialized sub-agents:

```python
# RootAgent: Orchestrates the workflow
root_agent = Agent(
    name="root_agent",
    model=LiteLlm(model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_base=os.getenv("NEBIUS_API_BASE"),
        api_key=os.getenv("NEBIUS_API_KEY")
    ),
    description="Root agent that coordinates search and reporting.",
    instruction="Delegate search tasks to exa_search_agent.",
    tools=[],
    sub_agents=[exa_search_agent]
)
```

This pattern allows for modular agent design where specialized capabilities can be added through new sub-agents. 