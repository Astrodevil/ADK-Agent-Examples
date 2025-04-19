# Sequential Agent (News Aggregator)

A news aggregation agent that combines IPL cricket news and AI news from Hacker News.

## Overview

This demo showcases a sequential agent that:
- Chains multiple search tools in sequence
- Uses Exa to gather IPL cricket news from April 2025
- Uses Tavily to collect AI news from Hacker News
- Combines and formats results into a structured, engaging summary

## Technical Pattern

Uses a 3-agent sequential pipeline:
1. **ExaAgent**: Fetches IPL 2025 news
2. **TavilyAgent**: Retrieves AI news from Hacker News
3. **SummaryAgent**: Combines both sources into a structured summary

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
TAVILY_API_KEY="your_tavily_api_key_here"
```

## Usage

Run with ADK CLI:
```bash
# Terminal - Run directly in the terminal
adk run seq_adk_agent

# Dev UI - Visual interface for testing and debugging
adk web
```

## Required API Keys

- [Nebius AI](https://dub.sh/AIStudio) - For LLM inference
- [Exa](https://exa.ai/) - For IPL news search
- [Tavily](https://tavily.com/) - For AI news search

## Customization

You can modify the search queries in the tool functions to search for different topics:

```python
# For Exa search
results = Exa(api_key=os.getenv("EXA_API_KEY")).search_and_contents(
    query="Your custom search query here",
    # Other parameters...
)

# For Tavily search
response = client.search(
    query="Your custom search query here",
    # Other parameters...
)
``` 