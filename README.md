# ADK Agent Demos

This repository contains various agent demos built with Google's [ADK (Agent Development Kit)](https://google.github.io/adk-docs/), showcasing different patterns and capabilities for building AI agents.

## Projects

### Analyzer Agent (AI Trends Analysis Pipeline)

A comprehensive AI analysis pipeline that:
- Fetches the latest AI news from Twitter/X using Exa search
- Retrieves AI benchmarks and analysis using Tavily search
- Scrapes and processes data from Nebius AI Studio using Firecrawl
- Synthesizes and structures this information into a comprehensive analysis
- Analyzes AI trends and provides specific Nebius model recommendations

**Technical Pattern:** Uses a 5-agent sequential pipeline with specialized agents for different data sources, summary generation, and in-depth analysis.

### Email Agent (Communication Demo)

A simple agent demonstrating email integration:
- Uses Resend API for sending emails
- Shows how to structure and format email content
- Demonstrates integration of external APIs with ADK agents
- Provides a foundation for building notification systems

**Technical Pattern:** Single agent with a specialized tool for email communication.

### Sequential Agent (News Aggregator)

A news aggregation agent that:
- Chains multiple search tools in sequence
- Uses Exa to gather IPL cricket news from April 2025
- Uses Tavily to collect AI news from Hacker News
- Combines and formats results into a structured, engaging summary

**Technical Pattern:** 3-agent sequential pipeline with aggregation of multiple data sources and summarization.

### Multi-Tool Search Agent (Web Search Demo)

A modular search implementation that:
- Demonstrates ADK's agent delegation pattern
- Uses Exa search to perform flexible web searches
- Implements a root agent architecture for task orchestration
- Shows how to process and structure search results

**Technical Pattern:** Uses a root agent with sub-agent delegation pattern.

## Key Differences Between Sequential Agents

This repository contains two different sequential agent implementations that serve different purposes:

1. **Analyzer Agent:** A complex 5-agent pipeline focused on AI trend analysis with multiple data sources and deep analysis.

2. **Sequential Agent:** A simpler 3-agent pipeline focused on news aggregation from different topics (IPL cricket and AI news).

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Astrodevil/Agent-Cookbook.git
cd Agent-Cookbook
```

2. Install ADK:
```bash
pip install google-adk
```

3. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up your environment variables for the agent you want to run:
```bash
# For analyzer agent
cp analyzer_agent/.env.example analyzer_agent/.env

# For email agent
cp email_adk_agent/.env.example email_adk_agent/.env

# For sequential agent
cp seq_adk_agent/.env.example seq_adk_agent/.env

# For multi-tool search agent
cp multi_tool_search_agent/.env.example multi_tool_search_agent/.env
```
Then edit the `.env` files to include your API keys.

## Usage

### Running Agents with ADK CLI

ADK provides multiple ways to run and interact with your agents:

```bash
# Dev UI - Visual interface for testing and debugging agents
adk web

# Terminal - Run agents directly in the terminal
adk run analyzer_agent
adk run email_adk_agent
adk run seq_adk_agent
adk run multi_tool_search_agent

# API Server - Create a local FastAPI server for API testing
adk api_server
```

### Running Agents Directly

You can also run each agent with Python directly:

```bash
# Analyzer Agent
cd analyzer_agent
python agent.py

# Email Agent
cd email_adk_agent
python agent.py

# Sequential Agent
cd seq_adk_agent
python agent.py

# Multi-Tool Search Agent
cd multi_tool_search_agent
python agent.py
```

## API Keys Required

You'll need to set up accounts and obtain API keys for:

| Service | Purpose | Required For |
|---------|---------|--------------|
| [Nebius AI](https://studio.nebius.ai/) | LLM inference | All agents |
| [Exa](https://exa.ai/) | AI & news search | Analyzer, Sequential, Multi-Tool agents |
| [Tavily](https://tavily.com/) | Specialized search | Analyzer and Sequential agents |
| [Firecrawl](https://firecrawl.dev/) | Web scraping | Analyzer agent only |
| [Resend](https://resend.com/) | Email sending | Email agent only |

## Implementation Patterns Demonstrated

This repository demonstrates several important ADK patterns:

1. **Sequential Agent Chaining** - Multiple agents executing in sequence, with each agent's output becoming input for the next.
2. **Tool Integration** - Using external APIs and services as tools within agents.
3. **Agent Delegation** - A root agent delegating tasks to specialized sub-agents.
4. **Multi-Model Approach** - Using different LLM models for different tasks based on capabilities.
5. **Specialized Agents** - Agents designed for specific tasks like search, summary, analysis, etc.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 