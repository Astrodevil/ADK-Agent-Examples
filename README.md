# ADK Agent Demos

This repository contains various agent demos built with Google's [ADK (Agent Development Kit)](https://google.github.io/adk-docs/), showcasing different patterns and capabilities for building AI agents.

## LLM Integration

All demos in this repository are powered by [Nebius AI](https://dub.sh/AIStudio) using open-source LLMs:

- **Meta-Llama-3.1-8B-Instruct** - Used in most agent implementations
- **Llama-3_1-Nemotron-Ultra-253B** - Used for advanced analysis in the Analyzer Agent

These models are integrated via [LiteLLM](https://github.com/BerriAI/litellm), which ADK supports for connecting to various model providers.

## Agent Demos

| Agent | Pattern | Description | Details |
|-------|---------|-------------|---------|
| [Analyzer Agent](./analyzer_agent/) | 5-agent sequential pipeline | AI trends analysis with multiple data sources | [README](./analyzer_agent/README.md) |
| [Email Agent](./email_adk_agent/) | Single agent with tool | Email integration with Resend API | [README](./email_adk_agent/README.md) |
| [Sequential Agent](./seq_adk_agent/) | 3-agent sequential pipeline | News aggregator combining IPL and AI news | [README](./seq_adk_agent/README.md) |
| [Multi-Tool Search](./multi_tool_search_agent/) | Root agent with delegation | Modular search with agent delegation | [README](./multi_tool_search_agent/README.md) |

For detailed information about each agent, please refer to the individual READMEs in their respective directories.

## Implementation Patterns Demonstrated

This repository demonstrates several important ADK patterns:

1. **Sequential Agent Chaining** - Multiple agents executing in sequence, with outputs becoming inputs for the next agent.
2. **Tool Integration** - Using external APIs and services as tools within agents.
3. **Agent Delegation** - A root agent delegating tasks to specialized sub-agents.
4. **Multi-Model Approach** - Using different LLM models for different tasks based on capabilities.
5. **Specialized Agents** - Agents designed for specific tasks like search, summary, analysis, etc.

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
# Example for analyzer agent
cp analyzer_agent/.env.example analyzer_agent/.env
```
Then edit the `.env` file to include your API keys.

## Usage

### Running Agents with ADK CLI

```bash
# Dev UI - Visual interface for testing and debugging agents
adk web

# Terminal - Run agents directly in the terminal
adk run analyzer_agent

# API Server - Create a local FastAPI server for API testing
adk api_server
```

### API Keys Required

You'll need to set up accounts and obtain API keys for:

| Service | Purpose | Required For |
|---------|---------|--------------|
| [Nebius AI](https://dub.sh/AIStudio) | LLM inference | All agents |
| [Exa](https://exa.ai/) | Web search | Most agents |
| [Tavily](https://tavily.com/) | Specialized search | Some agents |
| [Firecrawl](https://firecrawl.dev/) | Web scraping | Analyzer agent |
| [Resend](https://resend.com/) | Email sending | Email agent |

See each agent's README for specific requirements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 