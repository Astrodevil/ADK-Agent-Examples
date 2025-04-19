# ADK Agent Demos

This repository contains various agent demos built with Google's ADK (Agent Development Kit).

## Projects

### Analyzer Agent

An AI analysis pipeline agent that:
- Fetches the latest AI news from Twitter/X using Exa search
- Retrieves AI benchmarks and analysis using Tavily search
- Scrapes and processes data from Nebius AI Studio using Firecrawl
- Generates a structured summary and analysis of AI trends, statistics, and potential recommendations

### Email Agent

An agent that demonstrates email functionality:
- Uses Resend API for sending emails
- Shows how to integrate email capabilities into an ADK agent

### Sequential Agent

A sequential agent that:
- Shows how to chain multiple tools together
- Uses Exa and Tavily search APIs to gather information
- Demonstrates the sequential agent pattern in ADK

### Multi-Tool Search Agent

A simple agent that:
- Demonstrates how to use multiple search tools
- Uses Exa search to gather information
- Shows how to process and structure search results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ADK2.git
cd ADK2
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables for the agent you want to run:
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

To run each agent:

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

## API Keys

You'll need to set up accounts and obtain API keys for:
- [Nebius AI](https://studio.nebius.ai/) - For LLM inference (required for all agents)
- [Exa](https://exa.ai/) - For AI news search (required for analyzer, sequential, and multi-tool agents)
- [Tavily](https://tavily.ai/) - For search (required for analyzer and sequential agents)
- [Firecrawl](https://firecrawl.dev/) - For web scraping (required for analyzer agent)
- [Resend](https://resend.com/) - For email functionality (required for email agent)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 