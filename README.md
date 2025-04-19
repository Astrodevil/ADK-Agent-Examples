# ADK Agent Demos

This repository contains various agent demos built with Google's ADK (Agent Development Kit).

## Projects

### Analyzer Agent

An AI analysis pipeline agent that:
- Fetches the latest AI news from Twitter/X using Exa search
- Retrieves AI benchmarks and analysis using Tavily search
- Scrapes and processes data from Nebius AI Studio using Firecrawl
- Generates a structured summary and analysis of AI trends, statistics, and potential recommendations

### Setup

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

4. Set up your environment variables:
```bash
cp analyzer_agent/.env.example analyzer_agent/.env
```
Then edit the `.env` file to include your API keys.

### Usage

To run the AI Analysis agent:
```bash
cd analyzer_agent
python agent.py
```

## API Keys

You'll need to set up accounts and obtain API keys for:
- [Nebius AI](https://studio.nebius.ai/) - For LLM inference
- [Exa](https://exa.ai/) - For AI news search
- [Tavily](https://tavily.ai/) - For search
- [Firecrawl](https://firecrawl.dev/) - For web scraping

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 