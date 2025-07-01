# Multi-Agent Customer Query Routing and Resolution With Google ADK and A2A Protocol

A customer support system built with distributed AI agents using the A2A (Agent-to-Agent) protocol and Google's Agent Development Kit (ADK). The system intelligently routes customer inquiries based on sentiment analysis and provides automated responses through knowledge base search or human escalation.

## System Architecture

```
User Message → Coordinator Agent → Sentiment Analysis → Routing Decision
                                                     ↓
                                      ┌─────────────────────────────┐
                                      │                             │
                               Negative Sentiment            Positive/Neutral
                                      │                             │
                                      ↓                             ↓
                             Escalation Agent                Resolution Agent
                           (Human Handoff)                 (Knowledge Base Search)
                                      │                             │
                                      └─────────────────────────────┘
                                                     ↓
                                             Final Response to User
```

## Agent Overview

| Agent | Port | Purpose | Technology |
|-------|------|---------|------------|
| **Coordinator** | 10023 | Orchestrates workflow and routing | Llama 3.1 8B |
| **Intake** | 10020 | Sentiment analysis (positive/neutral/negative) | Llama 3.1 8B |
| **Resolution** | 10021 | Knowledge base search and answers | Qwen3-30B-A3B |
| **Escalation** | 10022 | Human support escalation | Llama 3.1 8B |


## Features

- **Intelligent Routing**: Automatic sentiment-based message routing
- **Knowledge Base Integration**: Semantic search through FAQ database
- **Human Escalation**: Automatic escalation for negative sentiment cases
- **Interactive UI**: Streamlit-based web interface for easy interaction

## Technology Stack & Integrations

### A2A (Agent-to-Agent) Protocol
The system leverages the [A2A protocol](https://github.com/a2aproject/A2A) to enable communication between distributed agents. Each agent operates independently while maintaining coordinated workflows through standardized message passing.

### Google ADK (Agent Development Kit) Integration
[Google's ADK](https://google.github.io/adk-docs/) is used to build the intelligent agents (Coordinator, Intake, and Escalation) in this system.

### Nebius AI Studio
[Nebius AI Studio](https://studio.nebius.com/) the underlying AI model infrastructure for all agents in the system
- **Integration Methods**:
  - **ADK Pipeline**: Uses LiteLLM integration to connect Google ADK agents to Nebius models
  - **[LlamaIndex](https://www.llamaindex.ai/)  Pipeline**: Direct integration for knowledge base operations and semantic search capabilities
- **Models Used via Nebius AI API**:
  - **Llama 3.1 8B**: Powers the Coordinator, Intake, and Escalation agents for sentiment analysis and workflow orchestration
  - **Qwen/Qwen3-30B-A3B**: Handles knowledge base query resolution and complex question answering in the Resolution agent
  - **Qwen/Qwen3-Embedding-8B**: Generates semantic embeddings for document indexing and similarity search



## Prerequisites

- Python 3.8+
- API access to Nebius AI models
- Required Python packages (see requirements.txt)

## Project Structure

```
a2a_customer_routing/
├── knowledge_base/
│   └── swiftcart_kb.json
├── multi_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── run_agents.py
│   └── tools.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Astrodevil/ADK-Agent-Examples.git
   cd ADK-Agent-Examples/a2a_customer_routing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

4. **Run the multi-agent system**
   ```bash
   python -m a2a_customer_routing.multi_agent.run_agents
   ```

5. **Launch the Streamlit interface** (in a separate terminal)
   ```bash
   streamlit run a2a_customer_routing/streamlit_app.py
   ```

## Workflow Examples

### Positive/Neutral Sentiment Flow
```
User: "How do I track my order?"
↓
Coordinator → Intake Agent → "neutral"
↓
Coordinator → Resolution Agent → KB Search
↓
Response: "You can track your order by..."
```

### Negative Sentiment Flow
```
User: "This product is terrible! I want my money back!"
↓
Coordinator → Intake Agent → "negative"
↓
Coordinator → Escalation Agent → Human Escalation
↓
Response: "I understand your frustration. A human agent will contact you shortly..."
```

## System Components

### Core Classes

- **`KB`**: Knowledge base management with LlamaIndex integration
- **`ADKAgentExecutor`**: Wrapper for Google ADK agents to work with A2A protocol
- **`A2AToolClient`**: Client for agent-to-agent communication

### Agent Tools

- **`resolve_query_fn`**: Searches knowledge base for answers
- **`classify_fn`**: Performs sentiment analysis
- **`escalate_fn`**: Handles human escalation logging

## Usage

1. Start the multi-agent system using the run command
2. Open the Streamlit interface in your browser
3. Enter customer queries to see the intelligent routing in action
4. Monitor the console output to see agent-to-agent communication

