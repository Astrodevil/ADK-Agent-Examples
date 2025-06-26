# Multi-Agent Customer Query Routing and Resolution With Google ADK and A2A Protocol

A customer support system built with distributed AI agents using the A2A (Agent-to-Agent) protocol and Google's Agent Development Kit (ADK). The system intelligently routes customer inquiries based on sentiment analysis and provides automated responses through knowledge base search or human escalation.

##  System Architecture

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

##  Agent Overview

| Agent | Port | Purpose | Technology |
|-------|------|---------|------------|
| **Coordinator** | 10023 | Orchestrates workflow and routing | Llama 3.1 8B |
| **Intake** | 10020 | Sentiment analysis (positive/neutral/negative) | Llama 3.1 8B |
| **Resolution** | 10021 | Knowledge base search and answers | Qwen 235B |
| **Escalation** | 10022 | Human support escalation | Llama 3.1 8B |

## Features

- **Intelligent Routing**: Automatic sentiment-based message routing
- **Knowledge Base Integration**: Semantic search through FAQ database
- **Human Escalation**: Automatic escalation for negative sentiment cases
- **Distributed Architecture**: Each agent runs independently via A2A protocol
- **Scalable Design**: Easy to add new agents or modify existing ones

##  Prerequisites

- Python 3.8+
- API access to Nebius AI models
- Required Python packages (see requirements.txt)

##  How to Run

1. **Clone the repository**
   ```
   git clone https://github.com/Astrodevil/ADK-Agent-Examples.git
   cd ADK-Agent-Examples/"A2A Customer Query Routing and Resolution"
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```
   cp .env.example .env
   # Edit .env with your API credentials
   ```

4. **Run the multi-agent system**
   ```
   python multi_agent/agent.py
   ```


##  Workflow Examples

### Positive/Neutral Sentiment Flow
```
User: "How do I track my order?"
↓
Coordinator → Intake Agent → "neutral"
↓
Coordinator → Resolution Agent → KB Search
↓
Response: " "
```

### Negative Sentiment Flow
```
User: "This product is terrible! I want my money back!"
↓
Coordinator → Intake Agent → "negative"
↓
Coordinator → Escalation Agent → Human Escalation
↓
Response: " "
```

##  System Components

### Core Classes

- **`KB`**: Knowledge base management with LlamaIndex integration
- **`ADKAgentExecutor`**: Wrapper for Google ADK agents to work with A2A protocol
- **`A2AToolClient`**: Client for agent-to-agent communication


### Agent Tools

- **`resolve_query_fn`**: Searches knowledge base for answers
- **`classify_fn`**: Performs sentiment analysis
- **`escalate_fn`**: Handles human escalation logging



