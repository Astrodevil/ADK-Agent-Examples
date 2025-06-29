import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from litellm import completion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# Knowledge Base class for handling customer support queries
class KB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KB, cls).__new__(cls)
            
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        logger.info("Initializing Knowledge Base...")
        try:
            
            kb_path = Path(__file__).resolve().parent.parent / "knowledge_base" / "swiftcart_kb.json"
            data = json.loads(kb_path.read_text())
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found at {kb_path}. Please ensure it exists.")
            data = {}

        docs = [
            Document(text=f"Q: {faq['question']}\nA: {faq['answer']}", metadata={"category": cat})
            for cat, faqs in data.items() for faq in faqs
            if isinstance(faq, dict) and 'question' in faq and 'answer' in faq
        ]

        if not docs:
            logger.warning("No documents loaded into Knowledge Base. Queries will likely fail.")
            docs.append(Document(text="Placeholder document."))

        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(docs)
        api_key = os.getenv("NEBIUS_API_KEY")

        self.index = VectorStoreIndex(
            nodes, embed_model=NebiusEmbedding(model_name="BAAI/bge-multilingual-gemma2", api_key=api_key)
        )
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize", similarity_top_k=3,
            llm=NebiusLLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=api_key)
        )
        logger.info(f"Knowledge base initialized with {len(docs)} documents.")


kb = KB()

# --- Tool Functions ---

def resolve_query_fn(question: str) -> str:
    """Resolves question from the KB."""
    logger.info(f"Resolving query from KB: '{question}'")
    resp = kb.query_engine.query(question.strip())
    has_answer = (
        resp.source_nodes and len(resp.source_nodes) > 0 and resp.response and resp.response.strip()
        and "don't know" not in resp.response.lower() and "cannot" not in resp.response.lower()
    )
    if has_answer:
        return f"KB_ANSWER: {resp.response.strip()}"
    else:
        return "NO_KB_INFO: No information found in knowledge base for this question"

def classify_fn(message: str) -> str:
    """Classifies sentiment (positive/neutral/negative)."""
    logger.info(f"Classifying sentiment for: '{message}'")
    prompt = f"""Analyze the sentiment of the following user message. Classify it as one of: positive, neutral, or negative.
- Use 'negative' for any message expressing frustration, anger, disappointment, or containing profanity.
- Use 'positive' for messages expressing satisfaction or praise.
- Use 'neutral' for questions or statements without strong emotion.

Return ONLY the single word classification.

User Message: "{message.strip()}"
Classification:"""
    resp = completion(
        model="nebius/meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("NEBIUS_API_KEY"),
        max_tokens=10,
        temperature=0.0 
    )
    sentiment = resp.choices[0].message.content.strip().lower().replace(".", "")
    valid_sentiments = ["positive", "neutral", "negative"]
    if sentiment in valid_sentiments:
        return sentiment
    logger.warning(f"Sentiment classification returned an invalid result: '{sentiment}'. Defaulting to neutral.")
    return "neutral" 

def escalate_fn(message: str) -> str:
    """Escalates message to human support."""
    logger.info(f"[ESCALATION] Forwarding to human support: {message.strip()}")
    return "Your message has been escalated to human support. We will contact you shortly."