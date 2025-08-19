import streamlit as st
import requests
import uuid
import json
import asyncio
import httpx
import threading
import traceback
from typing import Coroutine, Any

# Import modern A2A client components and types
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.types import AgentCard, Task
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

# --- Configuration ---
COORDINATOR_URL = "http://127.0.0.1:10023"
AGENT_HEALTH_URL = f"{COORDINATOR_URL}{AGENT_CARD_WELL_KNOWN_PATH}"
REQUEST_TIMEOUT = 120

# --- Agent Communication ---

def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Runs a coroutine in a new thread and returns the result, propagating exceptions.
    """
    result = None
    exception = None

    def runner():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if exception:
        raise exception

    return result

def check_agent_status():
    """Checks if the coordinator agent server is running."""
    try:
        response = requests.get(AGENT_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

async def query_coordinator_async(message: str) -> str:
    """
    Sends a message to the coordinator agent, now with robust artifact checking.
    """
    timeout_config = httpx.Timeout(REQUEST_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        agent_card_response = await httpx_client.get(AGENT_HEALTH_URL)
        agent_card_response.raise_for_status()
        agent_card = AgentCard(**agent_card_response.json())

        config = ClientConfig(httpx_client=httpx_client)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        message_obj = create_text_message_object(content=message)
        final_response = "The agent did not return a valid text artifact."

        async for response in client.send_message(message_obj):
            if isinstance(response, tuple) and len(response) > 0:
                task: Task = response[0]

                
                if task.artifacts:
                    try:
                    
                        text_response = task.artifacts[0].parts[0].root.text
                        if text_response:
                            final_response = text_response.strip()
                            break 
                    except (AttributeError, IndexError):
                       
                        pass
        
        return final_response

def query_coordinator(message: str) -> str:
    """Wrapper to call the async function from Streamlit's sync context."""
    return run_async_in_thread(query_coordinator_async(message))

# --- Streamlit UI ---
st.set_page_config(page_title="SwiftCart Support", page_icon="🛒")
st.title("🛒 SwiftCart Customer Support")
st.caption("This chat interface is powered by a multi-agent system.")

if not check_agent_status():
    st.error(
        "**Could not connect to the Support Agent System.**\n\n"
        "Please start the backend servers by running the following command in a separate terminal:\n\n"
        "```bash\n"
        "python -m a2a_customer_routing.multi_agent.run_agents\n"
        "```"
    )
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to SwiftCart support! How can I help you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_coordinator(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("An error occurred while communicating with the agent.")
                    st.code(f"{e}\n\n{traceback.format_exc()}")