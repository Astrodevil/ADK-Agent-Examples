import streamlit as st
import requests
import uuid
import time
import json

# --- Configuration ---
COORDINATOR_URL = "http://127.0.0.1:10023"
AGENT_HEALTH_URL = f"{COORDINATOR_URL}/.well-known/agent.json"
REQUEST_TIMEOUT = 120 

# --- Agent Communication ---
def check_agent_status():
    """Checks if the coordinator agent server is running."""
    try:
        response = requests.get(AGENT_HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def query_coordinator(message: str) -> str:
    """
    Sends a message to the coordinator agent and returns the response.
    This function mimics the A2A protocol request structure.
    """
    try:
        
        payload = {
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": str(uuid.uuid4().hex)
                }
            }
        }
        
        
        response = requests.post(COORDINATOR_URL + "/", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        response_data = response.json()

    
        if 'result' in response_data and 'artifacts' in response_data['result']:
            for artifact in response_data['result']['artifacts']:
                if 'parts' in artifact:
                    for part in artifact['parts']:
                        if 'text' in part and part['text'].strip():
                            return part['text'].strip()
        
        
        return "The agent returned a response, but I couldn't extract the text. Full response: " + json.dumps(response_data)

    except requests.Timeout:
        return "Error: The request to the agent timed out. The agent might be busy or the task is taking too long."
    except requests.RequestException as e:
        return f"Error: Could not communicate with the agent. Please ensure it's running. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="SwiftCart Support", page_icon="🛒")
st.title("🛒 SwiftCart Customer Support")
st.caption("This chat interface is powered by a multi-agent system.")


if not check_agent_status():
    st.error(
        "**Could not connect to the Support Agent System.**\n\n"
        "Please start the backend servers by running the following command in a separate terminal:\n\n"
        "```bash\n"
        "python a2a_customer_routing/multi_agent/run_agents.py\n"
        "```"
    )
else:
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to SwiftCart support! How can I help you today?"}]

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_coordinator(prompt)
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})