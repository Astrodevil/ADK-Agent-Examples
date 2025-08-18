import asyncio
import threading
import time
import logging
import uvicorn
import requests
from . import agent as agent_module

from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_server(create_agent_function, port):
    """Run a server in async context."""
    logger.info(f"🚀 Starting agent server on port {port}...")
    app = create_agent_function()
    config = uvicorn.Config(app.build(), host="127.0.0.1", port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()

def run_agent_in_background(create_fn, port, name):
    """Run an agent server in a background thread."""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_server(create_fn, port))
        except Exception as e:
            logger.error(f"{name} server error: {e}", exc_info=True)
    thread = threading.Thread(target=run, daemon=True, name=f"{name}-Thread")
    thread.start()
    return thread

def wait_for_agents(agent_urls, timeout=45):
    """Wait for all agent health endpoints to be available."""
    start_time = time.monotonic()
    ready_agents = set()
    logger.info(f"Waiting for {len(agent_urls)} agents to be ready...")
    while time.monotonic() - start_time < timeout:
        for url in agent_urls:
            if url in ready_agents: continue
            try:
                health_check_url = f"{url}{AGENT_CARD_WELL_KNOWN_PATH}"
                if requests.get(health_check_url, timeout=1).status_code == 200:
                    logger.info(f"  ✅ Agent at {url} is ready.")
                    ready_agents.add(url)
            except requests.ConnectionError: pass
            except Exception as e: logger.warning(f"  ⚠️ Error checking {url}: {e}")
        if len(ready_agents) == len(agent_urls):
            logger.info("🎉 All agents in this group are ready!")
            return True
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for agents. Ready: {list(ready_agents)}")

def start_all_agents():
    """Start all support agents and the coordinator. This is the main entry point."""
    
    support_agents_to_start = {
        "Intake": (agent_module.create_intake_agent_server, 10020),
        "Resolution": (agent_module.create_resolution_agent_server, 10021),
        "Escalation": (agent_module.create_escalation_agent_server, 10022),
    }
    threads = {name: run_agent_in_background(create_fn, port, name) for name, (create_fn, port) in support_agents_to_start.items()}
    support_agent_urls = [f"http://127.0.0.1:{port}" for _, port in support_agents_to_start.values()]
    wait_for_agents(support_agent_urls)


    agent_module.coordinator_agent = agent_module.create_coordinator_agent_with_registered_agents()
    threads["Coordinator"] = run_agent_in_background(agent_module.create_coordinator_agent_server, 10023, "Coordinator")
    wait_for_agents(["http://127.0.0.1:10023"])
    
    logger.info("\n✅ All A2A agents are running and orchestrated!")
    logger.info("   - Intake Agent: http://127.0.0.1:10020")
    logger.info("   - Resolution Agent: http://127.0.0.1:10021")
    logger.info("   - Escalation Agent: http://127.0.0.1:10022")
    logger.info("   - Coordinator Agent: http://127.0.0.1:10023")
    
    return threads

if __name__ == "__main__":
    start_all_agents()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down agents.")