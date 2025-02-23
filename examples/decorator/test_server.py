"""Test example with invalid server."""

import os
import asyncio
from mcp_agent.core.fastagent import FastAgent

agent_app = FastAgent("Test Example")

@agent_app.agent(
    name="broken",
    servers=["nonsense_server_name"],  # This should be caught in validation
    instruction="You are a helpful agent."
)
async def main():
    async with agent_app.run() as agent:
        print("If you see this, validation failed!")
        await agent("print the next number in the sequence")

if __name__ == "__main__":
    if os.environ.get("TEST_RUN"):
        print("WARNING: Script is being re-run!")
    else:
        os.environ["TEST_RUN"] = "1"
        print("Running main()...")
        asyncio.run(main())