## FastAgent

<p align="center">
<a href="https://pypi.org/project/fast-agent-mcp/"><img src="https://img.shields.io/pypi/v/fast-agent-mcp?color=%2334D058&label=pypi" /></a>
<a href="https://github.com/evalstate/fast-agent/issues"><img src="https://img.shields.io/github/issues-raw/evalstate/fast-agent" /></a>
<a href="https://lmai.link/discord/mcp-agent"><img src="https://shields.io/discord/1089284610329952357" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/fast-agent-mcp?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/evalstate/fast-agent-mcp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/fast-agent-mcp" /></a>
</p>

## Overview

**`fast-agent`** lets you define, test and interact with agents, tools and workflows in minutes.

The simple declarative syntax lets you concentrate on the prompts, MCP Servers and compositions to build effective agents.

Quickly compare how different models perform at Agent and MCP Server calling tasks, and build mixed multi-model workflows using the best provider for each task.

### Get started:

Start by installing the [uv package manager](https://docs.astral.sh/uv/) for Python. Then:

```bash
uv pip install fast-agent-mcp       # install fast-agent
fast-agent setup                    # create an example agent and config files
uv run agent.py                     # run your first agent
uv run agent.py --model=o3-mini.low # specify a model
fast-agent bootstrap workflow       # create "building effective agents" examples
```

Other bootstrap examples include a Researcher Agent (with Evaluator-Optimizer workflow) and Data Analysis Agent (similar to the ChatGPT experience), demonstrating MCP Roots support.

> Windows Users - there are a couple of configuration changes needed for the Filesystem and Docker MCP Servers - necessary changes are detailed within the configuration files.

## Agent Development

FastAgent lets you interact with Agents during a workflow, enabling "warm-up" and diagnostic prompting to improve behaviour and refine prompts. 

### Basic Agents

Defining an agent is as simple as:

```python
@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
```

We can then send messages to the Agent:

```python
  async with fast.run() as agent:
  
    moon_size = await agent("the moon")
    print(moon_size)
```

Or start an interactive session to chat with the Agent:
```python
  async with fast.run() as agent:  
    await agent()
```

The entire program `sizer.py`, with boilerplate code:
```python
import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)

async def main():
  async with fast.run() as agent:
    await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

Can then be run with `uv run sizer.py`, or with a specific model with `uv run sizer.py --model gpt-4o-mini`.

### Chaining Agents and using an MCP Server

_To generate runnable examples use `fastagent bootstrap workflow`._ 
_To run this example, type `uv run chaining.py`._

Agents can be chained together to build a workflow:
```python
@fast.agent(
    "url_fetcher",
    instruction="Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"], # Name of an MCP Server defined in fastagent.config.yaml
)
@fast.agent(
    "social_media",
    instruction="""
    Write a 280 character social media post for any given text.
    Respond only with the post, never use hashtags.
    """,
)

async def main():
    async with fast.run() as agent:
        await agent.social_media(
            await agent.url_fetcher("http://llmindset.co.uk/resources/mcp-hfspace/")
        )
```


### Agent Features

```python
@fast.agent(
  name="agent",
  instructions="instructions",
  servers=["filesystem"],     # list of MCP Servers for the agent, configured in fastagent.config.yaml
  model="o3-mini.high",       # specify a model for the agent
  use_history=True,           # agent can maintain chat history
  human_input=True,           # agent can request human input
)


### Chaining Agents



### Evaluator-Optimizer

Evaluator-Optimizers use 2 agents: one to generate content (the `generator`), and one to judge the content and provide actionable feedback (the `evaluator`). The pair run in a loop until either the evaluator is satisfied with the quality or a certain number of iterations have passed.

```python
@fast.evaluator_optimizer(
  name=""
  generator=""
  evaluator=""
  min_rating=""
  max_refinements=3
)
```

### Parallel

Parallels send the same message to multiple agents simultaneously (`fan-out`), and then use a final agent to aggregate the content (`fan-in`). 

```
@fast.parallel(
  name=""
  fan_out=[agent,agent,agent,...]
  fan_in=agent
)
```

### Router

Routers use an LLM to assess a message, identify the most appropriate destination direct . The routing prompt is generated automatically by the Router.

```python
@fast.router(
  name="foo"
  agents["agent1","agent2","agent3"]
)
```

### Orchestrator

Given a task, an Orchestrator uses an LLM to generate a plan to divide the task amongst the available agents and aggregate a result. The planning and aggregation prompts are generated by the Orchestrator, and benefits from more capable models. Plans can either be built once at the beginning (`plantype="full"`) or iteratively (`plantype="iterative"`).  

```python
@fast.orchestrator(
  name="orchestrate"
  agents=["task1","task2","task3"]
)

...


await agent.orchestrate.send("

## MCP Server Development

It's quick and easy to interact with MCP Servers via LLM Tool Calls, providing an excellent testbed to compare how different models behave with your tool definitions. 

## Workflow Patterns

Agents are defined 


### llmindset.co.uk fork:

- "FastAgent" style prototyping, with per-agent models
- Api keys through Environment Variables
- Warm-up / Post-Workflow Agent Interactions
- Quick Setup
- Interactive Prompt Mode
- Simple Model Selection with aliases
- User/Assistant and Tool Call message display
- MCP Sever Environment Variable support
- MCP Roots support
- Comprehensive Progress display
- JSONL file logging with secret revokation
- OpenAI o1/o3-mini support with reasoning level
- Enhanced Human Input Messaging and Handling
- Declarative workflows

## Get Started

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python projects:

## Table of Contents

We welcome any and all kinds of contributions. Please see the [CONTRIBUTING guidelines](./CONTRIBUTING.md) to get started.
