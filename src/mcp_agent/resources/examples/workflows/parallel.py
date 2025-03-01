"""
Parallel Workflow showing Fan Out and Fan In agents, using different models
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent(
    "Parallel Workflow",
)
SHORT_STORY = """
The Battle of Glimmerwood

In the heart of Glimmerwood, a mystical forest knowed for its radiant trees, a small village thrived. 
The villagers, who were live peacefully, shared their home with the forest's magical creatures, 
especially the Glimmerfoxes whose fur shimmer like moonlight.

One fateful evening, the peace was shaterred when the infamous Dark Marauders attack. 
Lead by the cunning Captain Thorn, the bandits aim to steal the precious Glimmerstones which was believed to grant immortality.

Amidst the choas, a young girl named Elara stood her ground, she rallied the villagers and devised a clever plan.
Using the forests natural defenses they lured the marauders into a trap. 
As the bandits aproached the village square, a herd of Glimmerfoxes emerged, blinding them with their dazzling light, 
the villagers seized the opportunity to captured the invaders.

Elara's bravery was celebrated and she was hailed as the "Guardian of Glimmerwood". 
The Glimmerstones were secured in a hidden grove protected by an ancient spell.

However, not all was as it seemed. The Glimmerstones true power was never confirm, 
and whispers of a hidden agenda linger among the villagers.
"""


@fast.agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
    Identify any awkward phrasing or structural issues that could improve clarity. 
    Provide detailed feedback on corrections.""",
)
@fast.agent(
    name="fact_checker",
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
    model="gpt-4o",
)
@fast.agent(
    name="style_enforcer",
    instruction="""Analyze the story for adherence to style guidelines.
    Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
    enhance storytelling, readability, and engagement.""",
    model="sonnet",
)
@fast.agent(
    name="grader",
    instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
    into a structured report. Summarize key issues and categorize them by type. 
    Provide actionable recommendations for improving the story, 
    and give an overall grade based on the feedback.""",
    model="o3-mini.low",
)
@fast.parallel(
    fan_out=["proofreader", "fact_checker", "style_enforcer"],
    fan_in="grader",
    name="parallel",
)
async def main():
    # Use the app's context manager
    async with fast.run() as agent:
        await agent.parallel(f"student short story submission: {SHORT_STORY}")

        # follow-on prompt to task agent
        await agent.style_enforcer.prompt(default_prompt="STOP")


if __name__ == "__main__":
    asyncio.run(main())
