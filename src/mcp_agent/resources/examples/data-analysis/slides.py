import asyncio
from typing import List

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompts.prompt_template import PromptTemplateLoader
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Create the application
fast = FastAgent("Data Analysis (Roots)")


# The sample data is under Database Contents License (DbCL) v1.0.

# Available here : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# The CSS files are distributed under the MIT License from the excellent
# marpstyle project : https://github.com/cunhapaulo/marpstyle


@fast.agent(
    name="slides",
    instruction="""
You produce compelling slide decks for impactful presentations. You usually try and keep the pack to between
8-12 slides, with key insights at the start, backed up with data, diagrams and analysis as available.
""",
    request_params=RequestParams(maxTokens=8192),
)
async def main() -> None:
    # Use the app's context manager
    async with fast.run() as agent:
        slide_templates: List[PromptMessageMultipart] = (
            PromptTemplateLoader().load_from_file("slides.md").to_multipart_messages()
        )
        await agent.apply_prompt(slide_templates)
        await agent()


if __name__ == "__main__":
    asyncio.run(main())


############################################################################################################
# Example of evaluator/optimizer flow
############################################################################################################
# @fast.agent(
#     "evaluator",
#     """You are collaborating with a Data Analysis tool that has the capability to analyse data and produce visualisations.
#     You must make sure that the tool has:
#      - Considered the best way for a Human to interpret the data
#      - Produced insightful visualasions.
#      - Provided a high level summary report for the Human.
#      - Has had its findings challenged, and justified
#     """,
#     request_params=RequestParams(maxTokens=8192),
# )
# @fast.evaluator_optimizer(
#     "analysis_tool",
#     generator="data_analysis",
#     evaluator="evaluator",
#     max_refinements=3,
#     min_rating="EXCELLENT",
# )
