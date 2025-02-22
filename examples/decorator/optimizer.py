"""
Example showing how to use the evaluator-optimizer functionality with the decorator API.
This demonstrates creating an optimizer and evaluator to iteratively improve content.
"""

import asyncio
from mcp_agent.core.fastagent import FastAgent

# Create the application
agent_app = FastAgent("Cover Letter Writer")
agent_app.app._human_input_callback = None


# Define optimizer agent
@agent_app.agent(
    name="optimizer",
    instruction="""You are a career coach specializing in cover letter writing.
    You are tasked with generating a compelling cover letter given the job posting,
    candidate details, and company information. Tailor the response to the company and job requirements.
    """,
    servers=["fetch"],
    model="gpt-4o-mini",  # Using a capable model for content generation
)
# Define evaluator agent
@agent_app.agent(
    name="evaluator",
    instruction="""Evaluate the following response based on the criteria below:
    1. Clarity: Is the language clear, concise, and grammatically correct?
    2. Specificity: Does the response include relevant and concrete details tailored to the job description?
    3. Relevance: Does the response align with the prompt and avoid unnecessary information?
    4. Tone and Style: Is the tone professional and appropriate for the context?
    5. Persuasiveness: Does the response effectively highlight the candidate's value?
    6. Grammar and Mechanics: Are there any spelling or grammatical issues?
    7. Feedback Alignment: Has the response addressed feedback from previous iterations?

    For each criterion:
    - Provide a rating (EXCELLENT, GOOD, FAIR, or POOR).
    - Offer specific feedback or suggestions for improvement.

    Summarize your evaluation as a structured response with:
    - Overall quality rating.
    - Specific feedback and areas for improvement.""",
    servers=[],  # Evaluator doesn't need special server access
    model="sonnet",  # Using a capable model for evaluation
)
# Define the evaluator-optimizer workflow
@agent_app.evaluator_optimizer(
    name="cover_letter_writer",
    optimizer="optimizer",  # Reference to optimizer agent
    evaluator="evaluator",  # Reference to evaluator agent
    min_rating="EXCELLENT",  # Strive for excellence
    max_refinements=3,  # Maximum iterations
)
async def main():
    async with agent_app.run() as agent:
        job_posting = (
            "Software Engineer at LastMile AI. Responsibilities include developing AI systems, "
            "collaborating with cross-functional teams, and enhancing scalability. Skills required: "
            "Python, distributed systems, and machine learning."
        )
        candidate_details = (
            "Alex Johnson, 3 years in machine learning, contributor to open-source AI projects, "
            "proficient in Python and TensorFlow. Motivated by building scalable AI systems to solve real-world problems."
        )
        company_information = (
            "Look up from the LastMile AI About page: https://lastmileai.dev/about"
        )

        # Send the task
        await agent.cover_letter_writer.send(
            f"Write a cover letter for the following job posting: {job_posting}\n\n"
            f"Candidate Details: {candidate_details}\n\n"
            f"Company information: {company_information}",
        )


if __name__ == "__main__":
    asyncio.run(main())
