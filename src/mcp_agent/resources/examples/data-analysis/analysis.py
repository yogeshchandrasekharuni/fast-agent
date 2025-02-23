import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Data Analysis (Roots)")


@fast.agent(
    name="data_analysis",
    instruction="""
You have access to a Python 3.12 interpreter and you can use this to analyse and process data. 
Common analysis packages such as Pandas, Seaborn and Matplotlib are already installed. 
You can add further packages if needed.
Data files are accessible from the /mnt/data/ directory (this is the current working directory).
Visualisations should be saved as .png files in the current working directory.
""",
    servers=["interpreter"],
)
async def main():
    # Use the app's context manager
    async with fast.run() as agent:
        await agent(
            "There is a csv file in the current directory. "
            "Analyse the file, produce a detailed description of the data, and any patterns it contains.",
        )
        await agent(
            "Consider the data, and how to usefully group it for presentation to a Human. Find insights, using the Python Interpreter as needed.\n"
            "Use MatPlotLib to produce insightful visualisations. Save them as '.png' files in the current directory. Be sure to run the code and save the files.\n"
            "Produce a summary with major insights to the data",
        )


if __name__ == "__main__":
    asyncio.run(main())
