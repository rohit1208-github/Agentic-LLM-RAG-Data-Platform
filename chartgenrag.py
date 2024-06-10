import urllib.request
import zipfile
import dotenv
import pandas as pd
from langchain import LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import (AgentExecutor, Tool, ZeroShotAgent, initialize_agent, load_tools)
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent
import datetime
import time
import matplotlib as plt
import seaborn as sns
from langchain_community.chat_models import ChatOllama

load_dotenv()

local_llm = 'llama3'
llm = ChatOllama(model=local_llm, temperature=0)

data_dict = {
    "year": [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
    "quarter": ["q1", "q2", "q3", "q4", "q1", "q2", "q3", "q4", "q1", "q2", "q3", "q4", "q1", "q2", "q3", "q4"],
    "revenue": [23, 53, 64, 23, 23, 53, 64, 23, 23, 53, 64, 23, 23, 53, 64, 23]
}
dataframe = pd.DataFrame.from_dict(data_dict)
dataframe.head()

pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=dataframe,
    verbose=True,
)

pandas_agent("What is the total number of rows?")

loaded_tools = load_tools(["python_repl"])

data_analysis_tool = Tool(
    name="Data Analysis Tool",
    func=pandas_agent.run,
    description="Utilize this tool when answering questions related to the data in the pandas dataframe. Prefer this over the Python tool for queries about revenue, year, etc. For example, 'What is the highest revenue?', 'Which year had the minimum revenue?'"
)

loaded_tools.append(data_analysis_tool)

agent_prompt = ZeroShotAgent.create_prompt(
    tools=loaded_tools,
    prefix="Please fulfill the following request to the best of your ability. You have access to these tools:",
    suffix="""When searching for data, fully utilize the Data Analysis Tool.
    
    Request: {input}\n
    {agent_scratchpad}
    """,
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
tool_names = [tool.name for tool in loaded_tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=loaded_tools, verbose=True
)

request = "Create a bar graph visualizing the revenue against the year"
agent_executor.run(request)

request = "Generate a pie chart showing the revenue for each quarter"
agent_executor.run(request)

def visualize_data(df: pd.DataFrame, llm):
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
    )
    loaded_tools = load_tools(["python_repl"])
    
    data_analysis_tool = Tool(
        name="Data",
        func=pandas_agent.run,
        description="Use this tool to answer questions about the data in the pandas dataframe. Prefer this over the Python tool for queries related to revenue, year, etc. For example, 'What is the maximum revenue?', 'Which year had the lowest revenue?'"
    )
    loaded_tools.append(data_analysis_tool)

    agent_prompt = ZeroShotAgent.create_prompt(
    tools=loaded_tools,
    prefix="Please fulfill the following request to the best of your ability. You have access to these tools:",
    suffix="""When searching for data, fully utilize the Data tool. 
    The provided dataframe is\n
    df = {dataframe}\n
    
    Request: {input}\n

    {agent_scratchpad}
    """,
    input_variables=["input", "dataframe", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    tool_names = [tool.name for tool in loaded_tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=loaded_tools, verbose=True
    )

    return agent_executor

plot_executor = visualize_data(dataframe, llm)
plot_executor.run("Generate a bar chart showing the relationship between year and revenue")
