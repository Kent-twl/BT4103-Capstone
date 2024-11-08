## Helper functions for creating multi-agent graph

## Imports
## System imports
import os
import sys

## Add relevant directories to path
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

## LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.prebuilt import ToolNode
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver

## Python packages
from dotenv import load_dotenv
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import functools

load_dotenv()
assert os.environ["LANGCHAIN_API_KEY"], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ["OPENAI_API_KEY"], "Please set the OPENAI_API_KEY environment variable"

## Self-defined modules
from llm.prompts.chatbot_prompts import *


## Function for creating an agent
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", multiagent_graph_agent_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


## Function for creating a chatbot with no tool-calling capabilities
def create_entry_point(llm, other_assistants, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", multiagent_graph_chatbot_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(other_assistants="\n".join(other_assistants))

    return prompt | llm


## State to be passed between nodes (agents and tools)
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str ## Tracks most recent sender


## Function for creating nodes of agents
def agent_node(state: State, agent, name):
    result = agent.invoke(state)
    
    ## Format output to a suitable format to be appended to state (unless it is a tool message)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    output = {
        "messages": [result],
        "sender": name
    }
    
    return output


## Function for creating nodes without tool-calling
def basic_node(state: State, runnable, name):
    result = runnable.invoke(state["messages"])
    result.name = name

    output = {
        "messages": [result],
        "sender": name
    }
    
    return output


## Define router with logic to handle tool-calling and ending
def router(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    elif "FINAL_ANSWER" in last_message.content:
        return END
    return "continue"


## Remove FINAL_ANSWER prefix
def remove_final_answer_prefix(message):
    return message.replace("FINAL_ANSWER:", "").strip()


## Function for creating graph with SQL agent and basic chatbot
def create_multiagent_graph(main_system_message, db_path, llm, agents, with_memory=False):
    ## Initialize variables
    all_tools = [] ## Full list of all tools used in network
    other_assistants = [] ## Descriptions for assistants other than the chatbot
    
    if "sql" in agents:
        ## Set up for SQL agent
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        sql_tools = sql_toolkit.get_tools()
        all_tools.extend(sql_tools)

        ## Create agents
        sql_agent = create_agent(
            llm, 
            sql_tools,
            system_message=hub.pull("langchain-ai/sql-agent-system-prompt").format(dialect="SQLite", top_k=5)
        )

        ## Create a partial function pre-filled with SQL agent details that accepts state as input
        sql_node = functools.partial(
            agent_node, 
            agent=sql_agent,
            name="sql_agent"
        )

        other_assistants.append("SQL assistant - Can query database to answer questions.")

    if "yfinance" in agents:
        ## Set up for Yahoo Finance agent
        yfinance_tools = [YahooFinanceNewsTool()]
        all_tools.extend(yfinance_tools)

        ## Create agent
        yfinance_agent = create_agent(
            llm, 
            yfinance_tools,
            system_message="""
            You are responsible for retrieving useful info from Yahoo Finance to support the other assistants. 
            Refer to relevant news to answer the user's questions or justify the findings made by the other assistants.
            """
        )

        ## Partial function that serves as the node for the Yahoo Finance agent
        yfinance_node = functools.partial(
            agent_node, 
            agent=yfinance_agent,
            name="yfinance_agent"
        )

        other_assistants.append("Yahoo Finance assistant - Provides insights from online financial news.")

    ## Create chatbot, ie the entry point of the multi-agent network
    entry_point = create_entry_point(
        llm,
        system_message=main_system_message,
        other_assistants=other_assistants
    )

    ## Create a chatbot node
    entry_point_node = functools.partial(
        basic_node,
        runnable=entry_point,
        name="entry_point"
    )

    ## Define tool node
    tool_node = ToolNode(all_tools) ## Full list of all tools needed by multi-agent network

    ## Create graph
    workflow = StateGraph(State)
    workflow.add_node("entry_point", entry_point_node)
    workflow.add_node("sql_agent", sql_node) if "sql" in agents else None
    workflow.add_node("yfinance_agent", yfinance_node) if "yfinance" in agents else None
    workflow.add_node("call_tool", tool_node)

    ## Agent nodes go to router
    if "sql" in agents:
        workflow.add_conditional_edges(
            "sql_agent", 
            router, 
            {
                "continue": "yfinance_agent" if "yfinance" in agents else "entry_point", 
                "call_tool": "call_tool", 
                END: END
            } ## Path map that maps router output to node names
        )
    
    if "yfinance" in agents:
        workflow.add_conditional_edges(
            "yfinance_agent", 
            router, 
            {
                "continue": "sql_agent" if "sql" in agents else "entry_point", 
                "call_tool": "call_tool", 
                END: END
            }
        )
    
    next_node = "sql_agent" if "sql" in agents else \
        "yfinance_agent" if "yfinance" in agents else \
        END ## Next node to go to after the entry point
    workflow.add_conditional_edges(
        "entry_point", 
        router, 
        {
            "continue": next_node, 
            END: END
        }
    )
    
    ## Tool node routes back to the agent that called it, ie the sender
    tool_node_path_map = {"entry_point": "entry_point"}
    if "sql" in agents:
        tool_node_path_map["sql_agent"] = "sql_agent"
    if "yfinance" in agents:
        tool_node_path_map["yfinance_agent"] = "yfinance_agent"
    
    workflow.add_conditional_edges(
        "call_tool",
        lambda state: state["sender"],
        tool_node_path_map
    )

    workflow.add_edge(START, "entry_point")

    ## Compile graph
    if with_memory:
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"}}
    else:
        graph = workflow.compile()
        config = None

    return graph, config