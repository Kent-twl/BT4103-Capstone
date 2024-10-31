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
            SystemMessage(multiagent_graph_agent_prompt            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt.partial(system_message=system_message)
    prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


## Function for creating a chatbot with no tool-calling capabilities
def create_chatbot(llm, other_assistants, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(multiagent_graph_chatbot_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt.partial(system_message=system_message)
    prompt.partial(other_assistants="\n".join(other_assistants))

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
def create_multiagent_graph(db_path, llm, with_memory=False):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()
    
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

    ## Create chatbot
    chatbot = create_chatbot(
        llm,
        system_message="""
            The user you are assisting today is a trader at an Australian financial institution,
            interested in analyzing some trading data and other matters related to finance and trading.
        """,
        other_assistants=[
            "1. SQL assistant - Can query database to answer questions."
        ]
    )

    ## Create a chatbot node
    chatbot_node = functools.partial(
        basic_node,
        runnable=chatbot,
        name="chatbot"
    )

    ## Define tool node
    tools = sql_tools ## Full list of all tools needed by multi-agent network
    tool_node = ToolNode(tools)

    ## Create graph
    workflow = StateGraph(State)
    workflow.add_node("sql_agent", sql_node)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("call_tool", tool_node)

    ## Agent nodes go to router
    workflow.add_conditional_edges(
        "sql_agent", 
        router, 
        {"continue": "chatbot", "call_tool": "call_tool", END: END} ## Path map that maps router output to node names
    )
    workflow.add_conditional_edges(
        "chatbot", 
        router, 
        {"continue": "sql_agent", END: END} ## Path map that maps router output to node names
    )
    ## Tool node routes back to the agent that called it, ie the sender
    workflow.add_conditional_edges(
        "call_tool",
        lambda state: state["sender"],
        {"chatbot": "chatbot", "sql_agent": "sql_agent"}
    )
    workflow.add_edge(START, "chatbot")

    ## Compile graph
    if with_memory:
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"}}
        return graph, config
    
    graph = workflow.compile()

    return graph, None