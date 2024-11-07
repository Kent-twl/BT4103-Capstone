## Multi-agent network with multiple tool-calling agents and a prompt engineer

## Imports
import os
import sys

## Add relevant directories to path
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

## LangChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver

## Python packages
from dotenv import load_dotenv
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import functools

## Self-defined modules
from llm.prompts.multiagent_network_prompts import *

load_dotenv()
assert os.environ["LANGCHAIN_API_KEY"], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ["OPENAI_API_KEY"], "Please set the OPENAI_API_KEY environment variable"


## Function for creating an agent
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(multiagent_graph_agent_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt.partial(system_message=system_message)
    prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


## Function for creating an entry point with no tool-calling capabilities
def create_entry_point(llm, other_assistants, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(multimodal_multiagent_entry_point),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt.partial(system_message=system_message)
    prompt.partial(other_assistants="\n".join(other_assistants))

    return prompt | llm


## Function for creating a prompt engineer
def create_prompt_engineer(llm, other_assistants, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(multiagent_graph_engineer_prompt),
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


## Function for creating graph with prompt engineer, SQL and Tavily
def create_man_with_prompt_engineer(llm, main_system_message, db_path, with_memory=True):
    ## Initialize variables
    all_tools = [] ## Full list of all tools used in network
    other_assistants = [] ## Descriptions for assistants other than the chatbot

    ## Set up SQL agent
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()
    all_tools.extend(sql_tools)

    sql_agent = create_agent(
        llm, 
        sql_tools,
        # system_message=hub.pull("langchain-ai/sql-agent-system-prompt").format(dialect="SQLite", top_k=5)
        system_message="""
            System: You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct SQLite query to run, 
            then look at the results of the query and return the answer.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.

            In this case, you are assisting a team of traders and analysts in deriving insights from a chart.
            Do this using available information in the database. For instance, you could provide historical data
            to make comparisons, or retrieve specific data points to answer questions.
        """
    )

    sql_node = functools.partial(
        agent_node, 
        agent=sql_agent,
        name="sql_agent"
    )

    other_assistants.append("SQL assistant - Can query database to answer questions.")

    ## Set up Tavily agent
    search = TavilySearchAPIWrapper()
    tavily_tools = [
        TavilySearchResults(
            api_wrapper=search,
            search_depth="advanced",
            include_answer=True
        )
    ]
    all_tools.extend(tavily_tools)

    tavily_agent = create_agent(
        llm,
        tavily_tools,
        system_message="""
            You are an AI assistant to a team of traders and analysts. Help them find relevant news or information to support their needs
            or address their problems. Pay attention to the specified date ranges. Run a search for each day of the date range, 
            and compile the results. If no dates are specified, retrieve the latest information you have.
            If you cannot find all the required information, it is ok, just respond with what you have.
            Make sure you answer the user's question directly in your response. Do not ask for follow-ups.
        """
    )

    tavily_node = functools.partial(
        agent_node,
        agent=tavily_agent,
        name="tavily_agent"
    )

    other_assistants.append("Tavily assistant - Searches the web for relevant info to supplement answers.")

    ## Set up prompt engineer
    prompt_engineer = create_prompt_engineer(
        llm,
        other_assistants=other_assistants,
        system_message="""
            Your team is assisting financial traders and analysts in deriving insights from dashboard charts.
            The entry point LLM will perform a preliminary description of the chart, and provide some avenues
            for further exploration. 
        """
    )

    prompt_engineer_node = functools.partial(
        basic_node,
        runnable=prompt_engineer,
        name="prompt_engineer"
    )

    ## Create entry point of multi-agent network
    entry_point = create_entry_point(
        llm,
        other_assistants=other_assistants,
        system_message=main_system_message
    )

    entry_point_node = functools.partial(
        basic_node,
        runnable=entry_point,
        name="entry_point"
    )

    ## Define tool node
    tools = all_tools 
    tool_node = ToolNode(tools)

    ## Create graph
    workflow = StateGraph(State)
    # workflow.add_node("entry_point", entry_point_node)
    workflow.add_node("prompt_engineer", prompt_engineer_node)
    workflow.add_node("sql_agent", sql_node) 
    workflow.add_node("tavily_agent", tavily_node)
    workflow.add_node("call_tool", tool_node)

    ##TODO: Try routing from agent to agent, then try routing from agent to prompt engineer
    # workflow.add_conditional_edges(
    #     "entry_point",
    #     router,
    #     {"continue": "prompt_engineer", END: END}
    # )
    workflow.add_conditional_edges(
        "prompt_engineer",
        router,
        {"continue": "sql_agent", END: END}
    )
    workflow.add_conditional_edges(
        "sql_agent",
        router,
        {"continue": "tavily_agent", "call_tool": "call_tool", END: END}
    )
    workflow.add_conditional_edges(
        "tavily_agent",
        router,
        {"continue": "sql_agent", "call_tool": "call_tool", END: END}
    )
    ## Tool node routes back to the agent that called it (ie. sender)
    workflow.add_conditional_edges(
        "call_tool",
        lambda state: state["sender"],
        {"sql_agent": "sql_agent", "tavily_agent": "tavily_agent"}
    )
    # workflow.add_edge(START, "entry_point")
    workflow.add_edge(START, "prompt_engineer")

    ## Compile graph
    if with_memory:
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "1"}}
    else:
        graph = workflow.compile()
        config = None

    return graph, config



## Multi-agent network class
class MANwithPromptEngineer():
    def __init__(self, main_system_message, db_path, with_memory=True):
        print("Initializing multi-agent network with prompt engineer...")

        self.db_path = db_path
        self.parser = StrOutputParser()

        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Database does not exist. Please check path provided or create database...")

        openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])
        
        self.graph, self.config = create_man_with_prompt_engineer(
            llm=openai_llm,
            main_system_message=main_system_message, 
            db_path=self.db_path,   
            with_memory=with_memory
        )

    
    ## Calls the graph to generate a response
    def invoke_messages(self, messages):
        print("Responding to user input...")
        try:
            if self.config:
                response = self.graph.invoke({"messages": messages}, config=self.config)
            else:
                response = self.graph.invoke({"messages": messages})
            response_str = self.parser.parse(response["messages"][-1].content)
            response_str = remove_final_answer_prefix(response_str)
        except Exception as e:
            print(f"Error: {e}")
            response_str = "I'm sorry, the question seems to be out of my area of expertise. Let me know if you have any other questions."
        return response_str
    

    ## Calls the graph to generate a response
    def invoke_str(self, message: str):
        print("Responding to user input...")
        try:
            if self.config:
                response = self.graph.invoke({"messages": [("user", message)]}, config=self.config)
            else:
                response = self.graph.invoke({"messages": [("user", message)]})
            response_str = self.parser.parse(response["messages"][-1].content)
            response_str = remove_final_answer_prefix(response_str)
        except Exception as e:
            print(f"Error: {e}")
            response_str = "I'm sorry, the question seems to be out of my area of expertise. Let me know if you have any other questions."
        return response_str
    

    ## Streams the conversation with the agent
    def stream(self, message: str):
        try:
            if self.config:
                events = self.graph.stream({"messages": [("user", message)]}, config=self.config, stream_mode="values")
            else:
                events = self.graph.stream({"messages": [("user", message)]}, stream_mode="values")
            ret_stream = [self.parser.parse(event["messages"][-1].content) for event in events]
        except Exception as e:
            print(f"Error: {e}")
            ret_stream = ["I'm sorry, the question seems to be out of my area of expertise. Let me know if you have any other questions."]
        return ret_stream
    

    ## Streams output with more info for debugging
    def debug_stream(self, message: str):
        if self.config:
                events = self.graph.stream({"messages": [("user", message)]}, config=self.config)
        else:
            events = self.graph.stream({"messages": [("user", message)]})
        for event in events:
            for value in event.values():
                if "sender" in value:
                    sender = value["sender"]
                else:
                    sender = "tool"
                print("-----------------")
                print(sender, "\n", value["messages"][-1].content)

