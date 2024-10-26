## Chatbot to help users understand data better

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
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain_core.tools import tool

## Python packages
from dotenv import load_dotenv
from pprint import pprint

## Self-defined modules
from utils.create_orders_db import create_orders_db


load_dotenv()
assert os.environ["LANGCHAIN_API_KEY"], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ["OPENAI_API_KEY"], "Please set the OPENAI_API_KEY environment variable"


## Chatbot class
class SQLAgent():
    def __init__(self, db_path, llm="openai"):
        print("Initializing SQL agent...")
        
        self.db_path = db_path
        self.agent = None
        self.parser = StrOutputParser()

        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Database does not exist. Please check path provided or create database...")

        openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])
        db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")

        toolkit = SQLDatabaseToolkit(db=db, llm=openai_llm)
        tools = toolkit.get_tools()

        ## Prompt template for SQL agent, which contains rules for generating required SQL queries
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="SQLite", top_k=5)
        # print("SQL agent system message: \n", self.parser.parse(system_message))

        ## Initialize agent
        agent_executor = create_react_agent(
            model=openai_llm, tools=tools, state_modifier=system_message
        )
        self.agent = agent_executor

    ## Calls the agent to generate a response
    def invoke(self, message):
        print("Chatbot is responding to user input...")
        response = self.agent.invoke({"messages": [("user", message)]})
        return self.parser.parse(response["messages"][-1].content)

    ## Streams the conversation with the agent
    def stream(self, message):
        events = self.agent.stream({"messages": [("user", message)]}, stream_mode="values")
        ret_stream = [self.parser.parse(event["messages"][-1].content) for event in events]
        return ret_stream