## Chatbot to help users understand data better

## Imports
import os
import sys

## Add relevant directories to path
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from multiagent_network_functions import *

load_dotenv()
assert os.environ["LANGCHAIN_API_KEY"], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ["OPENAI_API_KEY"], "Please set the OPENAI_API_KEY environment variable"


## Chatbot class
class MultiAgentChatbot():
    def __init__(self, db_path, llm="openai", with_memory=False):
        print("Initializing multi-agent chatbot...")

        self.db_path = db_path
        self.parser = StrOutputParser()

        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Database does not exist. Please check path provided or create database...")

        openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])
        
        self.graph, self.config = create_multiagent_graph(db_path=self.db_path, llm=openai_llm, with_memory=with_memory)

    
    ## Calls the graph to generate a response
    def invoke(self, message: str):
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
    def stream(self, message):
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