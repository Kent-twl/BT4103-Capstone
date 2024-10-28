import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.join(os.path.dirname(__file__), "../")) ## LLM directory

from dotenv import load_dotenv
import plotly
import plotly.express as px # Ensure kaleido version is 0.1.0.post1
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from llm.prompts import dashboard_prompts as prompts

## Add directories to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.dirname(__file__)) ## Current directory

load_dotenv()   
assert os.environ['LANGCHAIN_API_KEY'], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ['OPENAI_API_KEY'], "Please set the OPENAI_API_KEY environment variable"


def fig_description_generator(fig: plotly.graph_objs.Figure, dash_type="bi", chart_type="line", vars=[]):
    ## Initialize LLM
    openai_llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'])

    ## Convert figure to bytes
    try:
        fig_bytes =  base64.b64encode(fig.to_image(format='png', engine='kaleido')).decode('utf-8')
    except Exception as e:
        return "Error generating image bytes: " + e

    ## Get necessary prompts
    system_prompt = prompts.system_prompt
    dash_prompt = prompts.bi_dash_prompt if dash_type == "bi" else \
        prompts.asic_dash_prompt if dash_type == "asic" else prompts.anomaly_dash_prompt
    vars_prompt = "" ##TODO: Fetch descriptions of the variables used in the chart
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=dash_prompt),
        SystemMessage(content=vars_prompt)
    ]

    ## Generate description
    response = ""
    try:
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "I am an analyst looking at the chart. Provide a brief description of the chart for me."
                },
                ##TODO: Add more prompts wrt chart content and variables
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{fig_bytes}", "type": "png"
                    },
                },
            ],
        )
        messages.append(message)

        response = openai_llm.invoke(messages)
    except Exception as e:
        return "Error creating message: " + e
    
    return response.content
