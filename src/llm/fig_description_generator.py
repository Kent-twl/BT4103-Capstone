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
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from llm.prompts import dashboard_prompts, variable_descriptions

## Add directories to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.dirname(__file__)) ## Current directory

load_dotenv()   
assert os.environ['LANGCHAIN_API_KEY'], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ['OPENAI_API_KEY'], "Please set the OPENAI_API_KEY environment variable"


def fig_description_generator(fig: plotly.graph_objs.Figure, dash_type="bi", chart_type="line", date_range="today", vars=[], additional_info=None):
    """
    fig: plotly.graph_objs.Figure
    The figure object for which a description is to be generated.

    dash_type: str
    The dashboard type, one of "bi", "asic", or "anomaly".

    chart_type: str
    The type of chart being described, eg. "line", "bar", "scatter", etc.

    date_range: str
    The date range for the data displayed in the chart, as a string.

    vars: list
    List of variables used in the chart.

    additional_info: Any
    Additional info to be included in the description, eg. for anomaly detection dashboards.
    """
    ##TODO: Use multi-modal multi-agent graph
    ## Initialize LLM
    openai_llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.environ['OPENAI_API_KEY'])

    ## Convert figure to bytes
    try:
        fig_bytes =  base64.b64encode(fig.to_image(format='png', engine='kaleido')).decode('utf-8')
    except Exception as e:
        return "Error generating image bytes: " + e

    ## Get necessary prompts
    system_prompt = dashboard_prompts.system_prompt
    dash_prompt = dashboard_prompts.anomaly_dash_prompt if dash_type == "anomaly" else \
        dashboard_prompts.asic_dash_prompt if dash_type == "asic" else dashboard_prompts.bi_dash_prompt
    ## For anomaly detection dashboard, format the prompt with the reasons for classification
    if dash_type == "anomaly":
        # assert additional_info != None, \
        assert additional_info is not None and not additional_info.empty, \
            "Please provide the description generator with the anomaly detection results for this dashboard."
        assert isinstance(additional_info, pd.DataFrame), "additional_info must be a pandas DataFrame"
        anomaly_results = additional_info.to_dict(orient="records")
        dash_prompt = dash_prompt.format(anomalies_and_reasons=anomaly_results)
    
    vars_dict = variable_descriptions.variable_dict
    relevant_vars = ", ".join([vars_dict[var] for var in vars if var in vars_dict])
    vars_prompt = variable_descriptions.variable_prompt.format(variables=relevant_vars)
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
                    "text": f"""
                        I am an analyst looking at this {chart_type} chart displaying data from {date_range}. 
                        Provide a brief description of the chart for me.
                    """
                },
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