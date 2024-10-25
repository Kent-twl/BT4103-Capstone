import os
import sys
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px # Ensure kaleido version is 0.1.0.post1
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage



## Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

load_dotenv()   
assert os.environ['LANGCHAIN_API_KEY'], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ['OPENAI_API_KEY'], "Please set the OPENAI_API_KEY environment variable"

# Load your data (ensure the file path is correct)
DATA_DIR = '../../../data/raw/'
file_path = DATA_DIR + 'updated_data.xlsx'
data = pd.read_excel(file_path)

# Define the function
def create_BI_dash(bar_chart_field_name, pie_chart_field_name):
    """
    Creates an interactive dashboard with bar and pie charts.
    
    Parameters:
    bar_chart_field_name (str): The default field name to be used for the bar chart.
    pie_chart_field_name (str): The default field name to be used for the pie chart.
    """
    st.title("Interactive Dashboard with Bar and Pie Charts")

    # Dropdown options
    options = ['BuySell', 'OrderCapacity']

    # Dropdowns for selecting the fields dynamically
    selected_bar_chart_field = st.selectbox("Select field for Bar Chart:", options, index=options.index(bar_chart_field_name))
    selected_pie_chart_field = st.selectbox("Select field for Pie Chart:", options, index=options.index(pie_chart_field_name))

    # Bar Chart based on the selected field
    st.subheader(f"Bar Chart - {selected_bar_chart_field}")
    bar_chart_data = data[selected_bar_chart_field].value_counts().reset_index()
    bar_chart_data.columns = [selected_bar_chart_field, 'Count']
    bar_fig = px.bar(bar_chart_data, x=selected_bar_chart_field, y='Count', title=f"Count of {selected_bar_chart_field}")
    st.plotly_chart(bar_fig)

    # Pie Chart based on the selected field
    st.subheader(f"Pie Chart - {selected_pie_chart_field}")
    pie_chart_data = data[selected_pie_chart_field].value_counts().reset_index()
    pie_chart_data.columns = [selected_pie_chart_field, 'Count']
    pie_fig = px.pie(pie_chart_data, names=selected_pie_chart_field, values='Count', title=f"Count of {selected_pie_chart_field}")
    st.plotly_chart(pie_fig)

    # Chatbot
    openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])
    st.subheader("Chatbot")
    st.write("Ask me anything!")
    user_input = st.text_input("Your question:")
    if user_input:
        response = openai_llm.invoke(str(user_input))
        st.write("Response:")
        st.write(response.content)

    # Convert pie chart to image bytes
    st.subheader("Pie Chart Image Bytes")
    try:
        pie_chart_bytes =  base64.b64encode(pie_fig.to_image(format='png', engine='kaleido')).decode('utf-8')
    except Exception as e:
        st.write("Error generating image bytes: ", e)

    # pie_fig.write_image("pie_chart.png", engine="kaleido")
    # with open("pie_chart.png", "rb") as image_file:
    #     pie_chart_bytes = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Describe the charts in this dashboard."
                },
                ##TODO: Add more prompts wrt chart content and variables
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{pie_chart_bytes}", "type": "png"
                    },
                },
            ],
        )
    except Exception as e:
        st.write("Error creating message: ", e)

    response = openai_llm.invoke([message])
    st.write("Response:")
    st.write(response.content)


# Example call to the function with default values
create_BI_dash("BuySell", "OrderCapacity")
