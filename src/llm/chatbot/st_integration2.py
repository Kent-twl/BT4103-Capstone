## Script to test out the chatbot in a Streamlit app
import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from llm.chatbot.chatbot import Chatbot

DATA_DIR = ROOT_DIR + "data/raw"


@st.cache_resource
def create_chatbot():
    main_system_message = """
        The user you are assisting today is a trader at an Australian financial institution,
        interested in analyzing some trading data and other matters related to finance and trading.
    """
    return Chatbot(main_system_message, db_path=DATA_DIR + "/orders.db", agents=["sql", "yfinance"], with_memory=True)

st.title("Ask Chatbot Ad-Hoc Queries")

## Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

chatbot = create_chatbot()

## Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Enter your query here")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    response = chatbot.invoke(prompt)
    with st.chat_message("assistant"):
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
