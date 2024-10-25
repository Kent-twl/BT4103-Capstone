## Server for LangChain using FastAPI
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

load_dotenv()
assert os.environ['LANGCHAIN_API_KEY'], "Please set the LANGCHAIN_API_KEY environment variable"
assert os.environ['GROQ_API_KEY'], "Please set the GROQ_API_KEY environment variable"

## 1. Create prompt template
system_template = """
    Generate a report using the given template and data. Template:
    {report_template}
    Data:
"""
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{data}")
])

## 2. Create model
model = ChatGroq(model="llama3-8b-8192")

## 3. Create parser
parser = StrOutputParser()

## 4. Create chain
chain = prompt_template | model | parser

## 5. Create FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

## 6. Add chain routes
add_routes(
    app,
    chain,
    path="/chain",
)

## 7. Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)