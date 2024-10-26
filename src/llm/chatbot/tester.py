## Script for testing reliability of chatbot answers
import os
import sys
from llm.chatbot.sql_agent import SQLAgent
from utils.create_orders_db import create_orders_db

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/src")
sys.path.append(os.path.dirname(__file__))

DATA_DIR = ROOT_DIR + "data/raw"

def base_test():
    create_orders_db(source_file=DATA_DIR+"/updated_data.xlsx", dest_dir=DATA_DIR)
    sql_chatbot = SQLAgent(db_path=DATA_DIR+"/orders.db")

    response = sql_chatbot.invoke("Do you observe any trends in the price of the securities over the past week?")
    print(response)

    stream = sql_chatbot.stream("Which account code has the most buy and sell value respectively?")
    for message in stream:
        print(message)


if __name__ == "__main__":
    base_test()