from sqlalchemy import create_engine, MetaData
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import threading
import pymysql
import pyaudio
import wave
import os
import re

# Load environment variables
load_dotenv()

from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
genai_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

from urllib.parse import quote_plus


# Configure Google Generative AI
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

# Create the engine using the environment variables
encoded_password = quote_plus(DB_PASSWORD)

# Configure Google Generative AI
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

# Create the engine using the environment variables
engine = create_engine(f'mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}')
# Function to fetch column and table names
def fetch_table_and_column_names(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    table_column_info = {}
    
    for table_name in metadata.tables.keys():
        table = metadata.tables[table_name]
        columns = [column.name for column in table.columns]
        table_column_info[table_name] = columns
    
    return table_column_info

table_column_info = fetch_table_and_column_names(engine)

formatted_table_info = "\n".join(
    [f"Table: {table}\nColumns:\n" + "\n".join([f"  - {col}" for col in cols]) for table, cols in table_column_info.items()]
)


# Function to create MySQL Query
def get_gemini_sql_query(user_text):
    input_prompt = f""" 
    As an expert in MySQL query development, your task is to create sophisticated queries based on the provided table structures and column details {formatted_table_info}. Your goal is to craft queries that perform comprehensive data retrieval operations, utilizing joins and other advanced SQL techniques as necessary.
    Make sure your queries are robust, precise, and MySQL-compatible, demonstrating your proficiency in MySQL data retrieval operations.
    Do not provide any explanation and all you just have to create mysql queries
    """
    ai_response = model.generate_content([input_prompt, user_text])
    return ai_response.text

def clean_sql_query(query):
    query = query.strip().strip("```sql").strip("```")
    query = re.sub(r'\bmysql\b', '', query, flags=re.IGNORECASE)
    return query

def generate_sql_and_execute(user_input: str) -> dict:
    prompt = f""" 
        1.Generate only most accurate MySQL query 
        2.Use the provided {formatted_table_info} for table structures and column details.
        3.Focus on creating accurate and MySQL-compatible query.
        4.Apply advanced techniques like joins, subqueries, aggregations, and groupings as needed.
        5.Avoid explanations—deliver only the required SQL query.
    """

    response = model.generate_content([prompt, user_input])
    generated_sql = clean_sql_query(response.text)
    df = pd.read_sql_query(generated_sql, engine)
    result_list = df.to_dict(orient="records")  # ✅ Ensures it's a list of dicts
    return {
            "query": generated_sql,
            "result": result_list,
        }
    