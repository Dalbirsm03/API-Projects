# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from core import generate_sql_and_execute

app = FastAPI(title="SQLAGE API")

class QueryInput(BaseModel):
    query: str

@app.post("/generate-sql")
def generate_sql(data: QueryInput):
    return generate_sql_and_execute(data.query)
