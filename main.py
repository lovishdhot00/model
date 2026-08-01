from fastapi import FastAPI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
app=FastAPI()
origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/model/{input}')
def model(input):
    load_dotenv()
    api_key=os.getenv("api_token")
    llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",huggingfacehub_api_token=api_key,task="text-generation")
    model=ChatHuggingFace(llm=llm)
    return model.invoke(input).content