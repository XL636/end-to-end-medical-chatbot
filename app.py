from flask import Flask, render_template, jsonify, request
from langchain_pinecone import Pinecone as langchain_Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os 
from langchain_huggingface import HuggingFaceEmbeddings
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddins = download_hugging_face_embeddings()

#connect pinecone
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index("mchatbot")

#load Index
docsearch = langchain_Pinecone.from_existing_index(index_name="mchatbot", 
                                                   embedding=embeddins,
                                                   namespace="ns1",  
                                                   text_key="text")

# Prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=['context','question'])

# Create LLM inference 
llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = "gpt-4.5-preview", temperature=0.8)

# Create RetrievalQA to combine llm and index of pinecone
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs={"prompt": PROMPT})

# Create Flask decorator for homepage
@app.route("/")
def index():
    return render_template("chat.html")

# Crete Flask decorator for user to ask question
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke(input)
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug= True)

