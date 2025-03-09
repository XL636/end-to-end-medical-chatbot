# This is about medical chatbot

## steps to the project
## First part
''' bash, create new environment
conda create -n mchatbot python=3.8 -y
'''
''' 
conda init bash, and reopen git bash
'''
'''
conda activate mchatbot
'''
'''
Create ignore file > touch .gitignore
'''
'''
write the requirements.txt (pip install -r requirements.txt)
'''
'''
create .env
'''


'''
connect Pinecone, load data and turn into different chunks. 
'''
'''
create a embedding, I am use OpenAI embedding
'''
manually upsert chunks into Pinecone
'''
'''
from langchain_pinecone import Pinecone to create a inference to Pinecone index
'''
'''
Finally use RetrievalQA to combine llm and retriever, I am using LLM, you can use differnt Free LLM from Huggingface
'''

## Second Part
'''
Create scr that inluce differen module you need to use in other py module
'''
'''
Create setup.py to write basic information for this project and make sure you can use local package 
'''
Create a store_index for store data.
'''
'''
Create app.py to run the website
'''
'''
you also need to create HTLM,CSS file with flask
'''
