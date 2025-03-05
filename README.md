# This is about medical chatbot

## steps to the project

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
Create ignore file - touch .gitignore
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
Finally use RetrievalQA to combine llm and retriever
'''