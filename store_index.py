from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import Pinecone as langchain_Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
