from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls=PyPDFLoader)
    
    documents = loader.load() 
    
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download embedding model
def download_OpenAIEmbedings():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings


# Download embeeding model with huggingface that is free
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
 