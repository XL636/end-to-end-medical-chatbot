from src.helper import load_pdf, text_split, download_hugging_face_embeddings, pinecone_upsert
from pinecone import Pinecone
from langchain_pinecone import Pinecone as langchain_Pinecone
from dotenv import load_dotenv
import os
if __name__ == "__main__":
 load_dotenv()

 OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


 extracted_data = load_pdf("data/")
 text_chunks = text_split(extracted_data=extracted_data)
 embeddings = download_hugging_face_embeddings()

 pc = Pinecone(api_key=PINECONE_API_KEY)
 index = pc.Index("mchatbot")

 vectors_data = pinecone_upsert(chunks=text_chunks,embeddings=embeddings)

 batch_size = 150
 for i in range(0, len(vectors_data), batch_size):
    batch = vectors_data[i : i + batch_size]
    index.upsert(vectors=batch, namespace='ns1')
