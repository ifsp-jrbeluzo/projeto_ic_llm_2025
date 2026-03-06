from pypdf import PdfReader
import ollama
import chromadb
from chromadb.config import Settings

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunkify(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def generate_database():
    client = chromadb.PersistentClient(path="./database")
    return client

def create_collection(database, text_chunks, collection_name):
    collection = database.create_collection(name=collection_name)
    for i, chunk in enumerate(text_chunks):
        embedding = ollama.embeddings(
            model="embeddinggemma",
            prompt=chunk
        )["embedding"]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )


pdf_path = input("Type the PDF path: ")
pdf_text = pdf_to_text(pdf_path)
pdf_chunks = chunkify(pdf_text)
database = generate_database()
create_collection(database, pdf_chunks, "collecA")
print("Database and collection created successfully!")