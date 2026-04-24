from pypdf import PdfReader
import ollama
import chromadb
from pathlib import Path
import json

# This script reads PDF files, converts them to text,
# splits into chunks, and stores embeddings in ChromaDB

def load_config(config_path="ingest_config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunkify(text, size=250, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks

def generate_database():
    client = chromadb.PersistentClient(path="./database")
    return client

def add_chunks(collection, text_chunks, file_name):
    for i, chunk in enumerate(text_chunks):
        embedding = ollama.embeddings(
            model="embeddinggemma",
            prompt=chunk
        )["embedding"]

        collection.add(
            ids=[f"{file_name}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": file_name,
                "chunk": i
            }]
        )

def main():
    config = load_config()

    chunk_size = config.get("chunk_size", 250)
    chunk_overlap = config.get("chunk_overlap", 50)
    papers_path = Path(config.get("papers_path", "./papers"))
    files = config.get("files", [])
    collection_name = config.get("collection_name", "papers")

    database = generate_database()
    collection = database.get_or_create_collection(name=collection_name)

    for file_name in files:
        file_path = papers_path / file_name

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Processing file: {file_name}")

        pdf_text = pdf_to_text(file_path)
        pdf_chunks = chunkify(pdf_text, size=chunk_size, overlap=chunk_overlap)
        add_chunks(collection, pdf_chunks, file_name)

    print("Database created successfully!")

if __name__ == "__main__":
    main()