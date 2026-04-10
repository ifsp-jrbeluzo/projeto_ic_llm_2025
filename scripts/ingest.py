from pypdf import PdfReader
import ollama
import chromadb
from pathlib import Path

# This script take the papers pdf, convert to raw text and generate a chroma database with the dataset

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


papers_path = "./papers"
papers_folder = Path(papers_path)

database = generate_database()
collection = database.get_or_create_collection(name="papers")

#for pdf_file in papers_folder.glob("*.pdf"):
print(f"Processing artigo")
pdf_text = pdf_to_text("./papers/artigo.pdf")
pdf_chunks = chunkify(pdf_text)
add_chunks(collection, pdf_chunks, "artigo")

print("Database created successfully!")