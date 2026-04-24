from pypdf import PdfReader
import ollama
import chromadb
from pathlib import Path
import json
import re

# ---------------- CONFIG ---------------- #

def load_config(config_path="config.json"):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- PDF ---------------- #

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- CHUNK ---------------- #

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

# ---------------- NAMING ---------------- #

def clean_filename(name):
    name = name.lower()
    name = name.replace(".pdf", "")
    
    name = re.sub(r"\(\d+\)", "", name)

    name = re.sub(r"[^a-z0-9 ]", "", name)

    name = "_".join(name.split())

    return name[:40]

def sanitize_name(name):
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name

def generate_collection_name(file_path, chunk_size, overlap):
    base = clean_filename(file_path.name)
    name = f"{base}_{chunk_size}c_{overlap}o"
    return sanitize_name(name)

# ---------------- DB ---------------- #

def generate_database():
    return chromadb.PersistentClient(path="./database")

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

# ---------------- FILE SELECT ---------------- #

def select_files(papers_path):
    pdf_files = list(papers_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        exit()

    print("\nAvailable PDF files:")
    for i, file in enumerate(pdf_files):
        print(f"[{i}] {file.name}")

    selection = input("\nSelect files (e.g. 0,2 or 'all'): ").strip()

    if selection.lower() == "all":
        return pdf_files

    indexes = [int(i) for i in selection.split(",") if i.strip().isdigit()]
    selected = [pdf_files[i] for i in indexes if i < len(pdf_files)]

    if not selected:
        print("Invalid selection.")
        exit()

    return selected

# ---------------- MAIN ---------------- #

def main():
    config = load_config()

    chunk_size = config.get("chunk_size", 250)
    chunk_overlap = config.get("chunk_overlap", 50)
    papers_path = Path(config.get("papers_path", "./papers"))

    selected_files = select_files(papers_path)
    db = generate_database()

    for file_path in selected_files:
        collection_name = generate_collection_name(
            file_path,
            chunk_size,
            chunk_overlap
        )

        print(f"\nCreating collection: {collection_name}")
        print(f"Processing: {file_path.name}")

        collection = db.get_or_create_collection(
            name=collection_name,
            metadata={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_docs": 1
            }
        )

        text = pdf_to_text(file_path)
        chunks = chunkify(text, chunk_size, chunk_overlap)
        add_chunks(collection, chunks, file_path.name)

    print("\nAll databases created successfully!")

if __name__ == "__main__":
    main()