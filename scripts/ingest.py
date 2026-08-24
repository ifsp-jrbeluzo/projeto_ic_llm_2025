import ollama
import chromadb
from pathlib import Path
import json
import re
import os
import sys

# Import core modules
sys.path.append(str(Path(__file__).parent.parent))
from scripts.core.pdf_extractor import pdf_to_text
from scripts.core.chunker import chunkify

# ---------------- CONFIG ---------------- #

def load_config(config_path="configs/config.json"):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- NAMING ---------------- #

def clean_filename(name):
    name = name.lower()
    name = name.replace(".pdf", "")

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

def generate_database(db_path="./database"):
    return chromadb.PersistentClient(path=db_path)

def add_chunks(collection, text_chunks, file_name, model_name):
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(text_chunks):
        embedding = ollama.embeddings(
            model=model_name,
            prompt=chunk
        )["embedding"]
        
        ids.append(f"{file_name}_{i}")
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({
            "source": file_name,
            "chunk": i
        })
        
    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

# ---------------- FILE SELECT ---------------- #

def select_files(papers_path):
    pdf_files = sorted(list(papers_path.glob("*.pdf")))

    if not pdf_files:
        print("No PDF files found.")
        exit()

    # Check environment variable first (for web app runner)
    env_selection = os.environ.get("INGEST_FILES")
    if env_selection:
        if env_selection.lower() == "all":
            return pdf_files
        parts = [p.strip() for p in env_selection.split(",")]
        selected = []
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if idx < len(pdf_files):
                    selected.append(pdf_files[idx])
            else:
                for f in pdf_files:
                    if f.name == p or f.stem == p:
                        selected.append(f)
        return selected

    # Check CLI sys.argv
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "all":
            return pdf_files
        parts = [p.strip() for p in arg.split(",")]
        selected = []
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if idx < len(pdf_files):
                    selected.append(pdf_files[idx])
        if selected:
            return selected

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

    paths_cfg = config.get("paths", {})
    ingest_cfg = config.get("ingest", {})
    models_cfg = config.get("models", {})

    # Check environment variables for chunking params
    env_chunk_size = os.environ.get("INGEST_CHUNK_SIZE")
    env_chunk_overlap = os.environ.get("INGEST_CHUNK_OVERLAP")

    chunk_size = int(env_chunk_size) if env_chunk_size and env_chunk_size.isdigit() else ingest_cfg.get("chunk_size", 250)
    chunk_overlap = int(env_chunk_overlap) if env_chunk_overlap and env_chunk_overlap.isdigit() else ingest_cfg.get("chunk_overlap", 50)
    
    papers_path = Path(paths_cfg.get("papers", "./papers"))
    db_base_path = Path(paths_cfg.get("database", "./database"))

    # Subpasta isolada por configuração de chunking: db_{chunk_size}c_{chunk_overlap}o
    db_path = db_base_path / f"db_{chunk_size}c_{chunk_overlap}o"
    db_path.mkdir(parents=True, exist_ok=True)

    model_name = models_cfg.get("embedding", "embeddinggemma")

    print(f"\nUsando banco de dados: {db_path}")

    selected_files = select_files(papers_path)
    db = generate_database(str(db_path))

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
        add_chunks(collection, chunks, file_path.name, model_name)

    print("\nAll databases created successfully!")

if __name__ == "__main__":
    main()