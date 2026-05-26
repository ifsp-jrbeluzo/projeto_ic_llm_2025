from pypdf import PdfReader
import ollama
import chromadb
import sys
from pathlib import Path
import re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

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
    base = clean_filename(Path(file_path).name)
    name = f"{base}_{chunk_size}c_{overlap}o"
    return sanitize_name(name)

def generate_database(db_path="./database"):
    return chromadb.PersistentClient(path=db_path)

def add_chunks(collection, text_chunks, file_name, model_name, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    total = len(text_chunks)
    for i, chunk in enumerate(text_chunks):
        if i % 10 == 0 or i == total - 1:
            log(f"   -> Processando chunk {i+1}/{total}...")
        
        embedding = ollama.embeddings(
            model=model_name,
            prompt=chunk
        )["embedding"]

        collection.upsert(
            ids=[f"{file_name}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": file_name,
                "chunk": i
            }]
        )

def run_ingestion(selected_files, chunk_size, chunk_overlap, db_path, model_name, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log(f"Iniciando ingestão de {len(selected_files)} arquivos...")
    db = generate_database(db_path)

    for file_path in selected_files:
        file_path = Path(file_path)
        if not file_path.exists():
            log(f"Erro: Arquivo não encontrado: {file_path}")
            continue

        collection_name = generate_collection_name(
            file_path,
            chunk_size,
            chunk_overlap
        )

        log(f"\nCriando/Carregando coleção: {collection_name}")
        log(f"Extraindo texto de: {file_path.name}")

        try:
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
            log(f"Extraído com sucesso! Gerados {len(chunks)} chunks.")
            
            add_chunks(collection, chunks, file_path.name, model_name, log_callback)
            log(f"Coleção {collection_name} populada com sucesso!")
        except Exception as e:
            log(f"Erro ao processar arquivo {file_path.name}: {str(e)}")

    log("\nProcessamento de banco vetorial finalizado com sucesso!")
