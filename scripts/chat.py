import ollama
from ollama import Client
import chromadb
import re
import json
from pathlib import Path

# ---------------- CONFIG ---------------- #

def load_config(config_path="config.json"):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# ---------------- COLORS ---------------- #

text_reset   = "\033[0m"
text_yellow  = "\033[33m"
text_green   = "\033[32m"
text_magenta = "\033[35m"

# ---------------- CHROMA ---------------- #

chroma_client = chromadb.PersistentClient(path="./database")
collections = chroma_client.list_collections()

if not collections:
    print("No databases found. Run ingest.py first.")
    exit()

print("\nAvailable databases:\n")

collection_map = []

for i, col in enumerate(collections):
    col_name = col.name
    metadata = col.metadata or {}

    chunk_size = metadata.get("chunk_size", "?")
    overlap = metadata.get("chunk_overlap", "?")

    display_name = re.sub(r"_\d+c_\d+o$", "", col_name)

    print(f"[{i}] {display_name} <{chunk_size} chunks> <{overlap} overlap>")

    collection_map.append(col_name)

while True:
    try:
        choice = int(input("\nSelect database: "))
        if 0 <= choice < len(collection_map):
            break
        else:
            print("Invalid option.")
    except:
        print("Enter a valid number.")

collection_name = collection_map[choice]
collection = chroma_client.get_collection(collection_name)

print(f"\nUsing database: {collection_name}")

# ---------------- OLLAMA ---------------- #

ollama_client = Client(
    host=config.get("ollama_host", "https://ollama.com"),
    headers={
        "Authorization": "Bearer " + config.get("ollama_api_key", "")
    }
)

# ---------------- CONTEXT ---------------- #

def get_context(prompt):
    n_results = config.get("n_results", 3)

    query_embedding = ollama.embeddings(
        model=config.get("embedding_model", "embeddinggemma"),
        prompt=prompt
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    context_chunks = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta["source"]
        chunk = meta["chunk"]

        context_chunks.append(
            f"[Source: {source} | Chunk {chunk}]\n{doc}"
        )

    return "\n\n".join(context_chunks)

# ---------------- CHAT LOOP ---------------- #

while True:
    prompt = input(f"\n{text_yellow}Prompt:{text_reset} ")

    if prompt == "/end":
        break

    context = get_context(prompt)

    final_prompt = f"""
You are a scientist working on a systematic review.

Rules:
- Use ONLY the provided context
- Do not hallucinate
- If unknown, say you don't know
- Answer in Portuguese

<context>
{context}
</context>

<question>
{prompt}
</question>
"""

    print(f"\n{text_magenta}Final Prompt:{text_reset} {final_prompt}")

    messages = [{"role": "user", "content": final_prompt}]

    response_text = ""

    for part in ollama_client.chat(
        model=config.get("chat_model", "gemma3:27b"),
        messages=messages,
        stream=True
    ):
        chunk = part["message"]["content"]
        response_text += chunk
        print(chunk, end="", flush=True)

    print(f"\n{text_green}Done{text_reset}")