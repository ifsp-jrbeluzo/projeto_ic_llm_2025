import sys
sys.stdout.reconfigure(encoding='utf-8')

import ollama
from ollama import Client
import chromadb
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import gc
import time

# ---------------- LOAD FILES ---------------- #

def load_config(config_path="config.json"):

    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt(path="prompt.txt"):

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_question(path="question.txt"):

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Question file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------- LOAD CONTENT ---------------- #

config = load_config()

paths_cfg = config.get("paths", {})
rag_cfg = config.get("rag", {})
models_cfg = config.get("models", {})
ollama_cfg = config.get("ollama", {})

prompt_template = load_prompt(paths_cfg.get("prompt_template", "prompt.txt"))

QUESTION = load_question(paths_cfg.get("question_file", "question.txt"))

MAX_CONTEXT_CHARS = rag_cfg.get("max_context_chars", 6000)

N_RESULTS = rag_cfg.get("n_results", 3)
EMBEDDING_MODEL = models_cfg.get("embedding", "embeddinggemma")
CHAT_MODEL = models_cfg.get("chat", "gemma3:27b")

# ---------------- LOG DIR ---------------- #

LOG_DIR = Path(paths_cfg.get("logs", "./logs"))

LOG_DIR.mkdir(exist_ok=True)

# ---------------- CHROMA ---------------- #

chroma_client = chromadb.PersistentClient(
    path=paths_cfg.get("database", "./database")
)

collections = chroma_client.list_collections()

if not collections:

    print("Nenhum banco encontrado em ./database")

    exit()

# ---------------- MODE ---------------- #

print("\nModo:")
print("[1] Um banco")
print("[2] Todos os bancos")

mode = input("Escolha: ").strip()

if mode == "1":

    print("\nBancos disponíveis:\n")

    for i, col in enumerate(collections):

        print(f"[{i}] {col.name}")

    idx = int(input("\nSelecione: "))

    selected_collections = [collections[idx]]

else:

    selected_collections = collections

# ---------------- OLLAMA ---------------- #

ollama_client = Client(
    host=ollama_cfg.get("host", "http://localhost:11434"),
    headers={
        "Authorization": "Bearer " + ollama_cfg.get("api_key", "")
    }
)

# ---------------- GET CONTEXT ---------------- #

def get_context(collection, query):

    n_results = N_RESULTS

    print("\nQUERY RAG:\n")
    print(query)

    # ---------------- EMBEDDING ---------------- #

    embedding = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )["embedding"]

    # ---------------- VECTOR SEARCH ---------------- #

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]

    metas = results.get("metadatas", [[]])[0]

    grouped = defaultdict(list)

    raw_chunks = []

    # ---------------- ORGANIZE CHUNKS ---------------- #

    for doc, meta in zip(docs, metas):

        source = meta.get("source", "unknown")

        chunk = meta.get("chunk", "?")

        grouped[source].append((chunk, doc))

        raw_chunks.append({
            "source": source,
            "chunk": chunk,
            "text": doc
        })

    # ---------------- FORMAT CONTEXT ---------------- #

    context_parts = []

    for source, chunks in grouped.items():

        context_parts.append(
            f"\nDOCUMENTO: {source}\n"
        )

        sorted_chunks = sorted(
            chunks,
            key=lambda x:
            int(x[0])
            if str(x[0]).isdigit()
            else x[0]
        )

        for chunk, text in sorted_chunks:

            context_parts.append(
                f"- Chunk {chunk}:\n{text}\n"
            )

    final_context = "\n".join(context_parts)

    # ---------------- LIMIT CONTEXT ---------------- #

    final_context = final_context[
        :MAX_CONTEXT_CHARS
    ]

    return final_context, raw_chunks

# ---------------- MAIN LOOP ---------------- #

for col in selected_collections:

    # ---------------- LOG FILE ---------------- #

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_name = (
        col.name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    log_file = LOG_DIR / (
        f"{safe_name}_{timestamp}.json"
    )

    log_data = {
        "timestamp": timestamp,
        "collection": col.name,
        "query": QUESTION,
        "chunks": [],
        "prompt": "",
        "response": "",
        "error": None
    }

    # ---------------- COLLECTION ---------------- #

    collection = chroma_client.get_collection(
        col.name
    )

    print(f"\n{'=' * 80}")
    print(f"PROCESSANDO: {col.name}")
    print(f"{'=' * 80}")

    try:

        # ---------------- RETRIEVE CONTEXT ---------------- #

        context, chunks = get_context(
            collection,
            QUESTION
        )

        # ---------------- LOG CHUNKS ---------------- #

        log_data["chunks"] = chunks

        # ---------------- FINAL PROMPT ---------------- #

        final_prompt = prompt_template.format(
            question=QUESTION,
            context=context
        )

        log_data["prompt"] = final_prompt

        # ---------------- DEBUG ---------------- #

        prompt_size = len(final_prompt)

        print(f"\nTAMANHO PROMPT: {prompt_size}")

        # ---------------- CHAT ---------------- #

        messages = [{
            "role": "user",
            "content": final_prompt
        }]

        response_text = ""

        print("\nGerando resposta...\n")

        for part in ollama_client.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        ):

            chunk = part["message"]["content"]

            response_text += chunk

            print(chunk, end="", flush=True)

        # ---------------- SAVE OUTPUT ---------------- #

        log_data["response"] = response_text

        print("\n\nConcluído.\n")

    except Exception as e:

        error_msg = f"ERRO: {str(e)}"

        print(f"\n{error_msg}")

        log_data["error"] = error_msg
        
    finally:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

    # ---------------- CLEAN MEMORY ---------------- #

    gc.collect()

    time.sleep(2)

# ---------------- END ---------------- #

print("\nProcessamento finalizado.")
print(f"Logs salvos em: {LOG_DIR}")