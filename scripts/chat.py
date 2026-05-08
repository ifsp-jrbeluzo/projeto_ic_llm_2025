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

prompt_template = load_prompt("prompt.txt")

QUESTION = load_question("question.txt")

MAX_CONTEXT_CHARS = config.get(
    "max_context_chars",
    6000
)

# ---------------- LOG DIR ---------------- #

LOG_DIR = Path("logs")

LOG_DIR.mkdir(exist_ok=True)

# ---------------- CHROMA ---------------- #

chroma_client = chromadb.PersistentClient(
    path="./database"
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
    host=config.get(
        "ollama_host",
        "http://localhost:11434"
    ),
    headers={
        "Authorization":
        "Bearer " + config.get(
            "ollama_api_key",
            ""
        )
    }
)

# ---------------- GET CONTEXT ---------------- #

def get_context(collection, query):

    n_results = config.get("n_results", 3)

    print("\nQUERY RAG:\n")
    print(query)

    # ---------------- EMBEDDING ---------------- #

    embedding = ollama.embeddings(
        model=config.get(
            "embedding_model",
            "embeddinggemma"
        ),
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
        f"{safe_name}_{timestamp}.txt"
    )

    def write_log(text=""):

        with open(
            log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(text + "\n")

    # ---------------- COLLECTION ---------------- #

    collection = chroma_client.get_collection(
        col.name
    )

    print(f"\n{'=' * 80}")
    print(f"PROCESSANDO: {col.name}")
    print(f"{'=' * 80}")

    write_log("=" * 80)
    write_log(f"DATABASE: {col.name}")
    write_log("=" * 80)

    try:

        # ---------------- RETRIEVE CONTEXT ---------------- #

        context, chunks = get_context(
            collection,
            QUESTION
        )

        # ---------------- LOG QUESTION ---------------- #

        write_log("\nQUESTION:\n")

        write_log(QUESTION)

        # ---------------- LOG CHUNKS ---------------- #

        write_log("\nCHUNKS CAPTURADOS:\n")

        for c in chunks:

            write_log(
                f"{c['source']} | chunk {c['chunk']}"
            )

            write_log(c["text"])

            write_log("-" * 40)

        # ---------------- FINAL PROMPT ---------------- #

        final_prompt = prompt_template.format(
            question=QUESTION,
            context=context
        )

        # ---------------- DEBUG ---------------- #

        prompt_size = len(final_prompt)

        print(f"\nTAMANHO PROMPT: {prompt_size}")

        write_log("\nPROMPT SIZE:\n")

        write_log(str(prompt_size))

        # ---------------- LOG PROMPT ---------------- #

        write_log("\nFINAL PROMPT:\n")

        write_log(final_prompt)

        # ---------------- CHAT ---------------- #

        messages = [{
            "role": "user",
            "content": final_prompt
        }]

        response_text = ""

        print("\nGerando resposta...\n")

        for part in ollama_client.chat(
            model=config.get(
                "chat_model",
                "gemma3:27b"
            ),
            messages=messages,
            stream=True
        ):

            chunk = part["message"]["content"]

            response_text += chunk

            print(chunk, end="", flush=True)

        # ---------------- SAVE OUTPUT ---------------- #

        write_log("\nMODEL OUTPUT:\n")

        write_log(response_text)

        write_log("\n\n")

        print("\n\nConcluído.\n")

    except Exception as e:

        error_msg = f"\nERRO: {str(e)}"

        print(error_msg)

        write_log(error_msg)

    # ---------------- CLEAN MEMORY ---------------- #

    gc.collect()

    time.sleep(2)

# ---------------- END ---------------- #

print("\nProcessamento finalizado.")
print(f"Logs salvos em: {LOG_DIR}")