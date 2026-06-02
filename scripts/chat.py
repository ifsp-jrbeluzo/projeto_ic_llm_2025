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

def load_question(path="question.json"):

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Question file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- LOAD CONTENT ---------------- #

config = load_config()

paths_cfg = config.get("paths", {})
rag_cfg = config.get("rag", {})
models_cfg = config.get("models", {})
ollama_cfg = config.get("ollama", {})
ingest_cfg = config.get("ingest", {})

prompt_template = load_prompt(paths_cfg.get("prompt_template", "prompt.txt"))

QUESTION_DATA = load_question(paths_cfg.get("question_file", "question.json"))

MAX_CONTEXT_CHARS = rag_cfg.get("max_context_chars", 6000)

N_RESULTS = rag_cfg.get("n_results", 3)
EMBEDDING_MODEL = models_cfg.get("embedding", "embeddinggemma")
CHAT_MODEL = models_cfg.get("chat", "gemma3:27b")

# ---------------- LOG DIR ---------------- #

LOG_DIR = Path(paths_cfg.get("logs", "./logs"))

LOG_DIR.mkdir(exist_ok=True)

# ---------------- CHROMA ---------------- #

# Constrói o caminho dinâmico com base nos parâmetros de chunking
chunk_size = ingest_cfg.get("chunk_size", 250)
chunk_overlap = ingest_cfg.get("chunk_overlap", 50)
db_base_path = Path(paths_cfg.get("database", "./database"))
db_path = db_base_path / f"db_{chunk_size}c_{chunk_overlap}o"

print(f"\nCarregando banco de dados: {db_path}")

chroma_client = chromadb.PersistentClient(
    path=str(db_path)
)

collections = chroma_client.list_collections()

if not collections:

    print(f"Nenhum banco encontrado em {db_path}")

    exit()

# ---------------- MODE ---------------- #

print("\nModo:")
print("[1] Um banco")
print("[2] Todos os bancos")

import sys
if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
    mode = "2"
else:
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

    # ---------------- COLLECTION ---------------- #

    collection = chroma_client.get_collection(
        col.name
    )

    metadata = collection.metadata or {}

    log_data = {
        "timestamp": timestamp,
        "collection": col.name,
        "original_filename": None,
        "chunk_size": metadata.get("chunk_size", "Desconhecido"),
        "chunk_overlap": metadata.get("chunk_overlap", "Desconhecido"),
        "interactions": [],
        "error": None
    }

    print(f"\n{'=' * 80}")
    print(f"PROCESSANDO: {col.name}")
    print(f"{'=' * 80}")

    try:
        intro_text = QUESTION_DATA.get("intro", "")
        questions_list = QUESTION_DATA.get("questions", [])
        
        if not questions_list:
            raise ValueError("Nenhuma pergunta encontrada no arquivo JSON.")

        for idx, q_item in enumerate(questions_list):
            print(f"\n--- Processando Pergunta {idx+1}/{len(questions_list)} ---")
            
            if isinstance(q_item, dict):
                q_text = f"{q_item.get('id', idx+1)}. {q_item.get('title', '')}:\n   - {q_item.get('instruction', '')}"
                if "options" in q_item:
                    q_text += "\n   dominios = [\n"
                    for opt in q_item["options"]:
                        q_text += f"      \"{opt}\",\n"
                    q_text += "   ]"
            else:
                q_text = str(q_item)

            full_question = f"{intro_text}\n\n{q_text}".strip()

            # ---------------- RETRIEVE CONTEXT ---------------- #

            # Busca no banco vetorial usando apenas a pergunta específica 
            # para evitar que a introdução confunda o embedding
            context, chunks = get_context(
                collection,
                q_text
            )

            if not log_data["original_filename"] and chunks:
                log_data["original_filename"] = chunks[0]["source"]

            # ---------------- FINAL PROMPT ---------------- #

            final_prompt = prompt_template.format(
                question=full_question,
                context=context
            )

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

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    for part in ollama_client.chat(
                        model=CHAT_MODEL,
                        messages=messages,
                        stream=True
                    ):
                        chunk = part["message"]["content"]
                        response_text += chunk
                        print(chunk, end="", flush=True)
                    break
                except Exception as e:
                    print(f"\n[Aviso] Falha na API na tentativa {attempt+1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    print("Aguardando 5 segundos antes de tentar novamente...")
                    time.sleep(5)
                    response_text = ""

            print("\n\nConcluído.\n")

            # ---------------- LOG RESULTS ---------------- #
            log_data["interactions"].append({
                "question_index": idx + 1,
                "query": full_question,
                "chunks": chunks,
                "prompt": final_prompt,
                "response": response_text
            })

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