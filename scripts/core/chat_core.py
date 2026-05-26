import sys
import ollama
from ollama import Client
import chromadb
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import gc
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def get_context(collection, query, embedding_model, n_results=3, max_context_chars=6000, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log(f"\n[RAG] Buscando contexto para query: '{query[:60]}...'")

    # 1. Embedding da query usando ollama global
    embedding = ollama.embeddings(
        model=embedding_model,
        prompt=query
    )["embedding"]

    # 2. Busca vetorial
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    grouped = defaultdict(list)
    raw_chunks = []

    # 3. Organizar chunks
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", "?")
        grouped[source].append((chunk, doc))
        raw_chunks.append({
            "source": source,
            "chunk": chunk,
            "text": doc
        })

    # 4. Formatação do contexto
    context_parts = []
    for source, chunks in grouped.items():
        context_parts.append(f"\nDOCUMENTO: {source}\n")
        sorted_chunks = sorted(
            chunks,
            key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]
        )
        for chunk, text in sorted_chunks:
            context_parts.append(f"- Chunk {chunk}:\n{text}\n")

    final_context = "\n".join(context_parts)
    # Limitar o tamanho do contexto
    final_context = final_context[:max_context_chars]

    log(f"[RAG] Recuperados {len(raw_chunks)} chunks para contexto.")
    return final_context, raw_chunks

def run_chat_pipeline(
    selected_collection_names,
    config,
    prompt_template,
    question_data,
    db_path=None,
    log_callback=None,
    stream_callback=None
):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    paths_cfg = config.get("paths", {})
    rag_cfg = config.get("rag", {})
    models_cfg = config.get("models", {})
    ollama_cfg = config.get("ollama", {})

    max_context_chars = rag_cfg.get("max_context_chars", 6000)
    n_results = rag_cfg.get("n_results", 3)
    embedding_model = models_cfg.get("embedding", "embeddinggemma")
    chat_model = models_cfg.get("chat", "gemma3:27b")

    log_dir = Path(paths_cfg.get("logs", "./logs"))
    log_dir.mkdir(exist_ok=True)

    if db_path is None:
        db_path = paths_cfg.get("database", "./database")
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    ollama_client = Client(
        host=ollama_cfg.get("host", "http://localhost:11434"),
        headers={
            "Authorization": "Bearer " + ollama_cfg.get("api_key", "")
        }
    )

    log(f"Iniciando extração RAG para {len(selected_collection_names)} banco(s)...")

    for col_name in selected_collection_names:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = col_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        log_file = log_dir / f"{safe_name}_{timestamp}.json"

        log(f"\n{'='*80}\nPROCESSANDO: {col_name}\n{'='*80}")

        try:
            collection = chroma_client.get_collection(col_name)
            metadata = collection.metadata or {}
        except Exception as e:
            log(f"Erro ao carregar coleção {col_name}: {str(e)}")
            continue

        log_data = {
            "timestamp": timestamp,
            "collection": col_name,
            "original_filename": None,
            "chunk_size": metadata.get("chunk_size", "Desconhecido"),
            "chunk_overlap": metadata.get("chunk_overlap", "Desconhecido"),
            "interactions": [],
            "error": None
        }

        try:
            intro_text = question_data.get("intro", "")
            questions_list = question_data.get("questions", [])

            if not questions_list:
                raise ValueError("Nenhuma pergunta encontrada no question_data.")

            for idx, q_item in enumerate(questions_list):
                log(f"\n--- Pergunta {idx+1}/{len(questions_list)} ---")

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

                # Busca no banco vetorial
                context, chunks = get_context(
                    collection,
                    q_text,
                    embedding_model,
                    n_results,
                    max_context_chars,
                    log_callback
                )

                if not log_data["original_filename"] and chunks:
                    log_data["original_filename"] = chunks[0]["source"]

                final_prompt = prompt_template.format(
                    question=full_question,
                    context=context
                )

                log(f"Tamanho do Prompt: {len(final_prompt)} caracteres.")
                log("Gerando resposta via LLM...")

                messages = [{
                    "role": "user",
                    "content": final_prompt
                }]

                response_text = ""
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        for part in ollama_client.chat(
                            model=chat_model,
                            messages=messages,
                            stream=True
                        ):
                            chunk = part["message"]["content"]
                            response_text += chunk
                            if stream_callback:
                                stream_callback(chunk)
                            else:
                                print(chunk, end="", flush=True)
                        break
                    except Exception as e:
                        log(f"\n[Aviso] Falha na API na tentativa {attempt+1}/{max_retries}: {str(e)}")
                        if attempt == max_retries - 1:
                            raise e
                        log("Aguardando 5 segundos antes de tentar novamente...")
                        time.sleep(5)
                        response_text = ""

                log("\nPergunta concluída.")

                log_data["interactions"].append({
                    "question_index": idx + 1,
                    "query": full_question,
                    "chunks": chunks,
                    "prompt": final_prompt,
                    "response": response_text
                })

        except Exception as e:
            error_msg = f"ERRO no processamento: {str(e)}"
            log(f"\n{error_msg}")
            log_data["error"] = error_msg
        finally:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
            log(f"\nLog salvo em: {log_file.name}")

        gc.collect()
        time.sleep(1)

    log("\nProcessamento RAG finalizado!")
