import sys
sys.stdout.reconfigure(encoding='utf-8')

import chromadb
from pathlib import Path
from core.config_manager import load_config, load_prompt, load_questions
from core.chat_core import run_chat_pipeline

def main():
    config = load_config()
    paths_cfg = config.get("paths", {})
    
    prompt_template = load_prompt(paths_cfg.get("prompt_template", "prompt.txt"))
    question_data = load_questions(paths_cfg.get("question_file", "question.json"))

    chroma_client = chromadb.PersistentClient(
        path=paths_cfg.get("database", "./database")
    )

    collections = chroma_client.list_collections()
    if not collections:
        print("Nenhum banco encontrado em ./database")
        sys.exit()

    print("\nModo:")
    print("[1] Um banco")
    print("[2] Todos os bancos")

    if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        mode = "2"
    else:
        mode = input("Escolha: ").strip()

    if mode == "1":
        print("\nBancos disponíveis:\n")
        for i, col in enumerate(collections):
            print(f"[{i}] {col.name}")
        idx = int(input("\nSelecione: "))
        selected_collection_names = [collections[idx].name]
    else:
        selected_collection_names = [col.name for col in collections]

    run_chat_pipeline(selected_collection_names, config, prompt_template, question_data)

if __name__ == "__main__":
    main()