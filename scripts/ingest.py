import sys
from pathlib import Path
from core.config_manager import load_config
from core.ingest_core import run_ingestion

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def select_files(papers_path):
    pdf_files = list(papers_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        sys.exit()

    print("\nAvailable PDF files:")
    for i, file in enumerate(pdf_files):
        print(f"[{i}] {file.name}")

    if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        return pdf_files

    selection = input("\nSelect files (e.g. 0,2 or 'all'): ").strip()

    if selection.lower() == "all":
        return pdf_files

    indexes = [int(i) for i in selection.split(",") if i.strip().isdigit()]
    selected = [pdf_files[i] for i in indexes if i < len(pdf_files)]

    if not selected:
        print("Invalid selection.")
        sys.exit()

    return selected

def main():
    config = load_config()

    paths_cfg = config.get("paths", {})
    ingest_cfg = config.get("ingest", {})
    models_cfg = config.get("models", {})

    chunk_size = ingest_cfg.get("chunk_size", 250)
    chunk_overlap = ingest_cfg.get("chunk_overlap", 50)
    papers_path = Path(paths_cfg.get("papers", "./papers"))
    db_path = paths_cfg.get("database", "./database")
    model_name = models_cfg.get("embedding", "embeddinggemma")

    selected_files = select_files(papers_path)
    run_ingestion(selected_files, chunk_size, chunk_overlap, db_path, model_name)

if __name__ == "__main__":
    main()