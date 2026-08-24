from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, model_validator
from pathlib import Path
import json
import os
import shutil
import chromadb
from datetime import datetime
from server.runner import TaskRunner

router = APIRouter()
runner = TaskRunner()

BASE_DIR = Path(__file__).parent.parent

# Load config to get dynamic paths
config_path = BASE_DIR / "configs" / "config.json"
paths_cfg = {}
if config_path.exists():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            paths_cfg = json.load(f).get("paths", {})
    except Exception:
        pass

def resolve_path(cfg_val, default_rel):
    if not cfg_val:
        return BASE_DIR / default_rel
    path = Path(cfg_val)
    if path.is_absolute():
        return path
    return BASE_DIR / path

PAPERS_DIR = resolve_path(paths_cfg.get("papers"), "papers")
DATABASE_DIR = resolve_path(paths_cfg.get("database"), "database")
LOGS_DIR = resolve_path(paths_cfg.get("logs"), "logs")
RUNS_DIR = LOGS_DIR / "runs"

# Ensure directories exist
try:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

class IngestRequest(BaseModel):
    chunk_size: int
    chunk_overlap: int
    selected_pdfs: list[str]  # filenames or ["all"]

    @model_validator(mode="after")
    def check_chunk_bounds(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size deve ser maior que zero.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap nao pode ser negativo.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"O Overlap ({self.chunk_overlap}) nao pode ser maior ou igual ao Chunk Size ({self.chunk_size}).")
        return self

class ChatRequest(BaseModel):
    selected_databases: list[str]  # list of folder names, e.g. ["db_100c_50o"]
    chat_model: str
    n_results: int
    prompt_template: str
    question_json: dict  # JSON object containing "intro" and "questions" list

# Serve Main Validator Page
@router.get("/", response_class=HTMLResponse)
async def serve_index():
    validator_path = BASE_DIR / "validator.html"
    if not validator_path.exists():
        raise HTTPException(status_code=404, detail="validator.html não encontrado no diretório raiz.")
    
    with open(validator_path, "r", encoding="utf-8") as f:
        return f.read()

# API: List PDF papers
@router.get("/api/pdfs")
async def list_pdfs():
    try:
        pdfs = sorted([p.name for p in PAPERS_DIR.glob("*.pdf")])
        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: List databases and their collections
@router.get("/api/databases")
async def list_databases():
    try:
        databases = []
        # Find all folders starting with db_
        for folder in sorted(DATABASE_DIR.iterdir()):
            if folder.is_dir() and folder.name.startswith("db_"):
                # Load collections directly from sqlite to avoid locking the database file
                collections = []
                try:
                    db_file = folder / "chroma.sqlite3"
                    if db_file.exists():
                        import sqlite3
                        # Open in read-only mode to prevent write-locks
                        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM collections;")
                        rows = cursor.fetchall()
                        collections = [r[0] for r in rows]
                        conn.close()
                except Exception:
                    pass  # if folder is empty/invalid/locked
                
                databases.append({
                    "name": folder.name,
                    "collections": collections,
                    "path": str(folder)
                })
        return {"databases": databases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: List Ollama models
@router.get("/api/models")
async def list_models():
    try:
        config_path = BASE_DIR / "configs" / "config.json"
        ollama_host = "http://localhost:11434"
        ollama_key = ""
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                ollama_cfg = config_data.get("ollama", {})
                ollama_host = ollama_cfg.get("host", "http://localhost:11434")
                ollama_key = ollama_cfg.get("api_key", "")

        from ollama import Client
        headers = {"Authorization": f"Bearer {ollama_key}"} if ollama_key else None
        client = Client(host=ollama_host, headers=headers)
        res = client.list()
        
        models = []
        for m in res.get("models", []):
            name = m.get("model", m.get("name"))
            # Append -cloud suffix if fetched from the custom cloud endpoint
            if "ollama.com" in ollama_host and not name.endswith("-cloud"):
                name += "-cloud"
            models.append(name)
            
        return {"models": models}
    except Exception as e:
        return {
            "models": ["gemma3:4b", "gemma3:8b", "gemma3:27b", "llama3:8b", "llama3:70b", "phi4"],
            "error": str(e)
        }

# API: List historical runs
@router.get("/api/runs")
async def list_runs():
    try:
        runs = []
        if RUNS_DIR.exists():
            for folder in sorted(RUNS_DIR.iterdir(), reverse=True):
                if folder.is_dir() and folder.name.startswith("run_"):
                    run_config_path = folder / "run_config.json"
                    val_path = folder / "validation_results.json"
                    
                    config_data = {}
                    if run_config_path.exists():
                        try:
                            with open(run_config_path, "r", encoding="utf-8") as f:
                                config_data = json.load(f)
                        except Exception:
                            pass

                    overall_score = None
                    if val_path.exists():
                        try:
                            with open(val_path, "r", encoding="utf-8") as f:
                                val_results = json.load(f)
                                if val_results:
                                    scores = [item["overall_score"] for item in val_results if "overall_score" in item]
                                    if scores:
                                        overall_score = int(sum(scores) / len(scores))
                        except Exception:
                            pass

                    runs.append({
                        "id": folder.name,
                        "timestamp": config_data.get("timestamp", folder.name[4:19]),
                        "config": config_data,
                        "overall_score": overall_score
                    })
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: Get run details
@router.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run_path = RUNS_DIR / run_id
    if not run_path.exists() or not run_path.is_dir():
        raise HTTPException(status_code=404, detail="Run não encontrado")
    
    val_path = run_path / "validation_results.json"
    cfg_path = run_path / "run_config.json"
    
    val_data = []
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
            
    cfg_data = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
            
    return {
        "id": run_id,
        "config": cfg_data,
        "validation_results": val_data
    }

# API: Delete a run
@router.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    run_path = RUNS_DIR / run_id
    if not run_path.exists() or not run_path.is_dir():
        raise HTTPException(status_code=404, detail="Run não encontrado")
    try:
        shutil.rmtree(run_path)
        return {"success": True, "message": "Run deletado com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: Delete a database
@router.delete("/api/databases/{db_name}")
async def delete_database(db_name: str):
    db_path = DATABASE_DIR / db_name
    if not db_path.exists() or not db_path.is_dir():
        raise HTTPException(status_code=404, detail="Banco de dados não encontrado")
    try:
        shutil.rmtree(db_path)
        return {"success": True, "message": "Banco de dados deletado com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API: Get runner status and logs
@router.get("/api/status")
async def get_status():
    return runner.get_status()

# API: Kill active task
@router.post("/api/kill")
async def kill_task():
    killed = await runner.kill_active_task()
    return {"success": killed}

# API: Trigger Ingest Task
@router.post("/api/ingest")
async def trigger_ingest(req: IngestRequest):
    if runner.is_running:
        raise HTTPException(status_code=400, detail="Uma tarefa já está rodando em segundo plano.")

    # Prepare environment variables
    env_vars = {
        "INGEST_FILES": ",".join(req.selected_pdfs),
        "INGEST_CHUNK_SIZE": str(req.chunk_size),
        "INGEST_CHUNK_OVERLAP": str(req.chunk_overlap)
    }

    # Trigger async execution
    await runner.run_process("ingest.py", env_vars)
    return {"success": True, "message": "Ingestão iniciada em segundo plano."}

# API: Trigger Chat Pipeline Task
@router.post("/api/chat")
async def trigger_chat(req: ChatRequest):
    if runner.is_running:
        raise HTTPException(status_code=400, detail="Uma tarefa já está rodando em segundo plano.")

    if not req.selected_databases:
        raise HTTPException(status_code=400, detail="Nenhum banco de dados selecionado.")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # We will process each database selected sequentially or create a consolidated test folder.
    # To keep logging extremely organized: we create one single "run" folder for this test.
    # We will run chat.py inside this folder.
    run_dir_name = f"run_{timestamp_str}"
    target_run_dir = RUNS_DIR / run_dir_name
    target_run_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt template and questions to temporary files inside the run folder
    temp_prompt_path = target_run_dir / "temp_prompt.txt"
    temp_question_path = target_run_dir / "temp_question.json"

    with open(temp_prompt_path, "w", encoding="utf-8") as f:
        f.write(req.prompt_template)

    with open(temp_question_path, "w", encoding="utf-8") as f:
        json.dump(req.question_json, f, indent=4, ensure_ascii=False)

    # Write initial run_config.json
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "chat_model": req.chat_model,
        "n_results": req.n_results,
        "prompt_template": req.prompt_template,
        "question_json": req.question_json,
        "databases": req.selected_databases
    }
    with open(target_run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)

    # Executa chat.py uma vez por banco selecionado, sequencialmente
    env_sequence = []
    for db_name in req.selected_databases:
        db_path = DATABASE_DIR / db_name
        env_sequence.append({
            "RAG_DB_PATH": str(db_path),
            "RAG_COLLECTIONS": "all",
            "RAG_LOG_DIR": str(target_run_dir),
            "RAG_PROMPT_PATH": str(temp_prompt_path),
            "RAG_QUESTION_PATH": str(temp_question_path),
            "RAG_N_RESULTS": str(req.n_results),
            "RAG_CHAT_MODEL": req.chat_model
        })

    # Trigger sequence task in runner
    await runner.run_sequence_tasks(env_sequence, target_run_dir)
    
    return {"success": True, "message": "Pipeline RAG iniciado para os bancos selecionados."}
