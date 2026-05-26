import sys
import os
import json
import time
import threading
import re
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Adiciona a pasta 'scripts' ao path do sistema para permitir importações do 'core'
sys.path.append(str(Path(__file__).parent / "scripts"))

from core.config_manager import (
    load_config, save_config,
    load_prompt, save_prompt,
    load_questions, save_questions,
    load_ground_truth, save_ground_truth
)
from core.ingest_core import run_ingestion, generate_collection_name
from core.chat_core import run_chat_pipeline
from core.validate_core import run_validation

# Configurações do buffer de logs em memória
LOGS_BUFFER = []
LOGS_LOCK = threading.Lock()
PIPELINE_STATUS = {
    "running": False,
    "task": None,  # 'ingest', 'chat', 'validate'
    "progress": ""
}

def add_log(msg):
    with LOGS_LOCK:
        timestamp = time.strftime("[%H:%M:%S]")
        LOGS_BUFFER.append(f"{timestamp} {msg}")
        # Limita a 500 linhas em memória
        if len(LOGS_BUFFER) > 500:
            LOGS_BUFFER.pop(0)
    # Também printa no terminal para acompanhamento
    print(f"{timestamp} {msg}")

class UnifiedPipelineServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Desativa os logs de requisição padrão no console para não sujar a tela
        pass

    def send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_OPTIONS(self):
        # Suporte a CORS para requisições do navegador
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        # Faz o parse da URL e parâmetros de query
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        path_only = parsed_url.path

        # Rotas estáticas
        if path_only == '/' or path_only == '/index.html':
            self.serve_static_file('index.html', 'text/html')
            return
        elif path_only == '/validator.html':
            self.serve_static_file('validator.html', 'text/html')
            return

        # Rotas de API
        if path_only == '/api/status':
            with LOGS_LOCK:
                response = {
                    "running": PIPELINE_STATUS["running"],
                    "task": PIPELINE_STATUS["task"],
                    "logs": list(LOGS_BUFFER)
                }
            self.send_json(200, response)
            return

        elif path_only == '/api/config':
            try:
                self.send_json(200, load_config())
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/prompt':
            try:
                text = load_prompt()
                self.send_json(200, {"prompt": text})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/questions':
            try:
                self.send_json(200, load_questions())
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/ground_truth':
            try:
                self.send_json(200, load_ground_truth())
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/databases':
            try:
                config = load_config()
                paths = config.get("paths", {})
                db_base_path = Path(paths.get("database", "./database"))
                
                databases = []
                if db_base_path.exists():
                    for item in db_base_path.iterdir():
                        if item.is_dir() and item.name.startswith("db_"):
                            databases.append(item.name)
                databases.sort()
                self.send_json(200, {"databases": databases})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/papers':
            try:
                config = load_config()
                paths = config.get("paths", {})
                ingest_cfg = config.get("ingest", {})
                
                # Se passou o parametro ?database=db_600c_150o
                selected_db = query_params.get("database", [None])[0]
                db_base_path = Path(paths.get("database", "./database"))
                
                if selected_db and selected_db != "default":
                    db_path = db_base_path / selected_db
                    # Tenta ler chunk/overlap do nome do diretório
                    match = re.search(r'db_(\d+)c_(\d+)o', selected_db)
                    if match:
                        chunk_size = int(match.group(1))
                        chunk_overlap = int(match.group(2))
                    else:
                        chunk_size = ingest_cfg.get("chunk_size", 250)
                        chunk_overlap = ingest_cfg.get("chunk_overlap", 50)
                else:
                    chunk_size = ingest_cfg.get("chunk_size", 250)
                    chunk_overlap = ingest_cfg.get("chunk_overlap", 50)
                    db_path = db_base_path / f"db_{chunk_size}c_{chunk_overlap}o"

                papers_path = Path(paths.get("papers", "./papers"))

                import chromadb
                collections = []
                if db_path.exists():
                    try:
                        chroma_client = chromadb.PersistentClient(path=str(db_path))
                        collections = [c.name for c in chroma_client.list_collections()]
                    except Exception:
                        pass

                papers_list = []
                if papers_path.exists():
                    for f in papers_path.glob("*.pdf"):
                        col_name = generate_collection_name(f, chunk_size, chunk_overlap)
                        papers_list.append({
                            "name": f.name,
                            "size_bytes": f.stat().st_size,
                            "collection_name": col_name,
                            "ingested": col_name in collections
                        })
                
                self.send_json(200, {
                    "papers": papers_list,
                    "collections": collections
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif path_only == '/api/results':
            try:
                config = load_config()
                paths = config.get("paths", {})
                logs_dir = Path(paths.get("logs", "./logs"))
                res_file = logs_dir / "validation_results.json"
                
                if res_file.exists():
                    with open(res_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.send_json(200, data)
                else:
                    self.send_json(200, [])
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        # Rota não encontrada
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        # Suporte a CORS
        if self.path == '/api/config':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                save_config(data=data)
                add_log("Configurações atualizadas via painel.")
                self.send_json(200, {"status": "success"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/prompt':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                text = data.get("prompt", "")
                save_prompt(text=text)
                add_log("Template de prompt atualizado via painel.")
                self.send_json(200, {"status": "success"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/questions':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                save_questions(data=data)
                add_log("Perguntas atualizadas via painel.")
                self.send_json(200, {"status": "success"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/ground_truth':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                save_ground_truth(data=data)
                add_log("Gabarito de validação atualizado via painel.")
                self.send_json(200, {"status": "success"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/run/ingest':
            if PIPELINE_STATUS["running"]:
                self.send_json(400, {"error": "Outro pipeline já está em execução."})
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                files = data.get("files", [])
                chunk_size = data.get("chunk_size")
                chunk_overlap = data.get("chunk_overlap")
                
                # Executa em thread assíncrona
                threading.Thread(
                    target=self.bg_ingest_thread, 
                    args=(files, chunk_size, chunk_overlap)
                ).start()
                self.send_json(200, {"status": "started"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/run/chat':
            if PIPELINE_STATUS["running"]:
                self.send_json(400, {"error": "Outro pipeline já está em execução."})
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                collections = data.get("collections", [])
                selected_db = data.get("database")
                
                # Executa em thread assíncrona
                threading.Thread(
                    target=self.bg_chat_thread, 
                    args=(collections, selected_db)
                ).start()
                self.send_json(200, {"status": "started"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return

        elif self.path == '/api/run/validate':
            if PIPELINE_STATUS["running"]:
                self.send_json(400, {"error": "Outro pipeline já está em execução."})
                return
            
            threading.Thread(target=self.bg_validate_thread).start()
            self.send_json(200, {"status": "started"})
            return

        # Rota não encontrada
        self.send_response(404)
        self.end_headers()

    def serve_static_file(self, filename, content_type):
        path = Path(__file__).parent / filename
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Static file not found.")
            return

        self.send_response(200)
        self.send_header('Content-Type', f'{content_type}; charset=utf-8')
        self.end_headers()
        with open(path, 'rb') as f:
            self.wfile.write(f.read())

    # --- Métodos de Execução em Background ---

    def bg_ingest_thread(self, files, custom_chunk_size=None, custom_chunk_overlap=None):
        global PIPELINE_STATUS, LOGS_BUFFER
        with LOGS_LOCK:
            PIPELINE_STATUS["running"] = True
            PIPELINE_STATUS["task"] = "ingest"
            LOGS_BUFFER.clear()

        try:
            config = load_config()
            paths = config.get("paths", {})
            ingest_cfg = config.get("ingest", {})
            models_cfg = config.get("models", {})

            chunk_size = custom_chunk_size if custom_chunk_size is not None else ingest_cfg.get("chunk_size", 250)
            chunk_overlap = custom_chunk_overlap if custom_chunk_overlap is not None else ingest_cfg.get("chunk_overlap", 50)
            
            papers_path = Path(paths.get("papers", "./papers"))
            db_base_path = Path(paths.get("database", "./database"))
            
            # Subpasta parametrizada para isolamento dos bancos
            db_path = db_base_path / f"db_{chunk_size}c_{chunk_overlap}o"
            model_name = models_cfg.get("embedding", "embeddinggemma")

            selected_paths = []
            if files == "all":
                selected_paths = list(papers_path.glob("*.pdf"))
            else:
                for f in files:
                    selected_paths.append(papers_path / f)

            run_ingestion(
                selected_files=selected_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_path=str(db_path),
                model_name=model_name,
                log_callback=add_log
            )
        except Exception as e:
            add_log(f"ERRO CRÍTICO NO PIPELINE DE INGESTÃO: {str(e)}")
        finally:
            with LOGS_LOCK:
                PIPELINE_STATUS["running"] = False
                PIPELINE_STATUS["task"] = None

    def bg_chat_thread(self, selected_cols, selected_db=None):
        global PIPELINE_STATUS, LOGS_BUFFER
        with LOGS_LOCK:
            PIPELINE_STATUS["running"] = True
            PIPELINE_STATUS["task"] = "chat"
            LOGS_BUFFER.clear()

        try:
            config = load_config()
            paths = config.get("paths", {})
            prompt_template = load_prompt(paths.get("prompt_template", "configs/prompt.txt"))
            question_data = load_questions(paths.get("question_file", "configs/question.json"))

            db_base_path = Path(paths.get("database", "./database"))
            
            # Calcula o db_path do banco selecionado ou usa o default
            if selected_db and selected_db != "default":
                db_path = db_base_path / selected_db
            else:
                ingest_cfg = config.get("ingest", {})
                chunk_size = ingest_cfg.get("chunk_size", 250)
                chunk_overlap = ingest_cfg.get("chunk_overlap", 50)
                db_path = db_base_path / f"db_{chunk_size}c_{chunk_overlap}o"

            import chromadb
            chroma_client = chromadb.PersistentClient(path=str(db_path))
            
            if selected_cols == "all":
                cols = [c.name for c in chroma_client.list_collections()]
            else:
                cols = selected_cols

            run_chat_pipeline(
                selected_collection_names=cols,
                config=config,
                prompt_template=prompt_template,
                question_data=question_data,
                db_path=str(db_path),
                log_callback=add_log,
                stream_callback=lambda token: add_log(token) if token.strip() == "" else add_log(f"   -> LLM: {token}")
            )
        except Exception as e:
            add_log(f"ERRO CRÍTICO NO PIPELINE RAG: {str(e)}")
        finally:
            with LOGS_LOCK:
                PIPELINE_STATUS["running"] = False
                PIPELINE_STATUS["task"] = None

    def bg_validate_thread(self):
        global PIPELINE_STATUS, LOGS_BUFFER
        with LOGS_LOCK:
            PIPELINE_STATUS["running"] = True
            PIPELINE_STATUS["task"] = "validate"
            LOGS_BUFFER.clear()

        try:
            run_validation(base_dir_path=".", log_callback=add_log)
        except Exception as e:
            add_log(f"ERRO CRÍTICO NA VALIDAÇÃO: {str(e)}")
        finally:
            with LOGS_LOCK:
                PIPELINE_STATUS["running"] = False
                PIPELINE_STATUS["task"] = None

def run(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, UnifiedPipelineServer)
    print(f"\nServidor rodando em: http://localhost:{port}")
    print("Acesse no seu navegador para abrir o painel de controle.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor finalizado pelo usuário.")
        httpd.server_close()

if __name__ == '__main__':
    run()
