import asyncio
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

class TaskRunner:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.active_task = None
        self.is_running = False
        self.task_type = None  # 'ingest' or 'chat'
        self.console_logs = []
        self.max_logs = 1000
        self.process = None
        self.start_time = None

    def get_status(self):
        return {
            "is_running": self.is_running,
            "task_type": self.task_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "logs": "\n".join(self.console_logs)
        }

    def append_log(self, text: str):
        self.console_logs.append(text)
        if len(self.console_logs) > self.max_logs:
            self.console_logs.pop(0)

    async def run_process(self, script_name: str, env_override: dict, post_script_args=None):
        async with self.lock:
            if self.is_running:
                raise RuntimeError("Uma tarefa já está em execução.")
            
            self.is_running = True
            self.task_type = "ingest" if "ingest" in script_name else "chat"
            self.console_logs = []
            self.start_time = datetime.now()
            self.active_log_dir = env_override.get("RAG_LOG_DIR")
            
            # Build env variables, inheriting from current environment
            process_env = os.environ.copy()
            process_env.update({k: str(v) for k, v in env_override.items()})
            # Force unbuffered output so we get logs in real time
            process_env["PYTHONUNBUFFERED"] = "1"
            process_env["PYTHONPATH"] = str(Path(__file__).parent.parent)

            script_path = str(Path(__file__).parent.parent / "scripts" / script_name)
            python_exe = sys.executable

            cmd_args = [script_path]
            if post_script_args:
                cmd_args.extend(post_script_args)

            self.append_log(f"--- INICIANDO PROCESSO: {script_name} ---")
            self.append_log(f"Comando: {python_exe} {' '.join(cmd_args)}")
            self.append_log(f"Env overrides: {env_override}\n")

            try:
                self.process = subprocess.Popen(
                    [python_exe] + cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=process_env
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.append_log(f"Erro ao iniciar subprocesso:\n{tb}")
                self.is_running = False
            # Start reading logs in background task
            asyncio.create_task(self._read_output(self.process))

    async def run_sequence_tasks(self, env_sequence: list[dict], target_run_dir: Path):
        async with self.lock:
            if self.is_running:
                raise RuntimeError("Uma tarefa já está em execução.")
            
            self.is_running = True
            self.task_type = "chat"
            self.console_logs = []
            self.start_time = datetime.now()
            self.active_log_dir = str(target_run_dir)
            
            # Start background sequence execution
            asyncio.create_task(self._run_sequence_loop(env_sequence, target_run_dir))

    async def _run_sequence_loop(self, env_sequence: list[dict], target_run_dir: Path):
        python_exe = sys.executable
        script_path = str(Path(__file__).parent.parent / "scripts" / "chat.py")
        
        try:
            for idx, env_override in enumerate(env_sequence, 1):
                self.append_log(f"\n========================================================")
                self.append_log(f"EXECUTANDO PIPELINE {idx}/{len(env_sequence)} no banco: {Path(env_override['RAG_DB_PATH']).name}")
                self.append_log(f"========================================================\n")
                
                process_env = os.environ.copy()
                process_env.update({k: str(v) for k, v in env_override.items()})
                process_env["PYTHONUNBUFFERED"] = "1"
                process_env["PYTHONPATH"] = str(Path(__file__).parent.parent)
                
                # Execute chat.py with "all" argument to skip interactive mode
                self.process = subprocess.Popen(
                    [python_exe, script_path, "all"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=process_env
                )
                
                # Read output for this run
                while True:
                    line = await asyncio.to_thread(self.process.stdout.readline)
                    if not line:
                        break
                    self.append_log(line.decode('utf-8', errors='replace').rstrip())
                
                exit_code = await asyncio.to_thread(self.process.wait)
                self.append_log(f"\nFinalizado pipeline {idx} (Código de saída: {exit_code})")
                if exit_code != 0:
                    self.append_log(f"Pipeline falhou. Abortando sequência.")
                    break
            
            # Run validator step at the end of the sequence if there are logs generated
            if Path(target_run_dir).exists():
                await self._run_validation_step(str(target_run_dir))
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.append_log(f"\nErro durante execução da sequência:\n{tb}")
        finally:
            self.is_running = False
            self.task_type = None
            self.process = None
            self.active_log_dir = None

    async def _read_output(self, process):
        try:
            while True:
                line = await asyncio.to_thread(process.stdout.readline)
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace').rstrip()
                self.append_log(decoded_line)
        except Exception as e:
            self.append_log(f"\nErro de leitura de logs: {str(e)}")
        finally:
            exit_code = await asyncio.to_thread(process.wait)
            self.append_log(f"\n--- PROCESSO FINALIZADO (Código de saída: {exit_code}) ---")
            
            # Post RAG Validation step
            if self.task_type == "chat" and exit_code == 0:
                # Trigger validator on the generated run log dir
                if self.active_log_dir and Path(self.active_log_dir).exists():
                    await self._run_validation_step(self.active_log_dir)

            self.is_running = False
            self.task_type = None
            self.process = None
            self.active_log_dir = None

    async def _run_validation_step(self, log_dir: str):
        self.append_log("\n--- INICIANDO PASSO DE VALIDAÇÃO / AUDITORIA ---")
        script_path = str(Path(__file__).parent.parent / "scripts" / "validate.py")
        python_exe = sys.executable
        
        process_env = os.environ.copy()
        process_env.update({
            "VALIDATE_LOGS_DIR": str(log_dir),
            "VALIDATE_OUT_PATH": str(Path(log_dir) / "validation_results.json"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path(__file__).parent.parent)
        })

        try:
            val_process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=process_env
            )
            while True:
                line = await asyncio.to_thread(val_process.stdout.readline)
                if not line:
                    break
                self.append_log(line.decode('utf-8', errors='replace').rstrip())
            await asyncio.to_thread(val_process.wait)
            self.append_log("--- VALIDAÇÃO / AUDITORIA FINALIZADA ---")
        except Exception as e:
            self.append_log(f"Erro ao executar passo de validação: {str(e)}")

    async def kill_active_task(self):
        if self.process:
            self.append_log("\n--- CANCELAMENTO SOLICITADO PELO USUÁRIO ---")
            try:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                self.append_log(f"Erro ao encerrar processo: {str(e)}")
            finally:
                self.is_running = False
                self.task_type = None
                self.process = None
                return True
        return False
