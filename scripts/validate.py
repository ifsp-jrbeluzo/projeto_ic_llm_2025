import sys
from core.validate_core import run_validation

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def main():
    # Roda a validação a partir do diretório raiz
    run_validation(base_dir_path=".")

if __name__ == "__main__":
    main()
