from pypdf import PdfReader
from pathlib import Path
from ollama import Client

# Quantidade de caracteres do final do texto enviada à LLM para localizar
# o início da bibliografia. As referências sempre ficam no fim do artigo,
# então não é preciso (nem seria viável, por custo/contexto) mandar o texto
# inteiro - só a "cauda" onde a seção pode começar.
TAIL_CHARS = 30000

REFERENCES_DETECTION_PROMPT = """Abaixo está o final do texto de um artigo científico, extraído de um PDF. \
A extração pode conter ruído (números de página, cabeçalhos/rodapés repetidos, \
palavras quebradas por causa de fontes ou colunas do PDF original).

Cada linha começa com um número de índice entre colchetes, tipo [42]. Esse número \
não faz parte do texto original, é só uma referência para você responder.

Sua única tarefa: encontrar a linha onde começa a lista de referências bibliográficas \
do artigo (a lista final de citações - "References", "Referências", "Bibliography", \
numerada ou não, em qualquer idioma). Pode ser o próprio título da seção, ou, se o \
título estiver corrompido pela extração do PDF, a primeira linha que já faz parte da lista.

Responda APENAS com o número de índice dessa linha, sem mais nada.
Se não houver nenhuma lista de referências nesse trecho, responda apenas: NONE

TEXTO:
{numbered_lines}
"""


def _load_chat_client_and_model(config_path="configs/config.json"):
    """
    Carrega o client do Ollama e o modelo de chat configurados em config.json,
    para uso pontual (não é o cliente da pipeline de perguntas, é só para achar
    a bibliografia durante a ingestão).
    """
    import json

    path = Path(config_path)
    if not path.exists():
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ollama_cfg = config.get("ollama", {})
    model_name = config.get("models", {}).get("chat", "gemma3:27b")

    client = Client(
        host=ollama_cfg.get("host", "http://localhost:11434"),
        headers={"Authorization": "Bearer " + ollama_cfg.get("api_key", "")}
    )
    return client, model_name


def _find_references_start(text, client, model_name):
    """
    Pergunta à LLM em que ponto do texto começa a bibliografia, em vez de
    procurar por um título de seção com regex (frágil: PDFs quebram o título
    em várias linhas, usam idiomas/grafias diferentes, etc).

    Retorna o índice (em `text`) onde a bibliografia começa, ou -1 se a LLM
    não encontrar uma seção de referências no trecho analisado.
    """
    window_start = max(0, len(text) - TAIL_CHARS)
    window = text[window_start:]

    lines = window.split("\n")
    numbered_lines = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines))

    response = client.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": REFERENCES_DETECTION_PROMPT.format(numbered_lines=numbered_lines)
        }]
    )
    reply = response["message"]["content"].strip()

    if "NONE" in reply.upper():
        return -1

    import re
    match = re.search(r'\d+', reply)
    if not match:
        return -1

    line_idx = int(match.group())
    if line_idx < 0 or line_idx >= len(lines):
        return -1

    char_offset_in_window = sum(len(l) + 1 for l in lines[:line_idx])
    return window_start + char_offset_in_window


def pdf_to_text(file_path):
    """
    Extrai todo o texto de um PDF, removendo a seção de bibliografia/referências.
    O corte da bibliografia é feito pela própria LLM (configs/config.json ->
    models.chat), que analisa o final do texto extraído e aponta onde a lista
    de referências começa - não é mais um regex procurando o título da seção.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    try:
        client, model_name = _load_chat_client_and_model()
        if client is not None:
            ref_start_idx = _find_references_start(text, client, model_name)
            if ref_start_idx != -1:
                text = text[:ref_start_idx]
    except Exception as e:
        print(f"Aviso: falha ao detectar bibliografia via LLM, mantendo texto completo ({file_path}): {e}")

    # Salva a transcrição para conferência
    try:
        root_dir = Path(__file__).parent.parent.parent
        transcription_dir = root_dir / "papers_transcription"
        transcription_dir.mkdir(exist_ok=True)
        txt_path = transcription_dir / f"{Path(file_path).stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Erro ao salvar transcrição em papers_transcription: {e}")

    return text
