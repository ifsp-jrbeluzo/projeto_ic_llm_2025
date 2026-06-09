from pypdf import PdfReader
import re
from pathlib import Path

def pdf_to_text(file_path):
    """
    Extracts all text from a given PDF file path, excluding the bibliography/references section.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    # Procura por seções de referências/bibliografia para truncar o texto
    ref_patterns = [
        r'(?i)\n\s*references\s*\n',
        r'(?i)\n\s*refer[êˆ\^~]ncias\s*\n',
        r'(?i)\n\s*refer\s*[êˆ\^~]\s*encias\s*\n',
        r'(?i)\n\s*bibliography\s*\n'
    ]
    
    best_match_idx = -1
    for pattern in ref_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Pega o último match (geralmente a seção final de referências)
            last_match = matches[-1]
            # Garante que a seção de referências está na metade final do texto
            if last_match.start() > len(text) * 0.5:
                if best_match_idx == -1 or last_match.start() < best_match_idx:
                    best_match_idx = last_match.start()
                    
    if best_match_idx != -1:
        text = text[:best_match_idx]
        
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
