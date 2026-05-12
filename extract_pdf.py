import sys
from pypdf import PdfReader

reader = PdfReader('c:\\Users\\Aluno\\Documents\\projeto_ic_llm_2025\\main_paper\\Reviso_Sistemtica_de_Viso_Computacional_para_Veculos_Autnomos_Agrcolas.pdf')
with open('main_paper_text.txt', 'w', encoding='utf-8') as f:
    for page in reader.pages:
        f.write(page.extract_text() + '\n')
