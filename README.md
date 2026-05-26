# LLM RAG - Revisão Sistemática de Literatura Assistida por IA

Este é um sistema de **Revisão Sistemática de Literatura (RSL)** automatizada e assistida por Inteligência Artificial, aplicada à área de Visão Computacional e Inteligência Artificial na robótica agrícola. 

O sistema conta com um pipeline RAG (*Retrieval-Augmented Generation*) para extrair metadados estruturados de artigos científicos em formato PDF e um validador de precisão que compara as extrações geradas pela LLM contra um gabarito oficial (*Ground Truth*).

---

## 🛠️ Requisitos e Configuração

### 1. Requisitos do Sistema
*   **Python 3.11** ou superior.
*   **Ollama** instalado localmente.
    *   Baixe e instale em: [https://ollama.com](https://ollama.com)
    *   Baixe o modelo de embeddings padrão:
        ```bash
        ollama pull embeddinggemma
        ```
    *   Baixe o modelo de chat padrão (ou outro de sua preferência configurado no `config.json`):
        ```bash
        ollama pull gemma3:27b
        ```

### 2. Configurando o Ambiente
Crie um ambiente virtual do Python e instale as dependências:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No Windows (PowerShell/CMD):
venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate

# Instalar dependências necessárias
pip install -r requirements.txt
```

---

## 🚀 Como Iniciar e Usar (Painel Web)

A forma mais prática e moderna de utilizar o projeto é através da nossa **Interface Web Unificada**.

### 1. Iniciar o Servidor
Com o seu ambiente virtual ativado, rode o comando na raiz do projeto:

```bash
python app.py
```
Isso iniciará um servidor web leve e local em: **[http://localhost:8000](http://localhost:8000)**.

### 2. Fluxo de Trabalho Prático no Painel

1.  **Acessar a Interface**: Abra o endereço [http://localhost:8000](http://localhost:8000) no seu navegador.
2.  **Ingestão de Documentos (Aba "Ingestão de PDFs")**:
    *   O painel lista todos os arquivos PDF salvos na pasta `papers/`.
    *   Configure os parâmetros de chunking desejados (Tamanho do Chunk e Overlap) no formulário inferior.
    *   Selecione os artigos desejados por checkbox e clique em **📥 Ingerir Selecionados**. O processo rodará em segundo plano e você poderá assistir a conversão e gravação de vetores no terminal interativo em tempo real.
3.  **Extração de Dados RAG (Aba "Pipeline RAG")**:
    *   No dropdown **Base Vetorial de Origem**, selecione o banco de dados que você acabou de gerar (ex: `db_600c_150o`). A tabela listará as coleções disponíveis nela.
    *   Selecione as coleções desejadas por checkbox e clique em **🤖 Extrair do Selecionados**. A LLM processará as perguntas estruturadas e você verá a resposta e o raciocínio sendo gerados no terminal.
4.  **Auditar a Precisão (Aba "Auditoria Geral")**:
    *   Clique no botão **🔄 Atualizar Validação** no topo direito. O validador lerá os logs e calculará os percentuais de acerto confrontando as respostas com o gabarito.
    *   O dashboard exibirá o gráfico médio de precisão por categoria (Área Computacional, Sensores, Tecnologias e Solução Principal).
    *   Na barra lateral esquerda, clique em qualquer artigo para ver uma **comparação lado a lado detalhada** entre o Gabarito (Esperado) e o Extraído pela IA, colorida de acordo com o status (Verde = Correto, Amarelo = Parcial, Vermelho = Incorreto).
5.  **Ajustes Rápidos (Aba "Configurações")**:
    *   Modifique os modelos usados, tokens e chaves de API do Ollama sem abrir arquivos.
    *   Edite interativamente o template do prompt (`prompt.txt`), as perguntas (`question.json`) e o gabarito de validação (`ground_truth.json`). O painel valida a estrutura do JSON e salva automaticamente nos diretórios certos.

### 💻 Como Usar via Terminal (CLI)
Se preferir rodar no terminal sem a interface web, os wrappers de linha de comando foram totalmente preservados:

*   **Ingestão**: `python scripts/ingest.py` (interativo no terminal)
*   **Extração RAG**: `python scripts/chat.py` (interativo no terminal)
*   **Validação**: `python scripts/validate.py` (executa a auditoria e salva o arquivo de métricas)

---

## 🏗️ Como Funciona (Arquitetura do Projeto)

O projeto foi refatorado sob princípios de engenharia de software limpa, dividindo-se em módulos altamente reutilizáveis e isolando o armazenamento de bancos e configurações.


### 🧠 O Motor RAG e Isolamento
*   **Modularização**: Toda a lógica pesada de processamento de dados e LLM foi movida para pacotes dentro de `scripts/core/`. O servidor web (`app.py`) e os scripts de terminal (`scripts/ingest.py`, etc.) importam e compartilham esse mesmo "motor", eliminando código duplicado.
*   **Particionamento de Bancos de Dados**: Cada conjunto de configurações cria uma pasta isolada dentro de `./database/` (nomeada como `db_{chunk}c_{overlap}o`). Isso significa que você pode fazer experimentos com diferentes tamanhos de blocos sem que um sobrescreva ou apague a base de dados do outro.
*   **Reconfiguração UTF-8**: O projeto conta com defesa interna de codificação de saída (`sys.stdout.reconfigure`), garantindo que palavras acentuadas em português rodem sem gerar falhas de caracteres (`UnicodeEncodeError`) em terminais do Windows.
