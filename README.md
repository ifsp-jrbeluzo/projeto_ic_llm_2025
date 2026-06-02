# Projeto IC LLM 2025 - Agricultural Robotics RAG Pipeline

Este projeto implementa uma pipeline RAG (Retrieval-Augmented Generation) para análise, extração e validação de informações em artigos científicos na área de robótica agrícola. O sistema inclui ingestão vetorial de PDFs, extração semântica com grandes modelos de linguagem (LLMs) via Ollama, e um painel visual unificado (Web UI) para facilitar a execução de testes, auditoria dos resultados e gerenciamento dos dados.

## Requisitos

* Python 3.11+
* [Ollama](https://ollama.com) instalado

Após instalar o Ollama, baixe os modelos base necessários para os embeddings:

```bash
ollama pull embeddinggemma
```

---

## 1. Configuração do Ambiente

Clone o repositório e acesse a pasta do projeto:

```bash
git clone <repository_url>
cd <repository_folder>
```

Crie um ambiente virtual e ative-o:

**Linux / Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## 2. Configurações Iniciais (`config.json`)

O arquivo principal de configuração é o `config.json`. Nele, você pode definir as pastas de armazenamento (onde seus PDFs devem ser colocados, na pasta `papers/`) e as configurações de conexão da API do Ollama.

Caso esteja utilizando um provedor na nuvem (como um proxy Cloud do Ollama), o painel reconhecerá automaticamente modelos grandes (ex: `deepseek-v3.1:671b-cloud`, `gemma4:31b-cloud`) filtrando-os na interface.

---

## 3. Executando o Sistema (Web UI)

A principal maneira de utilizar a ferramenta agora é através do painel visual web, que une todas as ferramentas num único local.

Inicie o servidor local FastAPI:

```bash
python app.py
```

Abra o seu navegador e acesse: [http://localhost:8000](http://localhost:8000)

### Funcionalidades do Painel Web:

- **📊 Status & Auditoria:** Visualize os testes rodados (runs), compare as respostas dadas pela IA com as expectativas reais (`ground_truth.json`), e audite o **raciocínio interno** que o LLM tomou para chegar àquela conclusão. Você também pode apagar testes antigos por aqui.
- **🗄️ Gerar Banco de Dados:** Faça a ingestão inteligente dos PDFs localizados na pasta `papers/`. O sistema particiona os textos e gera os Embeddings usando ChromaDB e SQLite (seguro contra travamento de exclusões). Você pode definir os tamanhos de particionamento dinamicamente.
- **⚙️ Executar Pipeline RAG:** Configure a execução RAG sem precisar editar código cru:
  - Selecione entre diversos bancos de dados gerados e misture-os;
  - Escolha um dos modelos LLM disponíveis de forma dinâmica;
  - Defina suas questões de busca utilizando um **Construtor Visual Dinâmico de Perguntas** – adicione, edite e classifique perguntas sem precisar se preocupar com JSON formatado.
  - O log de execução aparece fixado em um terminal integrado na lateral, permitindo que você navegue em outras abas sem perder o status de execução.

---

## Scripts CLI Auxiliares

Caso queira rodar operações diretamente pelo terminal (sem a UI), você pode rodar:

- `python scripts/ingest.py`: Processa PDFs e gera bancos vetoriais.
- `python scripts/chat.py`: Interface de linha de comando (CLI) simples para fazer perguntas avulsas.
- `python scripts/validate.py`: Script para avaliar a eficácia dos testes do pipeline RAG.
