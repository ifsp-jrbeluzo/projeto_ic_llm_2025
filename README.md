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

---

## Como a Pipeline Funciona (passo a passo)

A pipeline tem três etapas independentes, cada uma disparada por um script em
`scripts/` (via CLI ou pelo painel web, que roda os mesmos scripts como
subprocesso através do `TaskRunner` em `server/runner.py`).

### 1. Ingestão (`scripts/ingest.py`)

Transforma os PDFs de `papers/` em um banco vetorial pesquisável.

1. **Extração de texto** (`scripts/core/pdf_extractor.py`): cada página do PDF
   é lida com `pypdf`.
2. **Remoção da bibliografia:** o texto extraído é enviado para a LLM de chat
   configurada em `config.json` (`models.chat`), que analisa o final do
   documento e aponta em que linha começa a lista de referências - a LLM
   recebe só a "cauda" do texto (últimos ~30.000 caracteres, onde a
   bibliografia sempre está) numerada linha a linha, e responde com o índice
   da linha onde a seção começa, ou `NONE` se não achar nenhuma. Isso troca
   um regex que procurava o título da seção ("References", "Bibliography"...)
   por uma decisão semântica da própria LLM, porque PDFs frequentemente
   quebram esse título em várias linhas ou fontes diferentes, o que fazia o
   regex passar reto e deixar a bibliografia inteira dentro do texto
   analisado depois. Se a chamada à LLM falhar por qualquer motivo (rede,
   timeout), a ingestão não é interrompida: o texto completo é mantido sem
   corte, e um aviso é impresso no console.
3. **Chunking** (`scripts/core/chunker.py`): o texto (já sem bibliografia) é
   dividido em pedaços de `chunk_size` palavras, com sobreposição de
   `chunk_overlap` palavras entre pedaços vizinhos, para não perder contexto
   nas bordas.
4. **Embeddings:** cada chunk vira um vetor usando o modelo de embedding
   configurado (`models.embedding`, padrão `embeddinggemma`). Embeddings
   **sempre** rodam num Ollama local (`http://localhost:11434`) - o endpoint
   cloud do Ollama (`https://ollama.com`) não serve `/api/embeddings` nem
   hospeda `embeddinggemma`, só modelos de chat.
5. **Armazenamento:** os vetores, os textos e os metadados (`source`, número
   do chunk) são gravados no ChromaDB, numa pasta isolada por configuração de
   particionamento (`database/db_{chunk_size}c_{chunk_overlap}o/`), com uma
   coleção por PDF.

### 2. Execução RAG / Perguntas (`scripts/chat.py`)

Para cada banco de dados selecionado e cada pergunta definida em
`question.json`:

1. A pergunta vira um embedding (mesmo modelo local usado na ingestão).
2. O ChromaDB retorna os `n_results` chunks mais similares à pergunta.
3. Os chunks recuperados são agrupados por documento de origem e formatados
   como contexto (limitado a `rag.max_context_chars` caracteres).
4. O template em `prompt.txt` é preenchido com a pergunta e o contexto e
   enviado para o modelo de chat configurado (local ou cloud, com retry
   automático em caso de falha de API).
5. A LLM responde em JSON estruturado, sempre incluindo um campo
   `"raciocinio"` explicando a escolha, além do valor extraído.
6. Cada interação (pergunta, chunks usados, prompt final, resposta) é
   registrada num log JSON por coleção/execução, em `logs/` (ou
   `logs/runs/<run>/` quando disparado pela UI).

### 3. Validação (`scripts/validate.py`)

Compara os logs gerados na etapa anterior com o gabarito em
`ground_truth.json`:

1. Identifica o artigo de cada log pelo número entre parênteses no nome do
   arquivo original (ex: `(23) Automatic detection...`).
2. Faz o parse da resposta JSON de cada interação e compara o valor extraído
   com o valor esperado no gabarito, usando comparação por conjunto de
   palavras (com um limiar de similaridade) para tolerar pequenas variações
   de escrita.
3. Calcula um score por campo (Correto/Parcial/Incorreto) e uma média geral
   por artigo, salvando tudo em `validation_results.json` dentro da mesma
   pasta de logs.

### 4. Painel Web (`app.py`, `server/`)

Um FastAPI (`app.py` + `server/routes.py`) expõe essas três etapas como
tarefas assíncronas disparadas via `TaskRunner` (`server/runner.py`), que
roda os scripts acima como subprocessos e transmite o log de execução em
tempo real para a interface (`validator.html`), junto com telas de auditoria
dos resultados e gerenciamento dos bancos/execuções salvas.
