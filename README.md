# Requirements

* Python 3.11
* Ollama installed

Download Ollama:
https://ollama.com

After installing Ollama, download the required models:

```
ollama pull embeddinggemma
```

---

# Project Setup

## 1. Clone the repository

```
git clone <repository_url>
cd <repository_folder>
```

---

## 2. Create a virtual environment

Create a Python virtual environment inside the project:

```
python -m venv venv
```

---

## 3. Activate the virtual environment

### Linux / Mac

```
source venv/bin/activate
```

### Windows

```
venv\Scripts\activate
```

After activation, your terminal should show something like:

```
(venv)
```

---

## 4. Install Python dependencies

Install all required Python libraries using the requirements file:

```
pip install -r requirements.txt
```

---

# Running the Project

Once the environment is activated and dependencies are installed, you can run the scripts normally:

```
python scripts/ingest.py
```

or

```
python scripts/chat.py
```

Make sure the virtual environment is activated before running the scripts.
