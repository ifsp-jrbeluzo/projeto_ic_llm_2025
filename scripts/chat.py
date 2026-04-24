import os
import ollama
from ollama import Client
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb

#log colors
text_reset      = "\033[0m"
text_black      = "\033[30m"
text_red        = "\033[31m"
text_green      = "\033[32m"
text_yellow     = "\033[33m"
text_blue       = "\033[34m"
text_magenta    = "\033[35m"
text_cyan       = "\033[36m"
text_white      = "\033[37m"

#init local database
chroma_client = chromadb.PersistentClient(path="./database")

#get collection
collection = chroma_client.get_collection("papers")

def get_embedding(prompt, n_results=3):
  query_embedding = ollama.embeddings(
        model='embeddinggemma',
        prompt=prompt
    )['embedding']

  results = collection.query(
      query_embeddings=[query_embedding],
      n_results=n_results
  )

  context_chunks = []

  for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
      source = meta["source"]
      chunk = meta["chunk"]

      context_chunks.append(
          f"[Source: {source} | Chunk {chunk}]\n{doc}"
      )

  context = "\n\n".join(context_chunks)

  return context
   
#init ollama client
ollama_client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + '3abac4c624ac4438ae50f4e78f5287cd.7CGPF_O-lLdkyXHW-RgFEAYt'}
)

while True:
  prompt = input(f"\n{text_yellow}Prompt:{text_reset} ")

  if prompt == "/end":
      break
  
  context = get_embedding(prompt)

  final_prompt = f"""
You are a scientist working on a Systematic review article
specialized in answering questions using scientific papers and synthesizing the infortation within it.

You'll receive chunks from a model that will do the embedding from several papers

Instructions:
- Use ONLY the information provided in the Papers, try to do synthezise and interpret not only spit raw data.
- If the answer is not contained within the papers, say you don't know.
- Do not invent information, always use the information provided to give your answer.
- Answer the question in portuguese even if the context is in english.

<Papers chunks>
{context}
</Papers chunks>

<Question>
{prompt}
</Question>
  """

  print(f"\n{text_magenta}Final Prompt:{text_reset} {final_prompt}")

  messages = [
    {
      'role': 'user',
      'content': final_prompt,
    },
  ]
  
  msg = ""
  for part in ollama_client.chat('gemma3:27b-cloud', messages=messages, stream=True):
    msg = msg + f"{part['message']['content']}"

  print(f"\n{text_green}Response:{text_reset} {msg}")