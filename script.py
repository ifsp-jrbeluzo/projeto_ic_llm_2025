import os
from ollama import Client

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

while True:
    prompt = input(f"\n{text_yellow}Prompt:{text_reset} ")

    if prompt == "/end":
        break

    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + '3abac4c624ac4438ae50f4e78f5287cd.7CGPF_O-lLdkyXHW-RgFEAYt'}
    )

    messages = [
      {
        'role': 'user',
        'content': prompt,
      },
    ]
    msg = ""
    for part in client.chat('gemma3:27b', messages=messages, stream=True):
      msg = msg + f"{part['message']['content']}"

    print(f"\n{text_green}Response:{text_reset} {msg}")