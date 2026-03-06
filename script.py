import requests

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

    r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    )

    print(f"\n{text_green}Response:{text_reset} {r.json()['response']}")