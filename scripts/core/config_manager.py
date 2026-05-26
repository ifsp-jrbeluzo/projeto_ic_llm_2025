import json
from pathlib import Path

def load_config(config_path="configs/config.json"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config_path="configs/config.json", data=None):
    if data is None:
        return
    path = Path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_prompt(prompt_path="configs/prompt.txt"):
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_prompt(prompt_path="configs/prompt.txt", text=""):
    path = Path(prompt_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_questions(questions_path="configs/question.json"):
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_questions(questions_path="configs/question.json", data=None):
    if data is None:
        return
    path = Path(questions_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_ground_truth(gt_path="configs/ground_truth.json"):
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_ground_truth(gt_path="configs/ground_truth.json", data=None):
    if data is None:
        return
    path = Path(gt_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
