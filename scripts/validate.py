import json
import os
import re
import difflib
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_words(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return set(w for w in text.split() if len(w) > 1) # ignore 1-char words like "e"

def match_score(str1, str2):
    w1 = get_words(str1)
    w2 = get_words(str2)
    
    if not w1 or not w2:
        return 0.0
        
    if w1.issubset(w2):
        return 1.0
        
    c1 = " ".join(sorted(w1))
    c2 = " ".join(sorted(w2))
    return difflib.SequenceMatcher(None, c1, c2).ratio()

def match_key(gt_key, llm_key):
    w1 = get_words(gt_key)
    w2 = get_words(llm_key)
    # Ignora 'ou', 'e'
    w1 = {w for w in w1 if w not in ['ou', 'e']}
    w2 = {w for w in w2 if w not in ['ou', 'e']}
    return len(w1.intersection(w2)) > 0

def flatten_and_split(raw_actual):
    actual_list = []
    raw_list = raw_actual if isinstance(raw_actual, list) else [str(raw_actual)]
    for item in raw_list:
        # Tenta separar por vírgulas ou o sinal de + caso a LLM tenha retornado tudo junto
        parts = re.split(r'[,+]', str(item))
        actual_list.extend([p.strip() for p in parts if p.strip()])
    return actual_list

def evaluate_field(expected, actual):
    if expected is None:
        return {"score": 0, "status": "Ignorado"}
        
    # Quando o gabarito é uma única string (ex: Área Computacional)
    if isinstance(expected, str):
        actual_list = flatten_and_split(actual)
            
        best_score = 0
        for act in actual_list:
            s = match_score(expected, act)
            if s > best_score:
                best_score = s
                
        # threshold de 0.8 para considerar full match
        if best_score >= 0.8:
            score_pct = 100
        else:
            score_pct = int(best_score * 100)
            
        status = "Correto" if score_pct == 100 else ("Incorreto" if score_pct == 0 else "Parcial")
        return {"score": score_pct, "status": status}
        
    # Quando o gabarito é um array de valores (ex: Sensores, Bibliotecas)
    if isinstance(expected, list):
        actual_list = flatten_and_split(actual)
            
        matched_expected = set()
        matched_actual = set()
        
        for i, exp in enumerate(expected):
            best_s = 0
            best_j = -1
            for j, act in enumerate(actual_list):
                if j in matched_actual:
                    continue
                s = match_score(exp, act)
                if s > best_s:
                    best_s = s
                    best_j = j
            
            if best_s >= 0.8 and best_j != -1:
                matched_expected.add(i)
                matched_actual.add(best_j)
                
        correct = len(matched_expected)
        extra = len(actual_list) - len(matched_actual)
        total_expected = len(expected)
        
        if total_expected == 0:
            if len(actual_list) > 0 and actual_list[0].lower() not in ["não informado", "nao informado no contexto"]:
                return {"score": 0, "status": "Incorreto"}
            return {"score": 100, "status": "Correto"}
            
        # A pontuação é Acertos divididos pelo total de itens esperados MAIS os itens inventados
        raw_score = correct / (total_expected + extra)
        score_pct = max(0, min(100, int(raw_score * 100)))
        
        if score_pct == 100:
            status = "Correto"
        elif score_pct == 0:
            status = "Incorreto"
        else:
            status = "Parcial"
            
        return {"score": score_pct, "status": status}

def main():
    base_dir = Path(__file__).parent.parent
    
    env_logs_dir = os.environ.get("VALIDATE_LOGS_DIR")
    logs_dir = Path(env_logs_dir) if env_logs_dir else base_dir / "logs"
    
    env_gt_path = os.environ.get("VALIDATE_GT_PATH")
    gt_path = Path(env_gt_path) if env_gt_path else base_dir / "ground_truth.json"
    
    env_out_path = os.environ.get("VALIDATE_OUT_PATH")
    out_path = Path(env_out_path) if env_out_path else logs_dir / "validation_results.json"
    
    if not gt_path.exists():
        print(f"Erro: ground_truth.json nao encontrado em {gt_path}.")
        return
        
    ground_truth = load_json(gt_path)
    results = []
    
    for log_file in logs_dir.glob("*.json"):
        if log_file.name == "validation_results.json":
            continue
            
        log_data = load_json(log_file)
        
        original_name = log_data.get("original_filename", "")
        if not original_name:
            original_name = log_data.get("collection", "")
            
        match = re.search(r'\((\d+)\)', original_name)
        if not match:
            match = re.search(r'^(\d+)_', original_name)
            
        if not match:
            continue
            
        article_id = match.group(1)
        if article_id not in ground_truth:
            continue
            
        gt_data = ground_truth[article_id]
        
        evaluation = {}
        total_score = 0
        field_count = 0
        
        gt_keys = list(gt_data.keys())
        
        for idx, interaction in enumerate(log_data.get("interactions", [])):
            if idx >= len(gt_keys):
                break
                
            gt_key = gt_keys[idx]
            expected_val = gt_data[gt_key]
            actual_val = None
            
            resp_str = interaction.get("response", "")
            resp_clean = re.sub(r'```json\n?|\n?```', '', resp_str).strip()
            reasoning_val = None
            try:
                parsed = json.loads(resp_clean)
                for k, v in parsed.items():
                    if k.lower() == "raciocinio":
                        reasoning_val = v
                    else:
                        if isinstance(v, bool) and v is True:
                            actual_val = k
                        else:
                            actual_val = v
            except Exception:
                pass
                
            eval_result = evaluate_field(expected_val, actual_val)
            evaluation[gt_key] = {
                "expected": expected_val,
                "actual": actual_val if actual_val is not None else "Não encontrado",
                "reasoning": reasoning_val,
                "raw_response": resp_str,
                "score": eval_result["score"],
                "status": eval_result["status"]
            }
            total_score += eval_result["score"]
            field_count += 1
            
        overall_score = int(total_score / field_count) if field_count > 0 else 0
            
        results.append({
            "article_id": article_id,
            "original_filename": original_name,
            "log_file": log_file.name,
            "timestamp": log_data.get("timestamp", ""),
            "overall_score": overall_score,
            "evaluation": evaluation
        })
        
    results.sort(key=lambda x: int(x["article_id"]))
        
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Validação aproximada concluída para {len(results)} logs. Salvo em {out_path}")

if __name__ == "__main__":
    main()
