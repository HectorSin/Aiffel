import json

def open_json(file_path:str) -> dict[str, int]:
    with open(file_path, 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)
    return word_to_index