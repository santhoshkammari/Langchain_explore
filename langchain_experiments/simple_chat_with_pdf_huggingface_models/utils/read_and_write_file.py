import json


def save_cache(data,save_path: str = None):
    if save_path.endswith(".json"):
        with open(save_path) as f:
            json.dump(data,indent=2)
    elif save_path.endswith(".txt"):
        with open(save_path) as f:
            f.write(f'{data}')

def load_data(save_path: str = None):
    if save_path.endswith(".json"):
        with open(save_path) as f:
            data = json.load(save_path)
    elif save_path.endswith(".txt"):
        with open(save_path) as f:
            data = f.read()
    return data