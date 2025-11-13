import json


class DataLoader():
    def __init__(self, data_path):
        pass
    
    
def load_jsonl_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                json_obj = json.loads(line)
                data.append(json_obj)
    return data