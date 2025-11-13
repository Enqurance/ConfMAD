import json
import re

def ExtractOption(file_path):
    pattern = r'\(.*\)'
    res = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            m = re.match(pattern, data['answer'])
            if m:
                option = data['question'].split(data['answer'])[1].split('\n')[0].strip()
                # print(data['question'])
                data['answer'] = data['answer'] + " " + option
            res.append(data)
    
    with open(file_path, 'w') as file:
        for item in res:
            file.write(json.dumps(item) + '\n')        
    
            
fp = "./bbh_test.jsonl"
fp = "./bbh_valid.jsonl"
ExtractOption(fp)