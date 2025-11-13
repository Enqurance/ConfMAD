import re
        
        
def GetPrompts(task, prompt, debater, item, history=[]):
    if task == 'GSM' or task == 'StrategyQA' or task == 'MMLU' or task == 'SciQ' or task == 'BBH' or task == 'CMS' or task == "MMLUPRO":
        system_prompt = prompt['system'].format(
            debater=debater.agent_name
        )
        
        if len(history) == 0:
            user_prompt = prompt['user'].format(
                question=item['question']
            )
        else:
            history_str = '\n'.join([f'{item['agent_name']}:\n{item['response']}' for item in history])
            user_prompt = prompt['user'].format(
                question=item['question'],
                debate_history=history_str
            )
    
        return system_prompt, user_prompt
    elif task == 'SQuAD':
        system_prompt = prompt['system'].format(
            debater=debater.agent_name
        )
        
        if len(history) == 0:
            user_prompt = prompt['user'].format(
                context=item['context'],
                question=item['question']
            )
        else:
            history_str = '\n'.join([f'{item['agent_name']}:\n{item['response']}' for item in history])
            user_prompt = prompt['user'].format(
                context=item['context'],
                question=item['question'],
                debate_history=history_str
            )
            
        return system_prompt, user_prompt
    
    ValueError('Dataset not supported')
    return None, None


def GetPromptsDebate(task, prompt, debater, item, history=[]):
    if task == 'GSM' or task == 'StrategyQA' or task == 'BBH' or task == 'MMLU' or task == 'SciQ' or task == 'CMS' or task == 'BIGGSM' or task == 'MATH' or task == 'GPQA' or task == 'MMLUPRO':
        system_prompt = prompt['system'].format(
            debater=debater.agent_name
        )
        if len(history) == 0:
            user_prompt = prompt['user'].format(
                question=item['question']
            )
        else:
            history_str = '\n'.join([f'{item['agent_name']}:\n{item['response']}' for item in history])
            user_prompt = prompt['user'].format(
                question=item['question'],
                debate_history=history_str
            )
        
        return system_prompt, user_prompt
        
    
    ValueError('Dataset not supported')
    return None


def FindAnswerIndices(text):
    pattern = r'(?i)(.*\banswer:\b.*)'
    match = re.search(pattern, text)
    
    if match:
        answer_line = match.group(1)
        before_answer = text[:match.start()]
        text = before_answer + answer_line

    # pattern = r'(?i)answer.*$'
    pattern = r'(?i)\banswer:.*'
    match = re.search(pattern, text)
    
    if match:
        start_index = match.start()
        end_index = match.end()
        # print(text[start_index:end_index])
        return start_index, end_index
    
    pattern = r'(?i)\banswer.*'
    match = re.search(pattern, text)
    if match:
        start_index = match.start()
        end_index = match.end()
        # print(text[start_index:end_index])
        return start_index, end_index
    
    pattern = r'(?i)\bboxed.*'
    match = re.search(pattern, text)
    if match:
        start_index = match.start()
        end_index = match.end()
        # print(text[start_index:end_index])
        return start_index, end_index
    else:
        print(text)
        print("===== NO LOGPROBS FOUND, RETESTING =====")
        return None, None


def GetTargetLogprobs(debate_response_ans, logprobs):
    target_tokens = []
    collect_tokens = False
    ans_start_index, ans_end_index = FindAnswerIndices(debate_response_ans)
    if ans_start_index is None or ans_end_index is None:
        return target_tokens

    current_index = 0
    for token in logprobs:
        text = token.token
        token_start_index = debate_response_ans.find(text, current_index)
        if token_start_index == -1:
            continue

        token_end_index = token_start_index + len(text)
        if (token_start_index <= ans_start_index and token_end_index > ans_start_index) or \
           (token_start_index < ans_end_index and token_end_index >= ans_end_index):
            collect_tokens = True
        if collect_tokens:
            target_tokens.append(token)
        if token_start_index > ans_end_index:
            break

        current_index = token_end_index
        
    return target_tokens
