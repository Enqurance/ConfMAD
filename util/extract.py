import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Extract result and return the list
def ExtractResultList(data, debaters, args, conf=True):
    res = []
    for debate in data.values():
        ground_truth = ExtractGroundTruth(debate['ground_truth'], args.task)
        
        debate_status = {
            'question': debate['question'],
            'ground_truth': ground_truth,
            'debate_agents_result': {d:[] for d in debaters},
        }
        
        for his in debate['debate_history']:
            response = his['response']
            reason = ExtractReason(response)
            answer = ExtractAnswer(response, args.task, ground_truth)
            cur_turn = {
                'reason': reason,
                'answer': answer,
                'correct': answer == ground_truth,
            }
            if conf:
                cur_turn['confidence'] = ExtractConfidenceScore(response)
            debate_status['debate_agents_result'][his['agent_model']].append(cur_turn)

        res.append(debate_status)
    return res


# Extract result and return the list
def ExtractResultListPost(data, debaters, args, conf=True):
    res = []
    for debate in data:
        ground_truth = ExtractGroundTruth(debate['ground_truth'], args['task'])
        
        debate_status = {
            'question': debate['question'],
            'ground_truth': ground_truth,
            'debate_agents_result': {d:[] for d in debaters},
        }
        
        for his in debate['debate_history']:
            response = his['response']
            reason = ExtractReason(response)
            answer = ExtractAnswer(response, args['task'], ground_truth)
            cur_turn = {
                'reason': reason,
                'answer': answer,
                'correct': answer == ground_truth,
            }
            if conf:
                cur_turn['confidence'] = ExtractConfidenceScore(response)
            print(his)
            debate_status['debate_agents_result'][his['agent_model']].append(cur_turn)

        res.append(debate_status)
    return res


def ExtractAnswer(response, task, ground_truth):
    pattern = r'answer:\s*(.*?)(?:\n|$)'
    match = re.findall(pattern, response, re.IGNORECASE)
    if match:
        answer_line = match[-1].strip()
        if task == 'GSM':
            ans_pattern = r'-?\d+(?:\.\d+)?'
            answer_line = answer_line.replace(',', '')
            all_numbers = re.findall(ans_pattern, answer_line)
            if all_numbers:
                return float(all_numbers[-1])
        elif task == 'BIGGSM':
            ans_pattern = r'-?\d+(?:\.\d+)?'
            answer_line = answer_line.replace(',', '')
            all_numbers = re.findall(ans_pattern, answer_line)
            if all_numbers:
                return float(all_numbers[-1])
        elif task == 'StrategyQA':
            ans_pattern = r'\b(true|false)\b'
            bool_val = re.search(ans_pattern, answer_line, re.IGNORECASE)
            if bool_val:
                return bool_val.group(0).lower() == 'true'
        elif task == 'SQuAD':
            response = response.lower().split('answer:')[1]
            if ground_truth.lower() in response.lower():
                return ground_truth
            return response
        elif task == 'MMLU':
            match = re.search(r'\((.*?)\)', ground_truth)
            if match:
                gt_choice = 'answer: ' + match.group(1)
                if gt_choice.lower() in response.lower():
                    return ground_truth
            response = response.lower().split('answer:')[1]
            if ground_truth.lower() in response.lower():
                return ground_truth
            return response
        elif task == 'SciQ':
            match = re.search(r'\((.*?)\)', ground_truth)
            if match:
                gt_choice = 'answer: ' + match.group(1)
                if gt_choice.lower() in response.lower():
                    return ground_truth
            response = response.lower().split('answer:')[1]
            if ground_truth.lower() in response.lower():
                return ground_truth
            return response
        elif task == 'BBH':
            answer = response.lower().split('answer:')[1].split('\n')[0].strip()

            if answer.lower() == ground_truth.lower():
                return ground_truth
            bracket_pattern = r'\(.*\)'
            multiple_choice = re.match(bracket_pattern, ground_truth)
            if multiple_choice:
                choice, content = ground_truth.split(" ", 1)
                if answer.lower() == content.lower():
                    return ground_truth
                elif len(answer) == 1:
                    if answer in choice.lower():
                        return ground_truth
            return response
        elif task == 'CMS':
            response = response.lower().split('answer:')[1]
            if ground_truth.lower() in response.lower():
                return ground_truth
    return None


# Extract ground truth
def ExtractGroundTruth(value, task):
    if task == 'GSM':
        return float(value.replace(',', ''))
    elif task == 'BIGGSM':
        return int(value)
    elif task == 'StrategyQA' or task == 'SQuAD' or task == 'MMLU' or task == 'SciQ' or task == 'BBH' or task == 'CMS':
        return value
    return None


# Extract reason from the debate history
def ExtractReason(response):
    pattern = r'reason:\s*(.*?)(?:\n|$)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# Extract the confidence from the debate history
def ExtractConfidenceScore(response):
    pattern = r'confidence score:\s*(.*?)(?:\n|$)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        conf_line = match.group(1).strip()
        conf_pattern = r'-?\d+(?:\.\d+)?'
        if re.search(conf_pattern, conf_line):
            conf = re.search(conf_pattern, conf_line)
            return float(conf.group())
    pattern = r'confidence:\s*(.*?)(?:\n|$)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        conf_line = match.group(1).strip()
        conf_pattern = r'-?\d+(?:\.\d+)?'
        if re.search(conf_pattern, conf_line):
            conf = re.search(conf_pattern, conf_line)
            return float(conf.group())
    return None