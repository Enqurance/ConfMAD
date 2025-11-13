import os
import math
import pickle
import joblib
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import torch
import torch.nn as nn

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.serialization.add_safe_globals([argparse.Namespace])

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
            if args['task'] != 'MATH':
                cur_turn = {
                    'reason': reason,
                    'answer': answer,
                    'correct': answer == ground_truth
                }
            else:
                cur_turn = {
                    'reason': reason,
                    'answer': answer,
                    'correct': CompareAnswerMATH(response, ground_truth)
                }
            if conf:
                cur_turn['confidence'] = ExtractConfidenceScore(response)

            debate_status['debate_agents_result'][his['agent_model']].append(cur_turn)

        res.append(debate_status)
    return res


def ExtractLogprobsPost(data, debaters, args):
    if args['calibration_scheme'] != 'temperature':
        return None
    
    res = []
    for debate in data:
        debate_status = {
            'question': debate['question'],
            'debate_agents_result': {d:[] for d in debaters},
        }
        
        for his in debate['debate_history']:
            debate_status['debate_agents_result'][his['agent_model']].append(his['top_logprobs'])

        res.append(debate_status)
    return res


def process_string(input_string):
	# 1. Replace all '\\\\' with '\\'
    processed_string = input_string.replace('\\\\', '\\')
    
    # 2. Replace all '\dfrac' with '\frac'
    processed_string = processed_string.replace('\\dfrac', '\\frac')

	# 3. Remove all whitespace
    processed_string = re.sub(r'\s+', '', processed_string)

    return processed_string


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
        elif task == 'BBH' or task == 'MMLU' or task == 'MMLUPRO':
            # print(response.lower().split('answer:')[1])
            answer = response.lower().split('answer:')[1].lstrip().split('\n')[0].strip()
            if answer.lower() == ground_truth.lower():
                return ground_truth
            bracket_pattern = r'\(.*\)'
            multiple_choice = re.match(bracket_pattern, ground_truth)
            if multiple_choice:
                choice, content = ground_truth.split(" ", 1)
                if answer.lower() == content.lower():
                    return ground_truth
                gt_choice = re.findall(r'\((.*?)\)', choice)
                an_choice = re.findall(r'\((.*?)\)', answer)

                if answer.lower() == gt_choice[0].lower():
                    return ground_truth
                if len(an_choice) > 0 and an_choice[0].lower() == gt_choice[0].lower():
                    return ground_truth
            return answer
        elif task == 'CMS':
            response = response.lower().split('answer:')[1]
            if ground_truth.lower() in response.lower():
                return ground_truth
    return None

from math_verify import parse, verify
from sympy import zoo, nan

def CompareAnswerMATH(res, gt):
    try:
        if 'answer:' not in res.lower():
            return False
        
        answer_line = res.lower().split('answer:')[1].lstrip().split('\n')[0].strip()
        
        if not answer_line:
            return False
        
        gold = parse(gt)
        answer = parse(answer_line)
        
        if answer is None or answer == zoo or answer == nan or gold is None or gold == zoo or gold == nan:
            return False

        res = verify(gold, answer)
        return res

    except ValueError as e:
        print(f"ValueError occurred: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# Extract ground truth
def ExtractGroundTruth(value, task):
    if task == 'GSM':
        return float(value.replace(',', ''))
    elif task == 'BIGGSM':
        return int(value)
    elif task == 'StrategyQA' or task == 'SQuAD' or task == 'MMLU' or task == 'SciQ' or task == 'BBH' or task == 'CMS' or task == 'MATH' or task == 'MMLUPRO':
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

base_dir = '/home/zijie/Experiments/Debate/checkpoints'

def SigmaBinning(scores, num_bins=5):
    # Using standard error to do binning
    max_score = max(scores)
    sigma = np.std(scores)

    binned_scores = {}
    edges = []
    for i in range(num_bins):
        binned_scores[i] = []
        edges.append(math.ceil(max_score - i * sigma))

    for s in scores:
        for idx, e in enumerate(edges):
            if s >= e - idx * sigma :
                binned_scores[idx].append(int(s))
                break
                
    return binned_scores, edges      


def IsotonicCalibration(confidence_list, model_path="isotonic_regression_model.pkl"):
    iso_reg = joblib.load(model_path)
    confidence_list = np.array(confidence_list) / 100.0
    calibrated_confidences = iso_reg.predict(confidence_list)

    return calibrated_confidences * 100


def TemperatureScaling(logprobs, temperature=1):
    top_logprobs = [[] for _ in logprobs]
    top_tokens = [[] for _ in logprobs]
    tokens = [l.token for l in logprobs]
    
    for idx, l in enumerate(logprobs):
        token_list = []
        for t in l.top_logprobs:
            token_list.append(t)
            top_logprobs[idx].append(t.logprob)
            top_tokens[idx].append(t.token)

    scaled_logprobs = []
    for idx, l in enumerate(top_logprobs):
        logits = np.array(l) / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        logprobs = np.log(probs)
        scaled_logprobs.append(logprobs)
    
    result = []
    for idx, t in enumerate(tokens):
        if t not in top_tokens[idx]:
            epsilon = 1e-9
            sum_prob = float(sum(np.exp(scaled_logprobs[idx])))
            result.append(math.log(max(epsilon, abs(1 - sum_prob))))
        else:
            target_index = top_tokens[idx].index(t)
            result.append(float(scaled_logprobs[idx][target_index]))

    return result


def load_model_with_scalar(model_path):
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    calibration_model = data["calibration_model"]
    scalar = data["scalar"]
    return calibration_model, scalar


def TemperatureScalingCalibration(target_tokens, model, task, conf):
    if len(target_tokens) == 0:
        return [50]
    scheme = 'temperature'
    path = os.path.join(base_dir, task)
    path = os.path.join(path, conf) 
    path = os.path.join(path, scheme)
    model_name = model
    model_path = os.path.join(path, model + '.plk')
    model, scalar = load_model_torch(TemperatureScaling, model_path)
    
    target_indices = []
    for i, t in enumerate(target_tokens):
        has_token = False
        token_str = t.token
        if i == len(target_tokens) - 1 and (token_str.isspace() or len(token_str) == 0):
            continue

        for idx, t_top in enumerate(t.top_logprobs):
            if t_top.token == token_str:
                has_token = True
                target_indices.append(idx)
                break

        if not has_token:
            target_indices.append(len(t.top_logprobs))
            
    target_indices = torch.tensor(target_indices, dtype=torch.int64)
    top_logprobs_list = torch.tensor([[top_token.logprob for top_token in token.top_logprobs] + [-10] for token in target_tokens], dtype=torch.float)
    seq_length = torch.tensor(len(target_tokens), dtype=torch.float)
    _, calibrated_conf = model(top_logprobs_list, target_indices, seq_length)
    calibrated_conf = calibrated_conf.item()
    return np.round(calibrated_conf * 100).astype(int).tolist()


def ProbabilityCalibration(scores, model, task, conf, scheme='platt'):
    calibrated_confidences = None
    
    path = os.path.join(base_dir, task)
    path = os.path.join(path, conf) 
    path = os.path.join(path, scheme)
    model_path = os.path.join(path, model + '.plk')

    if scheme == 'platt':
        platt_model, scalar = load_model_with_scalar(model_path)

        scaled_confidences = np.array(scores) / 100.0 * scalar
        scaled_confidences = scaled_confidences.reshape(-1, 1)

        calibrated_confidences = platt_model.predict_proba(scaled_confidences)[:, 1]
        
    elif scheme == 'histogram':
        calibration_model, scalar = load_model_with_scalar(model_path)
        bin_edges = calibration_model["bin_edges"]
        bin_true_ratios = calibration_model["bin_true_ratios"]

        confidence_list = np.array(scores)
        scaled_confidences = confidence_list / 100.0 * scalar
    
        bin_indices = np.digitize(scaled_confidences, bin_edges, right=True)
    
        calibrated_confidences = np.zeros_like(scaled_confidences)
        for i in range(1, len(bin_edges)):
            bin_mask = bin_indices == i
            calibrated_confidences[bin_mask] = bin_true_ratios[i - 1]

    elif scheme == 'isotonic':
        iso_reg, scalar = load_model_with_scalar(model_path)
        
        confidence_list = np.array(scores)
        scaled_confidences = confidence_list / 100.0 * scalar
        calibrated_confidences = iso_reg.predict(scaled_confidences)

    if calibrated_confidences is not None:
        return np.round(calibrated_confidences * 100).astype(int).tolist()

    raise ValueError(f"Unsupported calibration scheme: {scheme}")


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")


def apply_scalar(confidence_list, scalar=1.0):
    return np.clip(np.array(confidence_list) * scalar, 0, 1)


def save_model_torch(args, model, scalar, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    data_to_save = {
        "scalar": scalar,
        "args": args
    }
    
    torch.save({"model_state_dict": model.state_dict(), **data_to_save}, model_path)
    print(f"Model and scalar saved to: {model_path}")


def save_model_with_scalar(args, model, scalar, model_path):
    data_to_save = {
        "scalar": scalar,
        "model": model,
        "args": args
    }
    ensure_directory_exists(model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Torch model and scalar saved to: {model_path}")


def load_model_with_scalar(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data["model"], data["scalar"]


def load_model_torch(model_class, model_path):
    checkpoint = torch.load(model_path, weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    
    scalar = checkpoint["scalar"]
    args = checkpoint["args"]
    
    model = model_class()
    model.load_state_dict(model_state_dict)
    
    return model, scalar


def calculate_ece(confidence_list, label_list, n_bins=10):
    confidence_list = np.array(confidence_list)
    label_list = np.array(label_list)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_list, bin_edges, right=True)

    ece = 0
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(label_list[bin_mask])
            bin_confidence = np.mean(confidence_list[bin_mask])
            bin_weight = np.sum(bin_mask) / len(label_list)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
            # print(bin_accuracy, bin_confidence, bin_weight, bin_weight * abs(bin_accuracy - bin_confidence), ece)
    return ece * 100


import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, top_logprobs, selected_indices, seq_length):
        temperature = torch.exp(self.log_temperature)
        scaled_logprobs = top_logprobs / temperature  # [N, K]
        scaled_logprobs = scaled_logprobs - torch.max(scaled_logprobs, dim=-1, keepdim=True)[0]
        scaled_probs_all = torch.softmax(scaled_logprobs, dim=-1)  # [N, K]

        selected_probs = torch.gather(scaled_probs_all, dim=-1, index=selected_indices.unsqueeze(-1)).squeeze(-1)  # [N]
        log_selected_probs_sum = torch.sum(torch.log(selected_probs)) 
        confidence = torch.exp(log_selected_probs_sum / seq_length)

        return selected_probs, confidence
    

def binary_ece(conf_list, label_tensor, n_bins=10):
    confs = np.asarray(conf_list, dtype=np.float32)
    labels = label_tensor.cpu().numpy().astype(np.float32)
    assert len(confs) == len(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for b in range(n_bins):
        l, r = bin_edges[b], bin_edges[b + 1]
        mask = (confs > l) & (confs <= r) if b > 0 else (confs >= l) & (confs <= r)
        if not np.any(mask):
            continue
        bin_confs = confs[mask]
        bin_labels = labels[mask]
        bin_acc = bin_labels.mean()
        bin_conf = bin_confs.mean()
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece) * 100


def compute_conf_list(model, target_token_list, token_logprobs_list):
    confs = []
    for i, token_list in enumerate(target_token_list):
        if len(token_list) == 0:
            confs.append(0.5)
            continue

        token_indices = []
        for j, t in enumerate(token_list):
            token_str = t['token_str']
            has_token = False
            for idx, tt in enumerate(token_logprobs_list[i][j]):
                if tt['token_str'] == token_str:
                    token_indices.append(idx); has_token = True; break
            if not has_token:
                token_indices.append(len(token_logprobs_list[i][j]))
        token_indices = torch.tensor(token_indices, dtype=torch.int64)

        logprobs_list = []
        for j, top_logprobs in enumerate(token_logprobs_list[i]):
            top_token_probs = [t['logprob'] for t in top_logprobs]
            top_token_probs.append(-10)
            logprobs_list.append(top_token_probs)
        logprobs_list = torch.tensor(logprobs_list, dtype=torch.float)

        seq_length = torch.tensor(len(token_list), dtype=torch.float)

        with torch.no_grad():
            _, confidence = model(logprobs_list, token_indices, seq_length)
        confs.append(float(confidence.item()))
    return confs


def train_and_save_temperature_scaling(target_token_list, token_logprobs_list, labels, confs):
    labels = torch.tensor([0 if l == False else 1 for l in labels], dtype=torch.float)
    
    # pre_model = TemperatureScaling() 
    # pre_conf_list = compute_conf_list(pre_model, target_token_list, token_logprobs_list)
    # print(labels, pre_conf_list)
    # print(labels, confs)
    confs = [c / 100 for c in confs]
    pre_ece = calculate_ece(confs, labels, n_bins=10)

    model = TemperatureScaling()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 25

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, token_list in enumerate(target_token_list):
            if len(token_list) == 0:
                continue

            token_indices = []
            for j, t in enumerate(token_list):
                token_str = t['token_str']
                has_token = False
                for idx, tt in enumerate(token_logprobs_list[i][j]):
                    if tt['token_str'] == token_str:
                        token_indices.append(idx); has_token = True; break
                if not has_token:
                    token_indices.append(len(token_logprobs_list[i][j]))
            token_indices = torch.tensor(token_indices, dtype=torch.int64)

            logprobs_list = []
            for j, top_logprobs in enumerate(token_logprobs_list[i]):
                top_token_probs = [t['logprob'] for t in top_logprobs]
                top_token_probs.append(-10)
                logprobs_list.append(top_token_probs)
            logprobs_list = torch.tensor(logprobs_list, dtype=torch.float)

            seq_length = torch.tensor(len(token_list), dtype=torch.float)
            label = labels[i]

            optimizer.zero_grad()
            _, confidence = model(logprobs_list, token_indices, seq_length)
            loss = criterion(confidence, label)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Temperature: {torch.exp(model.log_temperature).item():.4f}")
    print("========== Finished Training ==========")

    calibrated_confidence_list = compute_conf_list(model, target_token_list, token_logprobs_list)
    post_ece = calculate_ece(calibrated_confidence_list, labels, n_bins=10)

    return pre_ece, post_ece, calibrated_confidence_list, model, None


def train_and_save_histogram_calibration(label_list, confidence_list, n_bins=10, scalar=1.0):
    label_list = np.array(label_list)
    
    
    for idx, conf in enumerate(confidence_list):
        if conf is None:
            confidence_list[idx] = 50.0
            
    confidence_list = np.array(confidence_list)

    confidence_list = apply_scalar(confidence_list / 100.0, scalar)

    if len(label_list) != len(confidence_list):
        raise ValueError("The count of labels and the count of confidence scores are not equal")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_list, bin_edges, right=True)

    bin_true_ratios = []
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_true_ratio = np.mean(label_list[bin_mask])
        else:
            bin_true_ratio = 0
        bin_true_ratios.append(bin_true_ratio)

    calibration_model = {
        "bin_edges": bin_edges,
        "bin_true_ratios": bin_true_ratios
    }
    
    ece_before = calculate_ece(confidence_list, label_list, n_bins=n_bins)

    calibrated_confidences = np.zeros_like(confidence_list)
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        calibrated_confidences[bin_mask] = bin_true_ratios[i - 1]
    ece_after = calculate_ece(calibrated_confidences, label_list, n_bins=n_bins)
    print(confidence_list[:5], calibrated_confidences[:5])
    return ece_before, ece_after, calibrated_confidences, calibration_model, scalar


def train_and_save_isotonic_regression(label_list, confidence_list, scalar=1.0):
    label_list = np.array(label_list)
    confidence_list = np.array(confidence_list)

    confidence_list = apply_scalar(confidence_list / 100.0, scalar)

    if len(label_list) != len(confidence_list):
        raise ValueError("The count of labels and the count of confidence scores are not equal")

    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(confidence_list, label_list)

    ece_before = calculate_ece(confidence_list, label_list)
    calibrated_confidence_list = iso_reg.predict(confidence_list)
    ece_after = calculate_ece(calibrated_confidence_list, label_list)

    return ece_before, ece_after, calibrated_confidence_list, iso_reg, scalar


def train_and_save_platt_scaling(label_list, confidence_list, scalar=1.0):
    label_list = np.array(label_list)
    confidence_list = np.array(confidence_list, dtype=object)

    valid_indices = [i for i, conf in enumerate(confidence_list) if conf is not None]

    label_list = label_list[valid_indices]
    confidence_list = confidence_list[valid_indices]

    confidence_list = apply_scalar(confidence_list.astype(float) / 100.0, scalar)

    if len(label_list) != len(confidence_list):
        raise ValueError("The count of labels and the count of confidence scores are not equal")

    platt_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    confidence_list = confidence_list.reshape(-1, 1)
    platt_model.fit(confidence_list, label_list)

    ece_before = calculate_ece(confidence_list.flatten(), label_list)
    calibrated_confidence_list = platt_model.predict_proba(confidence_list)[:, 1]
    ece_after = calculate_ece(calibrated_confidence_list, label_list)

    return ece_before * 100, ece_after * 100, calibrated_confidence_list, platt_model, scalar


"""
    We train calibration model for predicting the confidence score of the model here.
    The scalar and the calibration model are saved after training
"""
def TrainCalibrationModel(result, debaters, args, scalar=1.0):
    task, conf, scheme = args.task, args.conf_mode, args.calibration_scheme
    debate_agents = args.debate_agents
    train_turn = 0

    path = os.path.join(base_dir, task)
    path = os.path.join(path, conf) 
    path = os.path.join(path, scheme)
    models = [d for d in debaters.values()]
    debater_checkpoints_path = {m: os.path.join(path, m + '.plk') for m in models}
    result_list = ExtractResultList(result, models, args)

    res = {
        d: {'label': [], 'conf': []} for d in models
    }
    
    for r in result_list:
        for agent, turns in r['debate_agents_result'].items():
            res[agent]['label'].append(turns[train_turn]['correct'])
            res[agent]['conf'].append(turns[train_turn]['confidence'])
            
    if args.calibration_scheme == 'temperature':
        logprobs = ExtractLogprobs(result, models, args)
        for d in models:
            res[d]['top_logprobs'] = []
        for l in logprobs:
            for agent, turns in l['debate_agents_result'].items():
                res[agent]['top_logprobs'].append(turns[train_turn])
        
    calibrated_conf = {}
    for model, path in debater_checkpoints_path.items():
        labels, confs = res[model]['label'], res[model]['conf']
        calibrated_conf_list = None
        
        if scheme == 'platt':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_platt_scaling(labels, confs, scalar=scalar)
        elif scheme == 'histogram':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_histogram_calibration(labels, confs, scalar=scalar)
        elif scheme == 'isotonic':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_isotonic_regression(labels, confs, scalar=scalar)
        elif scheme == 'temperature':
            logprobs_list = res[model]['top_logprobs']
            
            target_token_list, top_logprobs_list = [], []
            for logprob in logprobs_list:
                target_token_list.append([])
                top_logprobs_list.append([])
                for token in logprob:
                    target_token_list[-1].append(token['target_token'])
                    top_logprobs_list[-1].append(token['top_logprobs'])
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_temperature_scaling(target_token_list, top_logprobs_list, labels, confs)
        else:
            raise ValueError(f"Unsupported calibration scheme: {scheme}")
        
        if args.calibration_overwrite or not os.path.exists(path):
            if args.calibration_scheme != 'temperature':
                save_model_with_scalar(args, calibration_model, scalar, path)
            else:
                save_model_torch(args, calibration_model, scalar, path)
        calibrated_conf[model] = calibrated_conf_list
        
    index = 0
    for r in result.values():
        for i in range(debate_agents * train_turn, debate_agents * (train_turn + 1)):
            r['debate_history'][i]['calibrated_confidence'] = round(calibrated_conf[r['debate_history'][i]['agent_model']][index] * 100)
        index += 1
    
    return result


def TrainCalibrationModelMannualy(file_path, scalar=1.0):
    with open(os.path.join(file_path), 'r') as file:
        data = json.load(file)
        args = data[0]
        result = data[1:1001]

    task, conf, scheme = args['task'], args['conf_mode'], args['calibration_scheme']
    debate_agents = args["debate_agents"]
    train_turn = 0

    path = os.path.join(base_dir, task)
    path = os.path.join(path, conf) 
    path = os.path.join(path, scheme)
    
    models = [d for d in debate_agents.values()]
    debater_checkpoints_path = {m: os.path.join(path, m + '.plk') for m in models}
    result_list = ExtractResultListPost(result, models, args)

    res = {
        d: {'label': [], 'conf': []} for d in models
    }
    
    for r in result_list:
        for agent, turns in r['debate_agents_result'].items():
            res[agent]['label'].append(turns[train_turn]['correct'])
            res[agent]['conf'].append(turns[train_turn]['confidence'])
            
    if args['calibration_scheme'] == 'temperature':
        logprobs = ExtractLogprobsPost(result, models, args)
        for d in models:
            res[d]['logprobs list'] = []
        for l in logprobs:
            for agent, turns in l['debate_agents_result'].items():
                res[agent]['logprobs list'].append(turns[train_turn])
        
    
    calibrated_conf = {}
    for model, path in debater_checkpoints_path.items():
        labels, confs = res[model]['label'], res[model]['conf']
        calibrated_conf_list = None
        
        if scheme == 'platt':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_platt_scaling(labels, confs, scalar=scalar)
            # print(ece_before, ece_after)
        elif scheme == 'histogram':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_histogram_calibration(labels, confs, scalar=scalar)
        elif scheme == 'isotonic':
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_isotonic_regression(labels, confs, scalar=scalar)
        elif scheme == 'temperature':
            logprobs_list = res[model]['logprobs list']
            
            target_token_list, top_logprobs_list = [], []
            for logprob in logprobs_list:
                target_token_list.append([])
                top_logprobs_list.append([])
                for token in logprob:
                    target_token_list[-1].append(token['target_token'])
                    top_logprobs_list[-1].append(token['top_logprobs'])
            ece_before, ece_after, calibrated_conf_list, calibration_model, scalar = train_and_save_temperature_scaling(target_token_list, top_logprobs_list, labels, confs)
        else:
            raise ValueError(f"Unsupported calibration scheme: {scheme}")
        print(f"{ece_before:.1f}", f"{ece_after:.1f}")
        # if args['calibration_overwrite'] or not os.path.exists(path):
        #     if args['calibration_scheme'] != 'temperature':
        #         save_model_with_scalar(args, calibration_model, scalar, path)
        #     else:
        #         save_model_torch(args, calibration_model, scalar, path)
        calibrated_conf[model] = calibrated_conf_list
        
     
# Run this file to train the calibration model mannually
if __name__ == "__main__":
    train_file_path = "/home/zijie/Experiments/Debate/result/MATH/2025_05_05/14_58_11.json"
    TrainCalibrationModelMannualy(train_file_path)


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

            if args.task != 'MATH':
                cur_turn = {
                    'reason': reason,
                    'answer': answer,
                    'correct': answer == ground_truth
                }
            else:
                cur_turn = {
                    'reason': reason,
                    'answer': answer,
                    'correct': CompareAnswerMATH(response, ground_truth)
                }
                
            if conf:
                cur_turn['confidence'] = ExtractConfidenceScore(response)
            debate_status['debate_agents_result'][his['agent_model']].append(cur_turn)

        res.append(debate_status)
    return res


def ExtractLogprobs(data, debaters, args):
    if args.calibration_scheme != 'temperature':
        return None
    
    res = []
    for debate in data.values():
        debate_status = {
            'question': debate['question'],
            'debate_agents_result': {d:[] for d in debaters},
        }
        
        for his in debate['debate_history']:
            debate_status['debate_agents_result'][his['agent_model']].append(his['top_logprobs'])

        res.append(debate_status)
    return res