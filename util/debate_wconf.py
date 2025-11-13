from .data_loader import load_jsonl_data
from .language_assistants import LoadModel
from .config.model_info import model_info
from .prompt import (
    debate_prompt_ground_truth_answer_and_reason,
    debate_prompt_multi_persona,
    debate_prompt_chat_eval,
    misconception_identify,
    misconception_fix
)
from .data_processing import (
    GetPromptsDebate,
    GetTargetLogprobs,
)
from .debate_agents import Debator
from .config.debater import debaters
from .config import datapath
from .semantic_entropy import (
    CalculateSemanticEntropy,
    CalculateSemanticEntropyDiscrete,
    ClusterConfidence
)
from .calibration import (
    ProbabilityCalibration,
    TrainCalibrationModel,
    TemperatureScalingCalibration,
    TemperatureScaling
)

import json
import datetime
import os
import re
import threading
import math
import traceback
import random

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple



model_loaded = {}
current_timestamp = datetime.datetime.now().strftime('%H_%M_%S')
current_timestamp_day = datetime.datetime.now().strftime('%Y_%m_%d')
lock = threading.Lock()


def ExtractAnswer(answer, task):
    if task == 'GSM':
        return answer.split('#### ')[1]
    elif task == 'StrategyQA' or task == 'SQuAD' or task == 'MMLU' or task == 'SciQ' or task == 'BBH' or task == 'CMS' or task == 'BIGGSM' or task == 'MATH' or task == 'GPQA' or task == "MMLUPRO":
        return answer
    else:
        ValueError('Task not supported')
    return None


def LoadDebateAgents(args, pos):
    debate_agents = []
    for agent_name, agent_info in debaters.items():
        debater_model_info = model_info[agent_info['model']]
        debater_model_info['name'] = agent_name
        if debater_model_info['model_name'] in model_loaded:
            model = model_loaded[debater_model_info['model_name']]
        else:
            model = LoadModel(debater_model_info, pos)
        d = Debator(
            agent_name=agent_name,
            model=model,
            agent_info=agent_info,
            use_role=True
        )
        debate_agents.append(d)
    return debate_agents[:args.debate_agents]


def DebateAgents(args):
    debate_agents = {}
    for i, (agent_name, agent_info) in enumerate(debaters.items()):
        if i >= args.debate_agents:
            break
        debater_model_info = model_info[agent_info['model']]
        debater_model_info['name'] = agent_name
        debate_agents[agent_name] = debater_model_info['model_name']
    return debate_agents


def LoadAssistantModel(assistant):
    if assistant in model_loaded:
        model = model_loaded[assistant]
    else:
        a_model_info = model_info[assistant]
        a_model_info['name'] = 'assistant'
        model = LoadModel(a_model_info)
        model_loaded[assistant] = model
    return model


def LoadAssistantModels(args):
    asst_1 = LoadAssistantModel(args.assistant_1)
    asst_2 = LoadAssistantModel(args.assistant_2)

    return asst_1, asst_2


def GetEmbeddings(text_list):
    from openai import OpenAI

    client = OpenAI(
	    api_key=os.environ["OPENAI_API_KEY"],
    )

    res = []
    for text in text_list:
        response = client.embeddings.create(
	        input=text,
	        model="text-embedding-3-small",
        )
        em = response.data[0].embedding
        res.append(em)
        
    return np.array(res)


def compute_embedding_distance(emb1, emb2):
	sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
	return 1 - sim


def quality_pruning(query_embedding, response_embeddings, k):
    n = response_embeddings.shape[0]

    if n <= k:
        return list(range(n))

    distances = []
    for i in range(n):
        dist = compute_embedding_distance(query_embedding, response_embeddings[i])
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    selected_indices = [idx for idx, _ in distances[:k]]

    return selected_indices


def diversity_pruning(response_embeddings, k, candidate_indices):
	if candidate_indices is None:
		candidate_indices = list(range(response_embeddings.shape[0]))

	n = len(candidate_indices)

	if n <= k:
		return candidate_indices

	selected_indices = []
	remaining_indices = candidate_indices.copy()

	if len(remaining_indices) > 0:
		center = np.mean([response_embeddings[i] for i in remaining_indices], axis=0)
		center = center / np.linalg.norm(center)  # 归一化

		min_dist = float('inf')
		first_idx = -1
		for i in remaining_indices:
			dist = compute_embedding_distance(center, response_embeddings[i])
			if dist < min_dist:
				min_dist = dist
				first_idx = i

		selected_indices.append(first_idx)
		remaining_indices.remove(first_idx)

	while len(selected_indices) < k and remaining_indices:
		max_diversity = -float('inf')
		max_idx = -1

		for i in remaining_indices:
			diversity_score = 0
			for j in selected_indices:
				diversity_score += compute_embedding_distance(
					response_embeddings[i], response_embeddings[j]
				)

			if diversity_score > max_diversity:
				max_diversity = diversity_score
				max_idx = i

		if max_idx != -1:
			selected_indices.append(max_idx)
			remaining_indices.remove(max_idx)
		else:
			break

	return selected_indices


def misconception_refutation(items, debaters, question):
    debaters_map = {d.agent_name: d for d in debaters}
    for item in items:
        model = debaters_map[item['agent_name']]
        messages = [
            {
                'role': 'system',
                'content': misconception_identify['system']
            },
            {
                'role': 'user',
                'content': misconception_identify['user'].format(
                    question=question,
                    answer=item['response']
                )
            }
        ]
        
        response, _ = model.debate(messages)
        response_split = response.split('Response:')
        if len(response_split) < 1:
            print(response)
        issue_list = response_split[-1]
        
        messages = [
            {
                'role': 'system',
                'content': misconception_fix['system'].format(
                    debater=item['agent_name'],
                    question=question,
                    response=item['response'],
                    issues=issue_list
                )
            },
            {
                'role': 'user',
                'content': misconception_fix['user']
            }
        ]

        response, _ = model.debate(messages)
        item['response'] = response
    
    return items
        


def SaveToFile(args, debate_agents, debate_history):
    save_data = {}
    save_data['args'] = vars(args).copy()
    save_data['args']['debate_agents'] = debate_agents.copy()
    save_data['args']['debate_turns'] = args.debate_turns

    sorted_keys = sorted(debate_history.keys())
    sorted_list = [debate_history[key] for key in sorted_keys]
    sorted_list.insert(0, save_data['args'])

    path = os.path.join(args.output_dir, args.task, current_timestamp_day)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, current_timestamp + '.json'), 'w') as f:
        json.dump(sorted_list, f, indent=4)
        
        
def SupportLogprob(model):
    if 'claude' in model.lower():
        return False
    elif 'gpt' in model.lower() or 'llama' in model.lower() or 'deepseek' in model.lower():
        return True


def DebateOneByOne(args):
    intervals_start = list(range(args.low_index, args.up_index, args.save_interval))
    intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}
    
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]
    if args.single or args.cot:
        args.debate_turns = 1

    combined_results = {}
    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")


def DebateOneByOneFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['noconf']
    if args.single:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['nodebate']
        args.debate_turns = 1
    
    elif args.cot:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['cot']
        args.debate_turns = 1

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                # if len(history) == 0:
                if len(history) < len(debate_agents):
                    # No chat history, we do initial debate
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                else:
                    # Debate afterwards
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                })

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneSelfElicit(args):
    print('Debating with self-elicitation confidence')
    interval = (args.up_index - args.low_index) // args.num_workers
    if args.calibration_train:
        intervals_start = list(
            range(args.low_index_calibration, args.up_index_calibration, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index_calibration]
    else:
        intervals_start = list(
            range(args.low_index, args.up_index, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}


    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneWithSelfElicitFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")

    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)


def DebateOneByOneWithSelfElicitFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    if args.categorical:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['self_elicit_categorical']
    else:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['self_elicit']

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        pattern = r"(?i)confidence score:\s*\d+"
        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                for _ in range(args.attempt_times):
                    # if len(history) == 0:
                    if len(history) < len(debate_agents):
                        # No chat history, we do initial debate
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['init']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item)
                        messages = [{'role': 'system', 'content': system_prompt}, {
                            'role': 'user', 'content': user_prompt}]
                        debate_response_ans, formatted_prompt = debater.debate(
                            messages)
                    else:
                        # Debate afterwards
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['debate']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item, history)
                        messages = [{'role': 'system', 'content': system_prompt}, {
                            'role': 'user', 'content': user_prompt}]
                        debate_response_ans, formatted_prompt = debater.debate(
                            messages)

                    match = re.search(pattern, debate_response_ans)
                    conf_score = None
                    if match:
                        if args.calibration:
                            pattern_int = r'\d+'
                            conf_score = int(
                                re.search(pattern_int, match.group()).group())
                            if args.calibration and args.calibration_scheme != 'temperature':
                                if args.categorical:
                                    conf_score *= 10
                                if args.calibration_conf is not None:
                                    conf_score_calibrated = ProbabilityCalibration(
                                        [conf_score], debater.model.model, args.task, args.calibration_conf, args.calibration_scheme)[0]
                                else:
                                    conf_score_calibrated = ProbabilityCalibration(
                                        [conf_score], debater.model.model, args.task, args.conf_mode, args.calibration_scheme)[0]
                                if args.categorical:
                                    conf_score_calibrated = discretize_confidence(
                                        conf_score_calibrated, bins=args.categorical_bins, clamp=True)
                                    conf_score /= 10
                            conf = f'Confidence score: {conf_score_calibrated}'
                            debate_response_ans = re.sub(
                                pattern, conf, debate_response_ans)
                        break

                if not match:
                    debate_response_ans = debate_response_ans + \
                        f'\nConfidence score: {50}'

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                    'Confidence score': conf_score
                })

                if args.calibration:
                    history[-1]['Calibrated confidence score'] = conf_score_calibrated

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneWithLogprob(args):
    print('Debating with logprob confidence')
    if args.calibration_train:
        intervals_start = list(
            range(args.low_index_calibration, args.up_index_calibration, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index_calibration]
    else:
        intervals_start = list(
            range(args.low_index, args.up_index, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}

    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneWithLogprobFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")

    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)
        
        
def discretize_confidence(score_0_100: float, bins: int = 10, clamp: bool = True) -> int:
    if clamp:
        score_0_100 = max(0.0, min(100.0, float(score_0_100)))
    width = 100.0 / bins
    if score_0_100 == 100.0:
        return bins
    return int(score_0_100 // width)


def DebateOneByOneWithLogprobFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['conf']

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):

        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                for _ in range(args.attempt_times):
                    if len(history) < len(debate_agents):
                        # No chat history, we do initial debate
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['init']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item)
                        messages = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ]
                        
                        if (args.calibration or args.calibration_train) and args.calibration_scheme == 'temperature':
                            debate_response_ans, formatted_prompt, logprobs = debater.debate(
                                messages,
                                logprob=True,
                                top_logprobs=args.top_logprobs
                            )
                        else:
                            debate_response_ans, formatted_prompt, logprobs = debater.debate(
                                messages,
                                logprob=True
                            )

                    else:
                        # Debate afterwards
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['debate']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item, history)
                        messages = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt}
                        ]
                        
                        if (args.calibration or args.calibration_train) and args.calibration_scheme == 'temperature':
                            debate_response_ans, formatted_prompt, logprobs = debater.debate(
                                messages,
                                logprob=True,
                                top_logprobs=args.top_logprobs
                            )
                        else:
                            debate_response_ans, formatted_prompt, logprobs = debater.debate(
                                messages,
                                logprob=True
                            )

                    # print(debater.model.model, logprobs)
                    target_tokens = GetTargetLogprobs(debate_response_ans, logprobs)
                    if len(target_tokens) != 0:
                        break

                target_logprobs = [token.logprob for token in target_tokens]

                logprobs_sum = sum(target_logprobs)
                if args.conf_mode == 'seq_prob':
                    conf_score = round(math.exp(logprobs_sum) * 100)

                elif args.conf_mode == 'length_norm':
                    if len(target_logprobs) == 0:
                        conf_score = 50
                    else:
                        conf_score = round(math.exp(logprobs_sum) ** (1 / len(target_logprobs)) * 100)
                
                original_conf_score = conf_score
                if args.calibration and args.calibration_scheme != 'temperature':
                    if args.calibration_conf is not None:
                        conf_score = ProbabilityCalibration(
                            [conf_score], debater.model.model, args.task, args.calibration_conf, args.calibration_scheme)[0]
                    else:
                        conf_score = ProbabilityCalibration(
                            [conf_score], debater.model.model, args.task, args.conf_mode, args.calibration_scheme)[0]
                elif args.calibration and args.calibration_scheme == 'temperature':
                    conf_score = TemperatureScalingCalibration(target_tokens, debater.model.model, args.task, args.conf_mode)
                    
                if args.categorical:
                    original_conf_score = discretize_confidence(original_conf_score, bins=args.categorical_bins)
                    conf_score = discretize_confidence(conf_score, bins=args.categorical_bins)
                
                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n' + f'\nConfidence score: {conf_score}\n',
                    'Confidence score': original_conf_score,
                })
                
                
                if args.calibration_train:
                    history[-1]['Target logprobs'] = target_logprobs
                    if args.calibration_scheme == 'temperature':
                        top_logprobs = []
                        for token in target_tokens:
                            top_logprobs_token = [{'token_str': t.token, 'logprob': t.logprob} for t in token.top_logprobs]
                            pair = {'target_token': {'token_str': token.token, 'logprob': token.logprob} , 'top_logprobs': top_logprobs_token}
                            top_logprobs.append(pair)
                        history[-1]['top_logprobs'] = top_logprobs
                
                if args.calibration:
                    history[-1]['Calibrated confidence score'] = conf_score

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneWithSemanticEntropy(args):
    print("Debating with semantic entropy confidence score")
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]

    combined_results = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []

        for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
            print(f"Submitting task for range: {start} to {end}")
            futures.append(executor.submit(
                DebateOneByOneWithSemanticEntropyFunc, args, start, end, pos))

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Task completed with {len(result)} items")
                with lock:
                    combined_results.update(result)
            except Exception as e:
                print(f"Task generated an exception: {e}")
                traceback.print_exc()

    print(f"Total results collected: {len(combined_results)}")
    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)


def DebateOneByOneWithSemanticEntropyFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, pos=position)

    # Load debate prompt w.r.t. confidence type
    conf_type = args.conf_type
    if conf_type == 'score':
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['conf']
    else:
        ValueError('Confidence type not supported')

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):

        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                if len(history) == 0:
                    # if len(history) < len(debate_agents):
                    # No chat history, we do initial debate
                    # We get the answer from the model first
                    # Best generation, temperature = 0.1
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt, _ = debater.debate(
                        messages, logprob=True, temperature=1.0)
                else:
                    # Debate afterwards
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt, _ = debater.debate(
                        messages, logprob=True, temperature=1.0)

                if SupportLogprob(debater.model.model):
                    se_score, h, h_max = CalculateSemanticEntropy(
                        debater, messages, item['question'], M=10)
                else:
                    se_score, h, h_max = CalculateSemanticEntropyDiscrete(
                        debater, messages, item['question'], M=10)

                conf_score = round(se_score * 100)
                if args.calibration:
                    # conf_score = ProbabilityCalibration([conf_score], debater.model.model, args.task, args.conf_mode, args.calibration_scheme)[0]
                    conf_score = ProbabilityCalibration(
                        [conf_score], debater.model.model, args.task, 'self_elicit', args.calibration_scheme)[0]

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'semantic_entropy': h,
                    'max_entropy': h_max,
                    'confidence_score': conf_score,
                    'response': debate_response_ans + '\n' + f'\nConfidence score: {conf_score}\n',
                })

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneWithClusterConf(args):
    print("Debating with cluster confidence score")
    intervals_start = list(
        range(args.low_index, args.up_index, args.save_interval))
    intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}

    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) / args.num_workers
        start_indices = [int(start + i * interval)
                         for i in range(args.num_workers)]
        end_indices = start_indices[1:] + [end]
        if end_indices[-1] != end:
            end_indices[-1] = end

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneWithClusterConfFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")

    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)


def DebateOneByOneWithClusterConfFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt w.r.t. confidence type
    conf_type = args.conf_type
    if conf_type == 'score':
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['conf']
    else:
        ValueError('Confidence type not supported')

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                # if len(history) == 0:
                if len(history) < len(debate_agents):
                    # No chat history, we do initial debate
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)
                else:
                    # Debate afterwards
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                
                conf_score = ClusterConfidence(
                    debater, messages, item['question'], M=args.cluster_sample_times, num_threads=2)
                original_conf_score = conf_score
                if args.calibration and args.calibration_scheme != 'temperature':
                    if args.calibration_conf is not None:
                        conf_score = ProbabilityCalibration(
                            [conf_score], debater.model.model, args.task, args.calibration_conf, args.calibration_scheme)[0]
                    else:
                        conf_score = ProbabilityCalibration(
                            [conf_score], debater.model.model, args.task, args.conf_mode, args.calibration_scheme)[0]
                    
                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n' + f'\nConfidence score: {conf_score}\n',
                    'Confidence score': original_conf_score,
                })

                if args.calibration:
                    history[-1]['Calibrated confidence score'] = conf_score

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneWithRandomConf(args):
    print('Debating with random confidence score')
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]

    combined_results = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []

        for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
            print(f"Submitting task for range: {start} to {end}")
            futures.append(executor.submit(
                DebateOneByOneWithRandomConfFunc, args, start, end, pos))

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Task completed with {len(result)} items")
                with lock:
                    combined_results.update(result)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    print(f"Total results collected: {len(combined_results)}")
    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)


def DebateOneByOneWithRandomConfFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt w.r.t. confidence type
    debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['conf']

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):

        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                if len(history) == 0:
                    # No chat history, we do initial debate
                    # We get the answer from the model first
                    # Best generation, temperature = 0.1
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)
                else:
                    # Debate afterwards
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                conf_score = random.randint(40, 59)

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'confidence_score': conf_score,
                    'response': debate_response_ans + '\n' + f'\nConfidence score: {conf_score}\n',
                })

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneInterventions(args):
    intervals_start = list(range(args.low_index, args.up_index, args.save_interval))
    intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}
    
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]
    if args.single or args.cot:
        args.debate_turns = 1

    combined_results = {}
    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneInterventionsFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")


def DebateOneByOneInterventionsFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['noconf']
    if args.single:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['nodebate']
        args.debate_turns = 1
    
    elif args.cot:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['cot']
        args.debate_turns = 1

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        history = []
        query_embedding = GetEmbeddings([item['question']])
        z_all = []
        for _ in range(args.debate_turns):
            for debater in debate_agents:
                # if len(history) == 0:
                if len(history) < len(debate_agents):
                    # No chat history, we do initial debate
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                else:
                    # Debate afterwards
                    # We get the answer from the model first
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, z_all)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                })
            
            for i in range(len(debate_agents)):
                z_all.append(history[-i-1])
                
            response_embeddings = GetEmbeddings([z['response'] for z in z_all])
            quality_indices = quality_pruning(query_embedding, response_embeddings, len(z_all))
            final_indices = diversity_pruning(response_embeddings, len(debate_agents), quality_indices)
            z_all = [z_all[i] for i in final_indices]
            z_all = misconception_refutation(z_all, debate_agents, result['question'])
            
        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateOneByOneMultiPersona(args):
    intervals_start = list(range(args.low_index, args.up_index, args.save_interval))
    intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}
    
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]
    if args.single or args.cot:
        args.debate_turns = 1

    combined_results = {}
    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateOneByOneMultiPersonaFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")
    
    
def DebateOneByOneMultiPersonaFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    debate_prompts = debate_prompt_multi_persona

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }
        
        moderator = debate_agents[-1]
        debate_agents = debate_agents[:-1]
        history = []
        for _ in range(args.debate_turns):
            for i, debater in enumerate(debate_agents):
                if len(history) == 0:
                    # Meta prompt for init answer
                    prompt_ans = debate_prompts['meta_prompt']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                else:
                    # Debate afterwards
                    if i == 0:
                        prompt_ans = debate_prompts['affirmative']
                    else:
                        prompt_ans = debate_prompts['negative']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                history.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                })

            prompt_ans = debate_prompts['moderator']
            system_prompt, user_prompt = GetPromptsDebate(
                args.task, prompt_ans, moderator, item, history)
            messages = [{'role': 'system', 'content': system_prompt}, {
                'role': 'user', 'content': user_prompt}]
            debate_response_ans, formatted_prompt = moderator.debate(
                messages)
            history.append({
                'agent_name': moderator.agent_name,
                'agent_model': moderator.model.model,
                'prompt': formatted_prompt,
                'response': debate_response_ans + '\n',
            })

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateSimultaneousChatEval(args):
    intervals_start = list(range(args.low_index, args.up_index, args.save_interval))
    intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}
    
    interval = (args.up_index - args.low_index) // args.num_workers
    start_indices = list(range(args.low_index, args.up_index, interval))
    end_indices = start_indices[1:] + [args.up_index]
    if args.single or args.cot:
        args.debate_turns = 1

    combined_results = {}
    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateSimultaneousChatEvalFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")
    
    
def DebateSimultaneousChatEvalFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    debate_prompts = debate_prompt_chat_eval

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }
        
        moderator = debate_agents[-1]
        debate_agents = debate_agents[:-1]
        history = []
        for _ in range(args.debate_turns):
            buffer = []
            for i, debater in enumerate(debate_agents):
                if _ == 0:
                    # Meta prompt for init answer
                    prompt_ans = debate_prompts['init']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                else:
                    # Debate afterwards
                    prompt_ans = debate_prompts['debate']
                    system_prompt, user_prompt = GetPromptsDebate(
                        args.task, prompt_ans, debater, item, history)
                    messages = [{'role': 'system', 'content': system_prompt}, {
                        'role': 'user', 'content': user_prompt}]
                    debate_response_ans, formatted_prompt = debater.debate(
                        messages)

                buffer.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                })
            history.extend(buffer)

            prompt_ans = debate_prompts['summary']
            system_prompt, user_prompt = GetPromptsDebate(
                args.task, prompt_ans, moderator, item, history)
            messages = [{'role': 'system', 'content': system_prompt}, {
                'role': 'user', 'content': user_prompt}]
            debate_response_ans, formatted_prompt = moderator.debate(
                messages)
            history.append({
                'agent_name': moderator.agent_name,
                'agent_model': moderator.model.model,
                'prompt': formatted_prompt,
                'response': debate_response_ans + '\n',
            })

        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history


def DebateSimultaneousSelfElicit(args):
    print('Debating with self-elicitation confidence sim')
    interval = (args.up_index - args.low_index) // args.num_workers
    if args.calibration_train:
        intervals_start = list(
            range(args.low_index_calibration, args.up_index_calibration, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index_calibration]
    else:
        intervals_start = list(
            range(args.low_index, args.up_index, args.save_interval))
        intervals_end = intervals_start[1:] + [args.up_index]
    combined_results = {}


    for start, end in zip(intervals_start, intervals_end):
        interval = (end - start) // args.num_workers
        start_indices = list(range(start, end, interval))
        end_indices = start_indices[1:] + [end]
        end_indices[-1] = max(end_indices[-1], end)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(args.num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(
                    DebateSimultaneousWithSelfElicitFunc, args, start, end, pos))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"Task completed with {len(result)} items")
                    with lock:
                        combined_results.update(result)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    print("Detailed traceback:")
                    print(traceback.format_exc())

        with lock:
            SaveToFile(args, DebateAgents(args), combined_results)

    print(f"Total results collected: {len(combined_results)}")

    with lock:
        debate_agents = DebateAgents(args)
        if args.calibration_train:
            combined_results = TrainCalibrationModel(
                combined_results, debate_agents, args)
        SaveToFile(args, DebateAgents(args), combined_results)


def DebateSimultaneousWithSelfElicitFunc(args, start_index=None, end_index=None, position=0):
    # Load dataset
    if args.calibration_train:
        data_path = datapath.data_path_valid
    else:
        data_path = datapath.data_path
    debate_data = load_jsonl_data(data_path[args.task])

    # Load debate agents
    debate_agents = LoadDebateAgents(args, position)

    # Load debate prompt
    if not args.categorical:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['self_elicit']
    else:
        debate_prompts = debate_prompt_ground_truth_answer_and_reason[args.task]['self_elicit_categorical']

    debate_history = {}
    # Iterate over the dataset
    for idx, item in enumerate(tqdm(debate_data[start_index:end_index], desc=f'Debating {start_index}:{end_index}', position=position)):
        real_index = idx + start_index
        # Result list for debating
        result = {
            'question': item['question'],
            'context': item['context'] if 'context' in item else None,
            'ground_truth': ExtractAnswer(item['answer'], args.task),
            'debate_history': [],
        }

        pattern = r"(?i)confidence score:\s*\d+"
        history = []
        for _ in range(args.debate_turns):
            buffer = []
            for debater in debate_agents:
                for _ in range(args.attempt_times):
                    # if len(history) == 0:
                    if len(history) < len(debate_agents):
                        # No chat history, we do initial debate
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['init']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item)
                        messages = [{'role': 'system', 'content': system_prompt}, {
                            'role': 'user', 'content': user_prompt}]
                        debate_response_ans, formatted_prompt = debater.debate(
                            messages)
                    else:
                        # Debate afterwards
                        # We get the answer from the model first
                        prompt_ans = debate_prompts['debate']
                        system_prompt, user_prompt = GetPromptsDebate(
                            args.task, prompt_ans, debater, item, history)
                        messages = [{'role': 'system', 'content': system_prompt}, {
                            'role': 'user', 'content': user_prompt}]
                        debate_response_ans, formatted_prompt = debater.debate(
                            messages)

                    match = re.search(pattern, debate_response_ans)
                    conf_score = None
                    if match:
                        if args.calibration:
                            pattern_int = r'\d+'
                            conf_score = int(
                                re.search(pattern_int, match.group()).group())
                            if args.categorical:
                                conf_score *= 10
                            if args.calibration and args.calibration_scheme != 'temperature':
                                if args.calibration_conf is not None:
                                    conf_score_calibrated = ProbabilityCalibration(
                                        [conf_score], debater.model.model, args.task, args.calibration_conf, args.calibration_scheme)[0]
                                else:
                                    conf_score_calibrated = ProbabilityCalibration(
                                        [conf_score], debater.model.model, args.task, args.conf_mode, args.calibration_scheme)[0]
                            if args.categorical:
                                conf_score_calibrated = discretize_confidence(conf_score_calibrated)
                            conf = f'Confidence score: {conf_score_calibrated}'
                            debate_response_ans = re.sub(
                                pattern, conf, debate_response_ans)
                        break

                if not match:
                    debate_response_ans = debate_response_ans + \
                        f'\nConfidence score: {50}'

                buffer.append({
                    'agent_name': debater.agent_name,
                    'agent_model': debater.model.model,
                    'prompt': formatted_prompt,
                    'response': debate_response_ans + '\n',
                    'Confidence score': conf_score
                })

                if args.calibration:
                    buffer[-1]['Calibrated confidence score'] = conf_score_calibrated

            # Add the buffer to the history
            history.extend(buffer)
            
        result['debate_history'] = history
        debate_history[real_index] = result

    return debate_history