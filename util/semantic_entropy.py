import os
import math
import time
import json
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data_processing import (
    GetTargetLogprobs
)

prompt = 'You are going to evaluating answers to the question {question} Here are two possible answers: Possible Answer 1: {text1} Possible Answer 2: {text2} Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.'

cluster_prompt = 'You are a helpful AI assistant. The user will provide you with a question and a set of numbered answers to that question. Your task is to cluster these answers. You are required to identify the answers that are semantically identical or similar according to the question. Then, group these answers into clusters by grouping their corresponding numbers together. Finally, identify the largest cluster and provide the number of answers it contains. Please ensure that the answers are grouped based on the question. If the answers are semantically identical for a given question, they should be clustered together, regardless of differences in formatting or phrasing. When encountering multiple-choice and calculation questions, carefully examine the comparison options and the details within the provided choices. Please strictly adhere to the following output format:\nCluster: [your cluster result, in the format of list]\nAnswer: [The number of replies the largest cluster contains, only a number]'

judge_client = openai.Client(
    api_key=os.getenv('OPENAI_API_KEY'),
)

def Entailment(question, text1, text2):
    user_prompt = prompt.format(question=question, text1=text1, text2=text2)
    messages = [{'role': 'system', 'content': user_prompt}]
    
    response = judge_client.chat.completions.create(
		model='gpt-4o-mini',
		messages=messages,
	)
    
    res = response.choices[0].message.content
    
    return res
    

def GenerateSample(model, prompt, logprob=False):
    if logprob:
        response, _, token_list = model.debate(prompt, logprob=logprob)
        target_tokens = GetTargetLogprobs(response, token_list)
        if len(target_tokens) == 0:
            return None

        return {
            'content': ''.join([t.token for t in target_tokens]),
            'logprob': [t.logprob for t in target_tokens]
        }
    else:
        response, _ = model.debate(prompt, logprob=logprob)

        return {'content': response}
    

def Generate(model, prompt, M=15, num_threads=5, logprob=False):
    responses = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {executor.submit(GenerateSample, model, prompt, logprob): i for i in range(M)}
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    responses[i] = result
            except Exception as e:
                print(f"Task {i} generated an exception: {e}")
    
    return responses


def Cluster(responses, question):
    clusters = {}
    cluster_cnt = 0
    judge_cnt = 0
    for r_key, r_value in responses.items():
        added = False
        for c_key, c_items in clusters.items():
            res = Entailment(question, r_value['content'], c_items[0])
            judge_cnt += 1
            if 'entailment' in res.lower():
                c_items.append(r_value)
                added = True
                break
            
            
        if not added:
            clusters[cluster_cnt] = [r_value]
            cluster_cnt += 1
    return clusters


def ClusterAtOnce(responses, question):
    responses = dict(sorted(responses.items(), key=lambda x: int(x[0])))
    user_prompt = \
		'Question: ' + question + '\nNumbered Answers:\n' + ''.join(
			['\nAnswer ' + str(key) + ':' + value['content'].split('Answer:')[-1] for key, value in
			 responses.items()])
    
    messages = [{'role': 'system', 'content': cluster_prompt}, {'role': 'user', 'content': user_prompt}]
    response = judge_client.chat.completions.create(
		model='gpt-4o-mini',
		messages=messages,
	)
    
    try:
        extracted_text = response.choices[0].message.content.lower().split('answer:')[1].strip()
        max_cluster = int(extracted_text)

    except (IndexError, ValueError) as e:
        max_cluster = len(responses) / 2
        
    return max_cluster


epsilon = 1e-12
def Compute(clusters):
    class_probs = []
    for _, c_responses in clusters.items():
        seq_probs = [math.exp(sum(r['logprob'])) for r in c_responses]
        class_probs.append(sum(seq_probs))
    
    total_probs = sum(class_probs) if sum(class_probs) > 0 else sum(class_probs) + epsilon
    
    for index, c_responses in clusters.items():
        c_responses.append({'likelihood': class_probs[index] / total_probs})

    h_max = math.log(len(clusters)) if len(clusters) > 1 else math.log(1 + epsilon)
    
    entropy_items = []
    for _, c in clusters.items():
        p = c[-1]['likelihood']
        if p == 0:
            p += epsilon
        entropy_items.append(-p * math.log(p))
    
    h = sum(entropy_items)
    if (1 - (h / h_max)) < 0:
        raise ValueError('Semantic entropy is negative')
    return clusters, h, h_max


def ComputeDiscrete(clusters, M):
    class_probs = [len(c) / M for _, c in clusters.items()]
    
    h_max = math.log(len(clusters))
    if h_max == 0:
        return clusters, 0, epsilon
    
    for index, c_responses in clusters.items():
        c_responses.append({'likelihood': class_probs[index]})
        
    entropy_items = []
    for _, c in clusters.items():
        p = c[-1]['likelihood']
        if p == 0:
            p += epsilon
        entropy_items.append(-p * math.log(p))
        
    h = sum(entropy_items)
    if (1 - (h / h_max)) < 0:
        print(clusters, h , h_max)
        raise ValueError('Semantic entropy is negative')
    return clusters, h, h_max


def CalculateSemanticEntropy(model, prompt, question, M):
    samples = Generate(model, prompt, M, num_threads=5, logprob=True)
    clusters = Cluster(samples, question)
    clusters, h, h_max = Compute(clusters)
    return 1 - (h / h_max), h, h_max


def CalculateSemanticEntropyDiscrete(model, prompt, question, M):
    samples = Generate(model, prompt, M, num_threads=5, logprob=False)
    clusters = Cluster(samples, question)
    clusters, h, h_max = ComputeDiscrete(clusters, M)
    return 1 - (h / h_max), h, h_max


def ClusterConfidence(model, prompt, question, M, num_threads=5):
    samples = Generate(model, prompt, M, num_threads=num_threads, logprob=False)
    max_cluster = ClusterAtOnce(samples, question)
    # save_samples(question, samples, clusters)

    return int(max_cluster / M * 100)


def save_samples(question, samples, clusters):
    data_to_append = {
        "question": question,
        "samples": samples,
        "max_cluster": max([len(c) for c in clusters.values()])
    }
    json_file_path = './test/cluster_probs.json'
    try:
        with open(json_file_path, 'r') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    except FileNotFoundError:
        existing_data = []

    existing_data.append(data_to_append)

    with open(json_file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

if __name__ == "__main__":
    text = [{'role': 'user', 'content': 'Please tell me the capital of Taiwan'}]
    question = 'What is the capital of Taiwan?'
    responses = {
        0: {
            'content':'It is Taipei',
            'logprob': [0.6, 0.8, 0.3]
        },
        1: {
            'content': 'Taipei',
            'logprob': [0.6, 0.5, 0.6]
        },
        2: {
            'content': 'Taipei is the capital of Taiwan',
            'logprob': [0.7, 0.6, 0.6, 0.1]
        },
        3: {
            'content': 'Taipei city',
            'logprob': [0.2, 0.1]
        },
        4: {
            'content': 'Beijing',
            'logprob': [0.2, 0.1, 0.1, 0.2]
        },
    }
    clusters = Cluster(responses, question)
    clusters, h, h_max = Compute(clusters)
    print(1 - (h / h_max))