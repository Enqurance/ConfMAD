import os
import json
import threading
import traceback

import openai
import anthropic

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

judge_file = [
    # "/home/zijie/Experiments/Debate/result/BBH/2025_01_27/17_20_27.json",
    # '/home/zijie/Experiments/Debate/result/BBH/2025_01_27/18_24_06.json',
    # "/home/zijie/Experiments/Debate/result/BBH/2025_01_27/18_55_46.json",
    "/home/zijie/Experiments/Debate/result/MMLU/2025_01_27/15_12_58.json",
    '/home/zijie/Experiments/Debate/result/MMLU/2025_01_27/15_23_18.json',
    # "/home/zijie/Experiments/Debate/result/MMLU/2025_01_27/16_40_51.json",
]

judge_prompt_no_conf = """
You are a judge tasked with evaluating a debate history provided by the user. The debate history includes a question, arguments, and answers presented by the debaters. Your responsibility is to carefully analyze the arguments and answers, then select the most reasonable and logically sound final answer based on the provided information. 

Please present your evaluation and final answer in the following format:

Reason: [Your detailed reasoning for selecting or proposing the answer]
Answer: [Your final answer, only the answer]

"""

judge_prompt_conf = """
You are a judge tasked with evaluating a debate history provided by the user. The debate history includes a question, arguments, answers, and confidence scores presented by the debaters. Your responsibility is to carefully analyze the arguments, answers, and confidence scores, then select the most reasonable and logically sound final answer based on the provided information. 

The confidence score reflects how certain a debater is about the correctness of its answer. When making your decision, you must give significant weight to the confidence scores, especially when the arguments and answers are equally plausible.

Please present your evaluation and final answer in the following format:

Reason: [Your detailed reasoning for selecting or proposing the answer, with specific consideration of confidence scores]
Answer: [Your final answer, only the answer]
"""


def ConcateFinalDebateHistory(item):
    history = item['debate_history'][-1]
    final_history = history['prompt'][-1]['content']
    final_response = history['response']
    final_debater = history['agent_name']
    debate_history = final_history + final_debater + ':\n' + final_response
    # print(debate_history)
    return debate_history


def ExtractInformation(content):
    answer = content.split('Answer:')[-1].split('\n')[0].strip()
    result = {
        'selected_answer': answer,
    }

    if 'confidence:' in content.lower():
        confidence = content.split('Confidence:')[-1].split('\n')[0].strip()
        result['selected_confidence'] = confidence

    return result


def Judge(args, data, lower_index, upper_index, pos):
    anthropic_client = anthropic.Anthropic(
        api_key=os.environ['ANTHROPIC_API_KEY']
    )

    final_result = {}

    idx = lower_index
    for item in tqdm(data[lower_index:upper_index], desc='Judging' + str(lower_index) + ':' + str(upper_index), position=pos):
        history = ConcateFinalDebateHistory(item)
        if args['debate_conf']:
            prompt = judge_prompt_conf
        else:
            prompt = judge_prompt_no_conf
        messages = [
            {
                "role": "user",
                "content": history
            }
        ]
        
        params = {
            "model": model,
            "system": prompt,
            "messages": messages,
            "max_tokens": 1024
        }
        
        response = anthropic_client.messages.create(
            **params
        )

        content = response.content[0].text
        result = ExtractInformation(content)
        result['response'] = content
        result['question'] = item['question']
        result['ground_truth'] = item['ground_truth']
        final_result[idx] = result
        idx += 1

    return final_result


if __name__ == '__main__':
    num_workers = 10
    model = "claude-3-5-sonnet-20241022"
    # model = "claude-3-5-haiku-20241022"
    
    for file in judge_file:
        with open(file, 'r') as f:
            result = json.load(f)
            args = result[0]
            result = result[1:201]
    
        up_index = len(result)
        interval = up_index // num_workers
        start_indices = list(range(0, up_index, interval))
        end_indices = start_indices[1:] + [up_index]
        lock = threading.Lock()
    
        combined_results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for start, end, pos in zip(start_indices, end_indices, range(num_workers)):
                print(f"Submitting task for range: {start} to {end}")
                futures.append(executor.submit(Judge, args, result, start, end, pos))

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

        print(f"Total results collected: {len(combined_results)}")
        if len(combined_results) > 0:
            sorted_results = [value for _, value in sorted(combined_results.items())]
            with lock:
                args['judge model'] = model
                sorted_results.insert(0, args)
                save_file_path = file.split('.json')[0] + '_judge_' + model + '.json'
                with open(save_file_path, 'w') as f:
                    json.dump(sorted_results, f, indent=4)
