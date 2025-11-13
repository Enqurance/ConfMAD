assistant_prompts = {
    "system": "You are an helpful assistant. You can help the user to find information, answer questions, and provide suggestions.",
}

debate_prompt_free_form = (
    "Question: {question}"
	"[The Start of Assistant 1's Answer]\n {response_one} \n[The End of Assistant 1's Answer]\n"
	"[The Start of Assistant 2's Answer]\n {response_two} \n[The End of Assistant 2's Answer]\n"
	"We would like to request your feedback on the performance of two AI assistants in "
	"response to the user question displayed above. Please consider the helpfulness, relevance, "
	"accuracy, and level of detail of their responses. Each assistant receives an overall score "
	"on a scale of 1 to 10, where a higher score indicates better overall performance. There are "
	"a few other referees assigned the same task, it's your responsibility to discuss with them and "
	"think critically before you make your final judgment. "
	"Here is your discussion history: {chat_history}"
    "{role_description} "
	"Now it's your time to talk, please make your talk short and clear, {agent_name}"
 	"Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. "
    "Then, output two lines indicating the scores for Assistant 1 and 2, respectively. "
	"Remember that you are not required to output the same value as other referees! "
	"Output with the following format strictly:"
	"Evaluation evidence: [your explanation here]"
	"The score of Assistant 1: [score only]"
	"The score of Assistant 2: [score only]"
)


self_conf_elicit_prompt_ground_truth = {
    "Self-Probing":
        {
            "GSM" : 
            {
                "init": 
                {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "You are required to provide your reason, answer to the question, and a confidence score. "
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "When giving your confidence score, please be aware that not to be overconfident. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n"
	                "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}"
                },
                
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please debate about the answer to the question based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning. "
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "When giving your confidence score,  "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here].\n"
	                "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                }
            },
            "StrategyQA" : 
            {
                "init": 
                {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please answer whether the statement in the question is true or false. "
                    "You are required to provide your reason, answer to the question, and a confidence score. The answer shall only contain true or false."
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only true or false].\n"
	                "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}"
                },
                
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please answer whether the statement in the question is true or false based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning. "
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only true or false].\n"
	                "Confidence score: [score only, 0-100].\n",
                    
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                }
            },
            "SQuAD": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a context on a specific topic. "
                    "Then, the user will ask you a question based on the context. "
                    "Please answer the question according to the context given by the user."
                    "You are required to provide your reason, answer to the question, and a confidence score. "
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please give your answer briefly. Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n"
                    "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Context: {context}\n"
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the context, the question and the debate history of the question. "
                    "Please answer the question according to the context given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning. Please give your answer briefly."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here].\n"
                    "Confidence score: [score only, 0-100].\n",
                
                    "user":
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "MMLU": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason, answer to the question, and a confidence score. "
                    "Your answer should be one of the four options provided by the user."
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n"
                    "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n"
                    "Confidence score: [score only, 0-100].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "SciQ": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason, answer to the question, and a confidence score. "
                    "Your answer should be one of the four options provided by the user."
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n"
                    "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n"
                    "Confidence score: [score only, 0-100].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "BBH": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant.  The user will give you a question. "
                    "Please provide the answer to the question given by the user. "
                    "You are required to provide your reason, answer to the question, and a confidence score. "
                    "Note that the confidence score is how likely you think the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n"
                    "Confidence score: [score only, 0-100].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "                    
                    "Please provide the answer to the question given by the user based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning."
                    "You do not have to provide the same confidence score as other debaters. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n"
                    "Confidence score: [score only, 0-100].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            }
        }
    }


debate_prompt_ground_truth = {
    "Default":
        {
            "GSM": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please debate about the answer to the question based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "StrategyQA": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please answer whether the statement in the question is true or false. "
                    "You are required to provide your reason and answer to the question. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only true or false].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please answer whether the statement in the question is true or false based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only true or false].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                }
            }, 
            "SQuAD": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a context on a specific topic. "
                    "Then, the user will ask you a question based on the context. "
                    "Please answer the question according to the context given by the user. "
                    "You are required to provide your reason and answer to the question. Please give your answer briefly."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n",

                    "user":
                    "Context: {context}\n"
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the context, the question and the debate history of the question. "
                    "Please answer the question according to the context given by the user and based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. Please give your answer briefly."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here].\n",
                
                    "user":
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "MMLU": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "SciQ": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "BBH": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question. "
                    "Please provide the answer to the question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question. "
                    "Please provide the answer to the question given by the user based on the answers and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            }
        },
    "Logprob":
        {
            "GSM": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "You are required to provide your reason, answer to the question. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only answer here].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please debate about the answer to the question based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. "
                    "Note that the confidence score is how likely the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only answer here].\n",

                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                }
            },
            "StrategyQA": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please answer whether the statement in the question is true or false. "
                    "You are required to provide your reason and answer to the question. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only true or false].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the question and the debate history of the question. "
                    "Please answer whether the statement in the question is true or false based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. "
                    "Note that the confidence score is how likely the answer is correct and it should be a number between 0 and 100. "
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only true or false].\n",
                    
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                }
            }, 
            "SQuAD": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a context on a specific topic. "
                    "Then, the user will ask you a question based on the context. "
                    "Please answer the question according to the context given by the user. "
                    "You are required to provide your reason and answer to the question. Please give your answer briefly."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here].\n",

                    "user":
                    "Context: {context}\n"
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. You are now required to debate with other language models about the answer to a certain question. "
                    "The user will give you the context, the question and the debate history of the question. "
                    "Please answer the question according to the context given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question and include your debating arguments in the reasoning. Please give your answer briefly."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here].\n",
                
                    "user":
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "MMLU": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            },
            "SciQ": {
                "init": {
                    "system":
                    "You are {debater}, a helpful AI assistant. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "You are required to provide your reason and answer to the question. "
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your reason here].\n"
                    "Answer: [your answer for the question here, only the option].\n",

                    "user":
                    "Question: {question}"
                },
                "debate": {
                    "system":
                    "You are {debater}. The user will give you a question with four options. "
                    "Please choose the answer you believe is correct from the four options provided to respond to the question given by the user. "
                    "Please answer the question given by the user and based on the answers, confidence scores and reasons of other debaters. "
                    "You are required to provide your answer to the question, a confidence score, and include your debating arguments in the reasoning."
                    "Your answer should be one of the four options provided by the user."
                    "Please think critically before answering the question and output with the following output format strictly:\n"
                    "Reason: [your debating arguments].\n"
                    "Answer: [your answer for the question here, only the option].\n",
                
                    "user":
                    "Question: {question}\n"
                    "Debate_history:\n {debate_history}"
                },
            }
        }
    }

debate_prompt_ground_truth_answer_and_reason = {
    "GSM": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "nodebate": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please output in the following format strictly:\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "cot": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step to slove the question. Then, please output your answer in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "BBH": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to debate with other debaters on a question. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "You do not need to output your confidence score. Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here briefly]\n"
                    "Answer: [your answer for the question here, must select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "nodebate": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please output in the following format strictly:\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "cot": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step to slove the question. "
                    "Then, please output your answer in the following format strictly:\n"
                    "Reason: [your reasoning steps, not too long]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "MATH": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "CMS": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "BIGGSM": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only a number]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on the reasons, answers, and confidence scores of other debaters, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only a number]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only a number]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only a number]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only a number]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only a number]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, only the answer]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        },
        "cot": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a math question given by the user. "
                    "Please think step by step to slove the question. "
                    "Then, please output your answer in the following format strictly:\n"
                    "Reason: [your reasoning steps, not too long]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
    },
    "MMLU": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here briefly]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    # "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    # "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here briefly]\n"
                    "Answer: [your answer for the question here, must select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here briefly]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "nodebate": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please output in the following format strictly:\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "cot": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step to slove the question. "
                    "Then, please output your answer in the following format strictly:\n"
                    "Reason: [your reasoning steps, not too long]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "MMLUPRO": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step in your reasoning part and give your reasoning for the question, Then, give your answer to the question. "
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, think step by step and formulate your debate arguments and provide your answer to the question."
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here briefly]\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step in your reasoning part and give your reasoning for the question, Then, give your answer to the question. "
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question. "
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [Select one option from the given choices, answers must begin with the option identifier]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "nodebate": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please include your options by enclosing the option identifier in parentheses."
                    "Please output in the following format strictly:\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please include your options by enclosing the option identifier in parentheses. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [select one option from the given choices, answers must begin with the option identifier]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "MATH": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a math question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer here]\n",
                    
            
                "user": 
                    "Question: {question}\n"

            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a math question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on the reasons, answers, and confidence scores of other debaters, formulate your debate arguments and provide your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here, begin with this]\n"
                    "Answer: [your answer here]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a math question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer here]\n",
                    
            
                "user": 
                    "Question: {question}\n",
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer here]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a math question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer here]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n",
                
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a math question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer here, please output in latex format]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n",
            }
        },
        "self_elicit_categorical": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a math question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer here]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",
            
                "user": 
                    "Question: {question}\n",
                
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a math question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons, answers, and most importantly, the confidence scores of other debaters. "
                    "The confidence score indicates how strongly a debater believes their answer is correct. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "Then, based on this information, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer here, please output in latex format]\n"
                    "Confidence score: [your confidence score only, 0-10]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n",
            }
        },
        "cot": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please think step by step to slove the question. "
                    "Then, please output your answer in the following format strictly:\n"
                    "Reason: [your reasoning steps, not too long]\n"
                    "Answer: [your answer for the question here, only the answer]\n",
            
                "user": 
                    "Question: {question}\n"
            },
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    },
    "GPQA": {
        "conf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system":
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "noconf": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please give your reasoning for the question, and give your answer to the question. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
                    "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question."
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n",
            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "self_elicit": {
            "init": {
                "system": 
                    "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
                    "Please provide a clear reasoning for your answer, followed by your answer to the question. "
                    "It is crucial to also include your confidence score, which reflects how strongly you believe your answer is correct. "
                    "Consider the confidence score carefully as it represents the likelihood of your answer being accurate. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your reason for the answer here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",
            
                "user": 
                    "Question: {question}\n"
            },
            "debate": {
                "system": 
                    "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
                    "PAY SPECIAL ATTENTION to these confidence scores as they reflect the reliability and conviction of each debater's argument. "
                    "If the confidence score is low, it may indicate uncertainty in the debater's answer. Please evaluate it further in this case."
                    "Then, formulate your debate arguments and provide your answer to the question."
                    "Finally, include your confidence score, which is a critical measure of how strongly you believe your answer is correct. "
                    "This score reflects the likelihood of your answer being accurate, so consider it with utmost importance. "
                    "Please output in the following format strictly:\n"
                    "Reason: [your debate arguments here]\n"
                    "Answer: [your answer for the question here, select one from given options]\n"
                    "Confidence score: [your confidence score only, 0-100]\n",

            
                "user": 
                    "Question: {question}\n"
                    "Debate history: {debate_history}\n"
            }
        },
        "only_ans": {
            "system": 
                "You are a helpful AI assistant. You are now required to answer a question given by the user."
                "Please only give the answer to the question. ",
            
            "user": 
                "Question: {question}\n"
        }
    }
}

debate_prompt_multi_persona = {
    # Meta prompt for initializing
    "meta_prompt": {
        "system":
            "You are a debater. Hello and welcome to the debate.\n"
            "It's not necessary to fully agree with each other's perspectives, "
            "as our objective is to find the correct answer.\n\n"
            "Output strictly in this format:\n"
            "Reason: [your reasoning here]\n"
            "Answer: [your answer here]\n",
        "user":
            "Question:\n{question}"
    },

    # Affirmative 
    "affirmative": {
        "system":
            "You are the **Affirmative Debater**.\n"
            "Present your reasoning and answer for the given topic, then defend it in subsequent rounds.\n\n"
            "Please consider the moderator's evaluation if it exists.\n"
            "Output strictly in this format:\n"
            "Reason: [your debate arguments here]\n"
            "Answer: [your answer here]\n",
        "user":
            "Question:\n{question}\n\n"
            "Debate history:\n{debate_history}"
    },

    # Negative
    "negative": {
        "system":
            "You are the **Negative Debater**.\n"
            "Disagree with the affirmative side, present your own reasoning and answer, "
            "and keep challenging the opponent in subsequent rounds.\n"
            "Please consider the moderator's evaluation if it exists.\n"
            "Output strictly in this format:\n"
            "Reason: [your debate arguments here]\n"
            "Answer: [your answer here]\n",
        "user":
            "Question:\n{question}\n\n"
            "Debate history:\n{debate_history}"
    },

    # Moderator
    "moderator": {
        "system":
            "You are the **Moderator**. There will be two debaters involved in a debate. They will present their answers and discuss their perspectives on a given question. "
            "At the end of each round, you will evaluate answers and decide which is correct. You, as the moderator, will evaluate both sides' answers and determine "
            "an answer candidate. Please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct."
            "Output strictly in this format:\n"
            "Reason: [your reasoning here]\n"
            "Answer: [your answer here]\n", 
            
        "user":
            "Question:\n{question}\n\n"
            "Debate history:\n{debate_history}"
    }
}


debate_prompt_chat_eval = {
    # Init Prompt
    "init": {
        "system": 
            "You are {debater}, a helpful AI assistant. You are now required to answer a question given by the user. "
            "Please give your reasoning for the question, and give your answer to the question. "
            "Please output in the following format strictly:\n"
            "Reason: [your reason for the answer here]\n"
            "Answer: [your answer for the question here, select one from given options]\n",
            
        "user": 
            "Question: {question}\n"
    },

    # Debate Prompt
    "debate": {
        "system": 
            "You are {debater}, a debater. You are now required to answer a question given by the user and debate with other debaters about the answer. "
            "The user will provide you with the question and the debate history, including the reasons and answers of other debaters. "
            "Then, based on the reasons, answers of other debaters, formulate your debate arguments and provide your answer to the question."
            "Please output in the following format strictly:\n"
            "Reason: [your debate arguments here]\n"
            "Answer: [your answer for the question here, select one from given options]\n",
            
        "user": 
            "Question: {question}\n"
            "Debate history: {debate_history}\n"
    },

    # Summary
    "summary": {
        "system":
            "You are a neutral summariser for this debate. At the end of the round you must:\n"
            "1.Briefly recap the key points made **in the most recent round only**.\n"
            "2.State the answer you now believe is most likely correct.\n\n"
            "Output strictly in this format:\n"
            "Summary: [your concise round-summary here]\n"
            "Answer: [your current best answer, choose from the given options]\n",

        "user":
            "Question: {question}\n"
            "Debate history: {debate_history}\n"
    }
}

misconception_identify = {
     "system": """
You are a helpful AI assistant. You are now required to evaluate a previous answer to a question.
Please evaluate this answer and identify any errors, misconceptions, or inconsistencies.
If you identify any such errors, please provide a short list of specific details, otherwise, just say "No errors found".
Then briefly discuss how the misconceptions can be fixed. You do not need to give a new answer.
""",

    "user": """
    Question: {question}\n
    Answer to Evaluate: {answer}\n
    Please output in the following format strictly and briefly:\n
    Response: [your list of errors and potential ways of fixing them briefly]\n
"""
}

misconception_fix = {
    "system": """
You are {debater}, a helpful AI assistant. You are now required to answer a question.
Please provide a clear reasoning for your answer, followed by your answer to the question.
Note that you may refer to the previous response and the possible issues, but you are not required to follow them.
Question: {question}\n
Previous Response: {response}\n
Possible Issues: {issues}\n
""",

    "user": """
    Please output in the following format strictly:\n
    Reason: [your new reason for your answer]\n
    Answer: [your new answer, only the answer to the question]\n

"""
}