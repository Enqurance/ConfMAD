import os
import openai
import anthropic
import transformers
import torch
import time

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import log_softmax

from util.prompt import assistant_prompts

import threading


class ChatCompletionTokenLogprob:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob
            
            
    def __repr__(self):
        return f"ChatCompletionTokenLogprob(token='{self.token}', logprob={self.logprob})"


class LanguageAssistant():
    def __init__(self, name, model):
        self.name = name
        self.model = model
        
    
    def generate_response(self):
        pass


class OpenAIAssistant(LanguageAssistant):
    def __init__(self, name, model, api_key=None, base_url=None, kwargs={}):
        super().__init__(name, model)
        self.api_key = api_key
        self.base_url = base_url
        self.openai_client = None
        self.kwargs = kwargs
        
        if api_key is None:
            raise ValueError("OPENAI_API_KEY must be provided if using openai models")
        elif base_url is None:
            self.openai_client = openai.OpenAI(
                api_key=self.api_key
            )
        else:
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
    def generate_response(self, messages, logprob=False, top_logprobs=None, temperature=None, top_p=None):
        super().generate_response()
        params = {
            "model": self.model,
            "messages": messages,
            "logprobs": logprob,
        }
        
        if top_logprobs:
            params["top_logprobs"] = top_logprobs
        
        
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        
        response = self.openai_client.chat.completions.create(
            **params
        )
        if logprob:
            token_list = response.choices[0].logprobs.content
            return response.choices[0].message.content, token_list
        return response.choices[0].message.content
    
    
class OpenRouterAPIAssistant(LanguageAssistant):
    def __init__(self, name, model, api_key=None, base_url=None, kwargs={}):
        super().__init__(name, model)
        self.api_key = api_key
        self.base_url = base_url
        self.openai_client = None
        self.kwargs = kwargs
        self.providers = {
            "llama": [
                "Fireworks"
            ],
            "qwen": ["Fireworks"],
            "deepseek": ["DeepSeek"],
            "phi": [
                "Nebius"
            ],
            "mixtral": ["Fireworks"]
        }
        
        if api_key is None:
            raise ValueError("OPENROUTER key must be provided if using openrouter llama models")
        self.openai_client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
            
    def generate_response(self, messages, logprob=False, top_logprobs=None, temperature=None, top_p=None):
        super().generate_response()
        
        provider = None
        for model in self.providers:
            if model in self.model:
                provider = self.providers[model]
                break
        
        params = {
            "model": self.model,
            "messages": messages,
            # "logprobs": logprob,
            "temperature": 1,
            "max_tokens": 2048,
            "extra_body": {
		        "provider": {
			        "order": provider,
			        "require_parameters": True
		        },
	        },
        }

        if len(provider) > 0 and provider[0] == 'Hyperbolic':
            params["logprobs"] = 1
        
        if logprob:
            params["logprobs"] = logprob
        
        if temperature is not None:
            params["temperature"] = temperature
            
        if top_logprobs is not None:
            if provider[0] == 'Fireworks':
                params["top_logprobs"] = 5
            else:
                params["top_logprobs"] = top_logprobs

        if logprob:
            while True:
                response = self.openai_client.chat.completions.create(
                    **params
                )
                if response is not None and response.choices is not None and response.choices[0].logprobs is not None:
                    break
                else:
                    print(response)
                    print("===== NO LOGPROBS =====")
                    time.sleep(5)
                
            token_list = response.choices[0].logprobs.content
            return response.choices[0].message.content, token_list
        
        while True:
            response = self.openai_client.chat.completions.create(
                **params
            )

            if response is not None and response.choices is not None:
                break
            else:
                print(response)
                time.sleep(5)
        return response.choices[0].message.content
    

class HuggingFaceModelAssistant(LanguageAssistant):
    def __init__(self, name, model, model_path, kwargs={}, device=None):
        super().__init__(name, model)
        self.model_path = model_path
        self.kwargs = kwargs
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # print({"": f"cuda:{device}"})
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": f"cuda:{device}"} if device is not None else "auto",
            # device_map={"": f"cuda:{3}"} if device is not None else "auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.hf_model = torch.compile(self.hf_model)



    def generate_response(self, messages, logprob=False, top_logprobs=False, temperature=None, top_p=None):
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = self.hf_model.generate(
                inputs,
                max_new_tokens=1024,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs.shape[1]
        generated_tokens = outputs.sequences[0, input_length:]  # shape: (seq_length,)
    
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if logprob:
            logits = torch.stack(outputs.scores, dim=0)  # shape: (seq_length, batch_size, vocab_size)
            logprobs = log_softmax(logits, dim=-1)       # shape: (seq_length, batch_size, vocab_size)

            logprobs = logprobs.transpose(0, 1)  # shape: (batch_size, seq_length, vocab_size)

            if len(generated_tokens.shape) == 1:
                generated_tokens = generated_tokens.unsqueeze(0)  # shape: (1, seq_length)

            generated_tokens = generated_tokens.unsqueeze(-1)

            token_logprobs = logprobs.gather(2, generated_tokens).squeeze(-1)  # shape: (batch_size, seq_length)

            tokens = []
            for token_id in generated_tokens.squeeze(-1).squeeze(0).tolist():
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                tokens.append(token_str)
            
            chat_completion_token_logprobs = []
            for token, logprob in zip(tokens, token_logprobs.squeeze(0).tolist()):
                chat_completion_token_logprobs.append(ChatCompletionTokenLogprob(token, logprob))
            
            token_str = " ".join(t for t in tokens)
            # print(token_str)
            return response, chat_completion_token_logprobs

        return response

    
    
class ClaudeAssistant(LanguageAssistant):
    def __init__(self, name, model, api_key=None, base_url=None, kwargs={}):
        super().__init__(name, model)
        # print("Logprob is not supported for Claude models, so the calculation of semantic entropty will be discreted.")
        self.api_key = api_key
        self.base_url = base_url
        self.openai_client = None
        self.kwargs = kwargs
        
        if api_key is None:
            raise ValueError("ANTHROPIC_API_KEY must be provided if using claude models")
        elif base_url is None:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key
            )
        else:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
    def generate_response(self, messages, logprob=False, top_logprobs=None, temperature=None, top_p=None):
        super().generate_response()
        
        system_prompt = messages[0]["content"]
        messages = messages[1:]
        
        params = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": self.kwargs["max_tokens"]
        }
        
        response = self.anthropic_client.messages.create(
            **params
        )
        
        if logprob:
            return response.content[0].text, None
        
        return response.content[0].text
        
        
def LoadModel(model_info, pos):
    if "llama" in model_info["model_name"].lower() or 'qwen' in model_info["model_name"].lower() \
        or 'deepseek' in model_info["model_name"].lower() or "phi" in model_info["model_name"].lower() \
            or "mixtral" in model_info["model_name"].lower():
        if model_info["api_key"] is None:
            model_info["api_key"] = os.getenv("OPENROUTER_API_KEY")
        return OpenRouterAPIAssistant(
            model_info["name"],
            model_info["model_name"],
            model_info["api_key"],
            model_info["base_url"],
            kwargs=model_info["kwargs"]
        )
    elif "gpt" in model_info["model_name"].lower() or "o1" in model_info["model_name"].lower():
        if model_info["api_key"] is None:
            model_info["api_key"] = os.getenv("OPENAI_API_KEY")
        return OpenAIAssistant(
            model_info["name"],
            model_info["model_name"],
            model_info["api_key"],
            model_info["base_url"],
            kwargs=model_info["kwargs"]
        )
    elif "claude" in model_info["model_name"].lower():
        if model_info["api_key"] is None:
            model_info["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        return ClaudeAssistant(
            model_info["name"],
            model_info["model_name"],
            model_info["api_key"],
            model_info["base_url"],
            kwargs=model_info["kwargs"]
        )
    # elif "phi" in model_info["model_name"].lower():
    #     return HuggingFaceModelAssistant(
    #         model_info["name"],
    #         model_info["model_name"],
    #         model_info["model_path"],
    #         kwargs=model_info["kwargs"],
    #         device=pos
    #     )