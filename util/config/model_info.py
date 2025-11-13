import os

openrouter_url = "https://openrouter.ai/api/v1"

model_info = {
    "mixtral-8x7b-instruct": {
        "model_name": "mistralai/mixtral-8x7b-instruct",
        "model_path": "mistralai/mixtral-8x7b-instruct",
        "api_key": None,
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "phi-4": {
        "model_name": "microsoft/phi-4",
        "model_path": "microsoft/phi-4",
        "api_key": None,
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "phi-3.5-mini-instruct": {
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "model_path": "microsoft/Phi-3.5-mini-instruct",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "qwen-2.5-72b-instruct": {
        "model_name": "qwen/qwen-2.5-72b-instruct",
        "api_key": None,
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "deepseek-v3": {
        "model_name": "deepseek/deepseek-chat",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "llama-3.1-8b-instruct": {
        "model_name": "meta-llama/llama-3.1-8b-instruct",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "llama-3.1-70b-instruct": {
        "model_name": "meta-llama/llama-3.1-70b-instruct",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": openrouter_url,
        "kwargs": {}  
    },
    "llama-3.3-70b-instruct": {
        "model_name": "meta-llama/llama-3.3-70b-instruct",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "llama-3.1-405b-instruct": {
        "model_name": "meta-llama/llama-3.1-405b-instruct",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": openrouter_url,
        "kwargs": {}
    },
    "o1-mini": {
        "model_name": "o1-mini",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "gpt-3.5-turbo-0125": {
        "model_name": "gpt-3.5-turbo-0125",
        "api_key": None,
        "base_url": None,
        "kwargs": {}
    },
    "claude-3-haiku-20240307": {
        "model_name": "claude-3-haiku-20240307",
        "api_key": None,
        "base_url": None,
        "kwargs": {"max_tokens": 512}
    },
    "claude-3-5-haiku-20241022": {
        "model_name": "claude-3-5-haiku-20241022",
        "api_key": None,
        "base_url": None,
        "kwargs": {"max_tokens": 512}
    },
    "claude-3-5-sonnet-20241022": {
        "model_name": "claude-3-5-sonnet-20241022",
        "api_key": None,
        "base_url": None,
        "kwargs": {"max_tokens": 512}
    },
    "claude-3-opus-20240229": {
        "model_name": "claude-3-opus-20240229",
        "api_key": None,
        "base_url": None,
        "kwargs": {"max_tokens": 512}
    }
}