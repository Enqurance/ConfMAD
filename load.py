import pickle
import torch
import torch.nn as nn
from util.calibration import TemperatureScaling

file_path = "/home/zijie/Experiments/Debate/checkpoints/BBH/length_norm/temperature/qwen/qwen-2.5-72b-instruct.plk"
file_path = "/home/zijie/Experiments/Debate/checkpoints/BIGGSM/length_norm/temperature/meta-llama/llama-3.1-70b-instruct.plk"

def load_model_torch(model_class, model_path):
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint["model_state_dict"]
    
    scalar = checkpoint["scalar"]
    args = checkpoint["args"]
    
    model = model_class()
    model.load_state_dict(model_state_dict)
    
    print(f"Model and scalar loaded from: {model_path}")
    print(torch.exp(model.log_temperature))
    return model, scalar, args


def load_model_pickle(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {model_path}")
    return model

if __name__ == "__main__":
    model, scalar, args = load_model_torch(TemperatureScaling, file_path)
    # model = load_model_pickle(file_path)
    # print(model)
    # print(scalar)
    # print(args)