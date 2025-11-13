from .prompt import self_conf_elicit_prompt_ground_truth
from .config.debater import role_prompts


class Debator():
    def __init__(self, agent_name, model, agent_info, use_role=False):
        self.agent_name = agent_name
        self.model = model
        self.use_role = use_role
        self.agent_info = agent_info
        
    
    def debate(self, messages, logprob=False, top_logprobs=None, temperature=None, top_p=None):
        if logprob:
            response, token_list = self.model.generate_response(messages, logprob, top_logprobs, temperature, top_p)
            return response, messages, token_list
        
        response = self.model.generate_response(messages)
        return response, messages