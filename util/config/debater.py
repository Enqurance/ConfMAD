role_prompts = {
    "General": "You are now General Public, one of the referees in this task. You are interested in the story and looking for updates on the investigation. Please think critically by yourself and note that it's your responsibility to choose one of which is the better first.",
    "Critic": "You are now Critic, one of the referees in this task. You will check fluent writing, clear sentences, and good wording in summary writing. Your job is to question others judgment to make sure their judgment is well-considered and offer an alternative solution if two responses are at the same level.",
    "News Author": "You are News Author, one of the referees in this task. You will focus on the consistency with the original article. Please help other people to determine which response is the better one.",
    "Psychologist": "You are Psychologist, one of the referees in this task. You will study human behavior and mental processes in order to understand and explain human behavior. Please help other people to determine which response is the better one.",
    "Scientist": "You are now Scientist, one of the referees in this task. You are a professional engaged in systematic study who possesses a strong background in the scientific method, critical thinking, and problem-solving abilities. Please help other people to determine which response is the better one.",
    "Mathematician": "You are now Mathematician, one of the referees in this task. As a professional dedicated to the study of mathematical structures, theories, and their applications, you have a keen analytical mind and a profound understanding of logical reasoning and abstract concepts. You are tasked with identifying which response is the better one or equally good.",
}

debaters = {
    "Bob": {
        # "gpt-3.5-turbo"
        "model": "gpt-4o-mini",
        # "model": "qwen-2.5-72b-instruct",
        # "model": "llama-3.1-8b-instruct",
        # "model": "phi-4",
        # "model": "llama-3.1-70b-instruct",
        "agent_name": "Bob",
        "role_description": "General",
    },
    "James": {
        # "gpt-3.5-turbo"
        # "model": "gpt-4o-mini",
        # "model": "claude-3-5-haiku-20241022",
        # "model": "claude-3-5-sonnet-20241022",
        # "model": "qwen-2.5-72b-instruct",
        "model": "phi-4",
        # "model": "llama-3.1-70b-instruct",
        # "model": "mixtral-8x7b-instruct",
        "agent_name": "James",
        "role_description": "Author",
    },
    "Alice": {
        # "model": "gpt-4o",
        # "model": "gpt-3.5-turbo",
        "model": "gpt-4o-mini",
        "agent_name": "Alice",
        "role_description": "Mathematician",
    },
    "John": {
        "model": "claude-3-haiku-20240307",
        "agent_name": "John",
        "role_description": "Critic",
    }
}