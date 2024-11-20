import bentoml

import torch

# Define the BentoML Service
@bentoml.service
class LlmService:
    def __init__(self):
        # Load the tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        if torch.cuda.is_available():
            self.model.to("cuda")

