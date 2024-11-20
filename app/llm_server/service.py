import os
import bentoml
import torch
from dotenv import load_dotenv

load_dotenv()


# Define the BentoML Service
@bentoml.service
class LlmService:
    def __init__(self):
        # Load the tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLM_MODEL"))
        self.model = AutoModelForCausalLM.from_pretrained(os.getenv("LLM_MODEL"))
        if torch.cuda.is_available():
            self.model.to("cuda")

    @bentoml.api()
    def generate(self, prompt: str) -> str:
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=os.getenv('MAX_TOKENS'))

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
