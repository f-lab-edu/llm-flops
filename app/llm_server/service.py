import os
import bentoml
import torch
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)


class LlmGennerationParams(BaseModel):
    prompt: str = Field(
        default="What is the tallest building in the world?", description="Prompt Text"
    )
    temperature: float = Field(
        default=0.1, description="LLM Sampling Temperature (0-1)"
    )


# Define the BentoML Service
@bentoml.service
class LlmService:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # API 서버에 가용 가능한 GPU를 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            bentoml_logger.info("Using CUDA for inference.")

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            bentoml_logger.info("Using MPS (Apple Silicon) for inference.")
        else:
            # 사용 가능한 GPU가 없다면 cpu로 inference 대체
            self.device = torch.device("cpu")
            bentoml_logger.info("Using CPU for inference.")

        # tokenizer/LLM 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLM_MODEL"))
        self.model = AutoModelForCausalLM.from_pretrained(os.getenv("LLM_MODEL")).to(
            self.device
        )
        # LLM모델이 open-end generation하지 않도록 padding token ID를 end-of-sentence token ID와 같게 설정
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

        bentoml_logger.info(f"""{os.getenv("LLM_MODEL")} loaded to {self.device}""")

    @bentoml.api
    def generate(self, params: LlmGennerationParams) -> str:
        """prompt를 입력받으면, __init__에서 설정한 tokenizer과 LLM 모델로 답변을 생성합니다

        Args:
            prompt (str): 사용자의 입력 prompt
            temperature (float): LLM모델 답변의 무작위성 조정 파라미터 (0~1 사이)

        Returns:
            str: LLM으로 생성된 prompt에 대한 답변
        """
        prompt = params.prompt
        temperature = params.temperature

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=int(os.getenv("MAX_TOKENS")),
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
