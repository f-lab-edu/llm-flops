import os
import pprint

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

        self.device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLM_MODEL"))
        model = AutoModelForCausalLM.from_pretrained(os.getenv("LLM_MODEL")).to(self.device)
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        if tokenizer.chat_template is None:
            raise NotFound("Tokenizer에 chat_template이 설정되지 않았습니다!")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(os.getenv("MAX_TOKENS")),
            device=self.device,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        self.model = ChatHuggingFace(llm=llm)

        self.app = build_graph(self.model)

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

        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        inputs = {
            "query": prompt
        }
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

        # # Generate text
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         **inputs,
        #         temperature=temperature,
        #         max_new_tokens=int(),
        #     )

        # # Decode the generated text
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return generated_text
