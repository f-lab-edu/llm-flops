import logging
import os
from typing import Literal
from functools import partial

import torch
from state import AgentState
from utils.env_setup import get_device
from bentoml.exceptions import NotFound
from bentoml.models import HuggingFaceModel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)

# 모델 output 형식 class
class grade(BaseModel):
    """모델 판단 결과 형식 정하는 class"""

    binary_score: str = Field(description="모델 판단 결과 (yes, no)")


def check_relevance(state: AgentState, model: ChatHuggingFace) -> Literal["format_question", END]:
    """질문(query)가 데이터 사이언스, ML/DL 토픽과 관련이 있는지 여부를 판단합니다.

    Args:
        state: 그래프의 현재 상태가 담긴 AgenticState

    Returns:
        str: User가 입력한 query가 데이터 사이언스, ML/DL과 관련이 있는지 여부에 대한 결정
            - "format_question": 관련 있으면 다음 node 이름 출력
            - END: 관련 없으면 그래프 종료 (END 출력)
    """    
    
    query = state["query"]
    model_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""Please read the question from the user and determine whether the question is related to any of the following topics: \n 
            - Machine Learning
            - Deep Learning
            - Data Science

        Here is the user question: {question} \n
        If the question contains keyword(s) or semantic meaning related to the give topics, grade it as relevant. \n
        Give a binary score 'yes' or 'no' to indicate whether the question is related to the topic.""",
        input_variables=["question"],
    )

    chain = prompt | model_with_tool
    score = chain.invoke(query)


    score = score.binary_score
    if score == "yes":
        return "format_question"
    elif score == "no":
        return END


def format_question(state: AgentState):
    """Graph place ho
    """
    return None
    

def build_graph(model: ChatHuggingFace) -> CompiledStateGraph:
    """최종 graph build하는 함수

    Returns:
        CompiledStateGraph: custom build한 nodes과 edges포함한 graph
    """
    workflow = StateGraph(AgentState)
    workflow.add_conditional_edges(START, partial(check_relevance, model=model)) # https://github.com/langchain-ai/langgraph/discussions/341#discussioncomment-11148281 참조
    workflow.add_node("format_question", format_question)
    graph = workflow.compile() 
    return graph

if __name__ == "__main__":
    import pprint
    print(check_relevance("What is the weather like in san francsco"))
    # workflow = build_graph()

    # inputs = {
    #     "query": [
    #         ("user", "What does Lilian Weng say about the types of agent memory?"),
    #     ]
    # }
    # for output in workflow.stream(inputs):
    #     for key, value in output.items():
    #         pprint.pprint(f"Output from node '{key}':")
    #         pprint.pprint("---")
    #         pprint.pprint(value, indent=2, width=80, depth=None)
    #     pprint.pprint("\n---\n")
    # res = model.invoke(
    # """Please read the question from the user and determine whether the question is related to any of the following topics: \n 
    #         - Machine Learning
    #         - Deep Learning
    #         - Data Science

    #     Here is the user question: What is the weather like in Canada? \n
    #     If the question contains keyword(s) or semantic meaning related to the give topics, grade it as relevant. \n
    #     Give a binary score 'yes' or 'no' to indicate whether the question is related to the topic.
    # """
    # )
    # print(res)