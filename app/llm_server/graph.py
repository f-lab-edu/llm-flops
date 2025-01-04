import logging
import os
from functools import partial
from typing import Literal

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from state import AgentState
from tools import vectorstore_search, web_search

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)

# Agent가 사용할 tools 초기화
toolset = [vectorstore_search, web_search]

MAX_ITERATION = int(os.getenv("MAX_ITERATION", 3))


# 모델 output 형식 class
class is_related(BaseModel):
    """모델 판단 결과 형식 정하는 class"""

    is_related: str = Field(
        description="one of 'yes' or 'no' string value depending on the query relevance to Machine Learning, Deep Learning or Data Science"
    )


def check_relevance(
    state: AgentState, model: BaseChatModel
) -> Literal["initiate_state_values", "query_not_related"]:
    """질문(query)가 데이터 사이언스, ML/DL 토픽과 관련이 있는지 여부를 판단합니다.

    Args:
        state(AgentState): 그래프의 현재 상태가 담긴 AgentState
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델

    Returns:
        str: User가 입력한 query가 데이터 사이언스, ML/DL과 관련이 있는지 여부에 대한 결정
            - "agent": 관련 있으면 다음 node 이름 출력
            - END: 관련 없으면 그래프 종료 (END 출력)
    """
    bentoml_logger.info("====check_relevance====")
    query = state["query"]
    model_with_tool = model.with_structured_output(is_related)

    messages = [
        SystemMessage(
            content="""Please read the query from the user and determine whether the query is related to any of the following topics: \n 
            - Machine Learning
            - Deep Learning
            - Data Science

        If the query contains keyword(s) or semantic meaning related to the given topics, grade it as relevant. \n
        Respond in JSON with `is_related` key.
        Only return a binary score 'yes' or 'no' without further explanation."""
        ),
        HumanMessage(content=f"Here is the query: {query}"),
    ]

    score = model_with_tool.invoke(messages)
    score = score.is_related
    if score == "yes":
        bentoml_logger.info("Query is relatd to ML/DL/Data Science")
        return "initiate_state_values"
    elif score == "no":
        bentoml_logger.info("Query is not related to ML/DL/Data Science")
        state['is_relevant'] = False
        return "query_not_related"

def query_not_related(state: AgentState) -> dict:
    return {"is_relevant": False}

def initiate_state_values(state: AgentState) -> dict:
    return {"num_iteration": 0, "is_relevant": True}


def agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    에이전트 상태에서 user query를 불러와 모델을 사용하여 쿼리를 처리하고 응답을 반환합니다

    Args:
        state (AgentState): 그래프의 현재 상태가 담긴 AgenticState: "query" 에 user 질문을 포함
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델

    Returns:
        return_response (dict): 처리된 쿼리에 대해 어떤 tool을 사용할지에 대한 정보가 담긴 응답 메시지를 포함하는 dictionary
    """
    bentoml_logger.info("====Agent====")
    bentoml_logger.info("Deciding which tool to use...")
    query = state["query"]
    model_with_tools = model.bind_tools(toolset)
    response = model_with_tools.invoke(query)

    return_response = {"messages": [response]}

    return return_response


def grade_document(
    state: AgentState, model: BaseChatModel
) -> Literal["generate", "increment_iteration", END]:
    """
    사용자의 쿼리에 대해 검색된 문서의 관련성을 평가하는 함수입니다.

    Args:
        state (AgentState): 현재 그래프 상태를 담고 있는 AgentState 객체입니다.
        model (BaseChatModel): 쿼리를 처리하고 응답을 생성하는 데 사용하는 모델입니다.

    Returns:
        Literal: 검색된 문서가 사용자의 쿼리와 관련이 있는 경우 'generate'로 이동합니다.
                 문서가 관련이 없고 반복 횟수가 최대 반복 횟수를 초과했으면 END로 이동되며,
                 그렇지 않으면 'increment_iteration'로 이동하여 state의 반복 횟수(num_iteration)를 증가시킵니다.
    """
    bentoml_logger.info("====grade_document====")
    print(state)
    bentoml_logger.info("Grading the document...")

    query = state["query"]
    docs = state["messages"][-1].content
    num_iteration = state["num_iteration"]

    print("num_iteration: ", num_iteration)

    messages = [
        SystemMessage(
            content="""Please assess the given documents and determine whether they are relevant to the user query. \n

            If the documents contains keyword(s) or semantic meaning related to the given topics, grade it as relevant. \n
            Respond in JSON with `is_related` key.
            Only return a binary score 'yes' or 'no' without further explanation.
            """
        ),
        HumanMessage(
            content=f"""
                    Here is the retrieved document:
                    {docs}

                    Here is the user query:
                    {query}"""
        ),
    ]

    model_with_tool = model.with_structured_output(is_related)
    score = model_with_tool.invoke(messages)

    score = score.is_related
    if score == "yes":
        bentoml_logger.info("Document is relevant")
        goto = "generate"
    elif score == "no":
        bentoml_logger.info("Document is not relevant")
        if num_iteration >= MAX_ITERATION:
            bentoml_logger.warning("Max iteration reached. Ending the conversation.")
            goto = END
        else:
            goto = "increment_iteration"

    return goto

def increment_iteration(state: AgentState) -> dict:
    """
    현재 state의 반복 횟수를 증가시키는 함수입니다.

    Args:
        state (AgentState): 현재 그래프 상태를 담고 있는 AgentState 객체입니다.

    Returns:
        dict: 상태 내의 반복 횟수('num_iteration')가 1 증가된 후의 상태를 포함하는 딕셔너리를 반환합니다.
    """
    num_iteration = state["num_iteration"]
    num_iteration += 1
    return {"num_iteration": num_iteration}
    

def rewrite(state: AgentState, model: BaseChatModel) -> dict:
    """
    사용자의 쿼리를 분석하여 더 구체적인 내용으로 재작성하는 함수입니다. 

    Args:
        state (AgentState): 현재 그래프의 상태를 담고 있는 AgentState 객체입니다.
        model (BaseChatModel): 주어진 메시지를 처리하고 응답을 생성하는 데 사용하는 모델입니다.

    Returns:
        dict: 재작성된 쿼리를 포함하는 딕셔너리를 반환합니다.
              반환된 딕셔너리의 "query" 키는 새로운 재작성된 쿼리 문자열을 가리킵니다.
    """
    bentoml_logger.info("====rewrite====")
    bentoml_logger.info("Rewriting the query...")
    query = state["query"]

    messages = [
        SystemMessage(
            content="""The tool did not find any relevant documents.
            Read the user query and figure out the underlying semantic intent and meaning.
            Please rewrite the query to be more specific."""
        ),
        HumanMessage(
            content=f"""
                    Here is the user query:
                    {query}"""
        )
    ]

    response = model.invoke(messages)
    new_query = response.content
    bentoml_logger.info(f"Rewritten query: {new_query}")
    return {"query": new_query}


def generate(state: AgentState, model: BaseChatModel) -> dict:
    """
    사용자의 쿼리에 대한 최종 응답을 생성하는 함수입니다.

    Args:
        state (AgentState): 현재 그래프의 상태를 담고 있는 AgentState 객체입니다.
        model (BaseChatModel): 주어진 메시지를 처리하고 응답을 생성하는 데 사용하는 모델입니다.

    Returns:
        dict: 처리된 쿼리에 대한 최종 응답을 포함하는 딕셔너리를 반환합니다.
              반환된 딕셔너리의 "final_response" 키는 최신 응답의 내용을 가리킵니다.
    """
    bentoml_logger.info("====generate====")
    query = state["query"]
    docs = state["messages"][-1].content if len(state['messages'])>0 else "No document found"
    is_related = state["is_relevant"]

    prompt = [
        HumanMessage(
            content=f"""
            You are an assistant for research question that user provide. You will be provided with user query and related context documnet.
            Before answering the question, check the query_related paramter first.
            If the query_related is False, notify the user that the query is not related to the topic, and guide them to ask query about ML/DL/Data Science.
            If you don't know the answer, just say that you don't know. Give detailed explanation with answer.

            query_related: {is_related}
            Query: {query}
            Context: {docs}
            """
        )
    ]

    response = model.invoke(prompt)

    return {"final_response": response}


def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    """최종 graph build하는 함수입니다

    Args:
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델입니다

    Returns:
        CompiledStateGraph: custom build한 nodes과 edges포함한 graph를 반환합니다.
    """
    workflow = StateGraph(AgentState)

    # Nodes 추가
    workflow.add_node("initiate_state_values", initiate_state_values)
    workflow.add_node("query_not_related", query_not_related)
    workflow.add_node(
        "agent", partial(agent, model=model)
    )  # partial를 사용하여 bemtoml LlmService에서 생성된 모델을 node안에서 사용할 수 있도록 합니다. 참조: https://github.com/langchain-ai/langgraph/discussions/341#discussioncomment-11148281 참조
    retrieve = ToolNode(toolset)  # ToolNode를 사용하여 tools를 사용할 수 있도록 합니다.
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("increment_iteration", increment_iteration)
    workflow.add_node("rewrite", partial(rewrite, model=model))
    workflow.add_node("generate", partial(generate, model=model))

    # Edges/conditional edges 추가
    # workflow.add_conditional_edges(START, "check_relevance")
    workflow.add_conditional_edges(START, partial(check_relevance, model=model))
    workflow.add_edge("initiate_state_values", "agent")
    workflow.add_edge("query_not_related", "generate")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges("retrieve", partial(grade_document, model=model))
    workflow.add_edge("increment_iteration", "rewrite")
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    # 생서된 graph가 잘 연결되어 있는지 확인하기 위해 graph 시각화 후 저장
    png_data = graph.get_graph().draw_mermaid_png()

    # PNG파일 저장
    with open("langgraph_diagram.png", "wb") as file:
        file.write(png_data)

    print("LangGraph diagram saved as langgraph_diagram.png")
    return graph
