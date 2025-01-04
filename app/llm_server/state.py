from typing import Annotated

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Langgraph에서 node와 edges 옮겨 다니면서 state들을 저장하는 클래스"""

    query: list[str]  # user가 입력한 query
    messages: Annotated[
        list, add_messages
    ]  # node와 edge의 message output들을 순차적으로 저장
    num_iteration: int # 현재까지의 retrieval / rewrite iteration 횟수
    is_relevant: bool # query가 topic과 연관이 있는지 여부
    final_response: Annotated[list, add_messages]
    # documents: Annotated[list[Document]]
