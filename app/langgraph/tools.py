import getpass
import os
import sys

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# custom 패키지 import 위해 sys.path에 Parent directory 추가
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from vectorstore.opensearch_hybrid import OpenSearchHybridSearch

# tool에 필요한 class 초기화
duckduckgo_search = DuckDuckGoSearchRun()

opensearch_password = getpass.getpass("Enter your Opensearch password: ")
opensearch = OpenSearchHybridSearch(user="admin", pw=opensearch_password)


@tool(parse_docstring=True)
def web_search(query: str):
    """A tool to use when websearch is needed. Use this tool when there isn't enough information in OpenSearch vector store Database

    Args:
        query : a query to websearch

    Returns:
        list[str]: websearch result in list of strings
    """
    res = duckduckgo_search.invoke({"query": query})
    return [res]


@tool(parse_docstring=True)
def vectorstore_search(query: str, search_type: str = "hybrid"):
    """search OpenSearch vector databsee for documents that are relevant to a given query

    Args:
        query: a query to search vector database
        search_type (str, optional): one of three similarity search method: hybrid, BM25, and Cosine similarity search Defaults to "hybrid".

    Returns:
        list[pd.DataFrmae]: list of langchain document
    """
    if search_type == "hybrid":
        result = opensearch.hybrid_search(query)
    elif search_type == "bm25":
        result = opensearch.bm25_search(query)
    elif search_type == "cosine":
        result = opensearch.cosine_similarity_search(query)

    result["text"].tolist()

    return result
