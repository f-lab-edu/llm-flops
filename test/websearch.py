import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults

import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

LLM_MODEL = ChatOllama(model=os.getenv("OLLAMA_LLM_MODEL"), temperature=0)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))


# vector_store = FAISS(
#     embedding_function=EMBEDDINGS,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )
def sub_queries(original_query):
    print('----Rewriting Query----')

    """
    본 함수는 원본 쿼리를 입력 받아, 해당 쿼리를 재구성하여 좀 더 구체적이고, 세부적인 정보를 담고 있으며 인터넷 검색에 적합한
    다른 3가지 쿼리로 반환하는 작업을 수행합니다. 반환된 재구성된 쿼리들은 원본 쿼리와 동일한 언어를 사용합니다.

    Input:
    original_query (str): 사용자로부터 받은 원본 쿼리

    Returns:
    list: 원본 쿼리를 재구성한 3가지 쿼리가 담긴 리스트
    """

    query_rewrite_template = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    The original queries will be question related to Machine Learning and Deep learning, such as LLM, Transormers, etc.
    Imagine you are a researcher who is doing a research on a given original query and list three rewritten queries that are suitable for websearch.
    Rewritten queries need to be more specific, detailed, and likely to retrieve relevant information.
    Return rewritten query in English.

    Just return the query in list form without additional explanations.

    Original query: {original_query}

    Rewritten query:"""

    query_rewrite_prompt = PromptTemplate(
        input_variables=["original_query"], template=query_rewrite_template
    )

    REWRITE_LLM_MODEL = ChatOllama(model=os.getenv("OLLAMA_LLM_MODEL"), temperature=0.3)

    query_rewriter = query_rewrite_prompt | REWRITE_LLM_MODEL

    response = query_rewriter.invoke(original_query)

    query_list = response.content.split("\n")

    return query_list


def websearch(query_list):
    print('----Searching Web----')
    tool = TavilySearchResults(
        max_results=3, include_raw_content=True, include_images=False
    )
    doc_list = list()
    for query in query_list:
        search_result = tool.invoke({"query": query})
        content_list = [res['content'] for res in search_result]
        doc_list.extend(content_list)

    doc_result = '\n'.join(doc_list)

    return doc_result


def generate_response(question, context):
    
    final_template = """Answeer the question based on given context, which is a web-search result of given question.

    Question: {question}

    Context: {context}
    Answer:"""

    model_input_prompt = PromptTemplate(
        input_variables=["question", "context"], template=final_template
    )

    chain = model_input_prompt | LLM_MODEL
    response = chain.invoke(input={"question": question, "context": context})

    return response.content


if __name__ == "__main__":
    original_query = "머신러닝 / 딥러닝 쪽에서 새로 나온 기술들은 어떤 것들이 있어?"
    rewritten_queries = sub_queries(original_query)
    print(rewritten_queries)
    context = websearch(rewritten_queries)
    final_result = generate_response(original_query, context)
    print(final_result)
