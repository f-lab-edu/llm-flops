import os
from dotenv import load_dotenv

from langchain.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection

from data_collection.blog_data import WebsiteDataCrawler

load_dotenv()
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = os.getenv("OPENSEARCH_PORT")
OPENSEARCH_PW = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
opensearch_http_auth = ((OPENSEARCH_USER, OPENSEARCH_PW),)

# Embedding 모델 설정
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large"
)

# OpenSearch 연결 설정
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=opensearch_http_auth,
    use_ssl=True,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
)

# OpenSearchVectorSearch 초기화
openserach_url = f"https://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"
vector_store = OpenSearchVectorSearch(
    index_name="test_blog",
    embedding_function=embeddings,
    opensearch_url=openserach_url,
    http_auth=opensearch_http_auth,
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

