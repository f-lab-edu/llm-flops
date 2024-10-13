import os
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from blog_data import WebsiteDataCrawler
load_dotenv()

blog_loader = WebsiteDataCrawler()
print("Retrieving documents from blog website")
docs = blog_loader.get_all_docs()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

docsearch = OpenSearchVectorSearch.from_documents(
    docs, embeddings, opensearch_url="http://localhost:9200"
)