import os
from langchain_community.document_loaders import RedditPostsLoader
from dotenv import load_dotenv
load_dotenv()

# load using 'subreddit' mode
loader = RedditPostsLoader(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="extractor by u/tmdqja75",
    categories=["hot"],  # List of categories to load posts from
    mode="subreddit",
    search_queries=[
        "LocalLLaMA",
        "LLM"
    ],  # List of subreddits to load posts from
    number_posts=50,  # Default value is 10
)

documents = loader.load()
documents