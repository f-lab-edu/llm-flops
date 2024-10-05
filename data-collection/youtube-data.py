from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url(
    video_url, 
    add_video_info=True,
    language=['ko'])
loader.load()
