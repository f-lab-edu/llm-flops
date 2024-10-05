from langchain_community.document_loaders import YoutubeLoader

# 주어진 유튜브 채널들의 영상 URL과 영상 데이터 가져오기 

# 
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=uWnAVYgCd0k", 
    add_video_info=True,
    language=['ko'])

video_transcripts = loader.load()

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": f"다음 발표문에서 한국어 문법적으로 이상한 부분들을 수정해줘: {video_transcripts}"},
]
pipe = pipeline("text-generation", model="MLP-KTLim/llama-3-Korean-Bllossom-8B")
pipe(messages)