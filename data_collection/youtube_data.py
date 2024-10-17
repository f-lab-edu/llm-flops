import langchain

import langchain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.document import Document
class YoutubeVideoTranscript:
    def __init__(self, video_id):
        """
        Initialize with a YouTube video ID.
        """
        self.video_id = video_id

    def fetch_transcript(self):
        """
        Fetch the transcript for the given YouTube video ID.
        """
        try:
            # Retrieve transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            # Combine transcript parts into a single text
            transcript = " ".join([entry['text'] for entry in transcript_list])
            return transcript
        except Exception as e:
            print(f"Error retrieving transcript: {e}")
            return None

    def convert_to_langchain_document(self):
        """
        Convert the fetched transcript into a LangChain document.
        """
        transcript = self.fetch_transcript()
        if transcript:
            # Create a LangChain document
            document = Document(text=transcript)
            return document
        else:
            print("Transcript could not be fetched.")
            return None

# Example usage
# video_transcript = YoutubeVideoTranscript("dQw4w9WgXcQ")
# langchain_doc = video_transcript.convert_to_langchain_document()
# if langchain_doc:
#     print(langchain_doc.text)
