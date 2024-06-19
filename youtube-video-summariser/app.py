import os
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI model
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
else:
    st.error("API key not found. Please add it to your .env file.")

def get_video_metadata(video_url):
    try:
        yt = YouTube(video_url)
        metadata = {
            'Title': yt.title,
            'Description': yt.description,
            'Publish_date': yt.publish_date,
            'Views': yt.views,
            'Length': yt.length,
            'Rating': yt.rating,
            'Author': yt.author,
            'Keywords': yt.keywords,
            'Thumbnail_url': yt.thumbnail_url,
            'Video_id': yt.video_id,
            'Channel_url': yt.channel_url
        }
        return metadata
    except Exception as e:
        st.error(f"An error occurred while fetching video metadata: {e}")
        return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcripts = {f"{entry['start']} - {entry['start'] + entry['duration']}": entry['text'] for entry in transcript}
        return transcripts
    except Exception as e:
        st.error(f"An error occurred while fetching video transcript: {e}")
        return None

def generate_summary_and_takeaways(metadata, transcripts):
    with open('./prompts/transcripts_prompt.txt', 'r') as file:
        system_prompt = file.read()

    input_prompt = {
        **metadata,
        'Transcripts': transcripts
    }

    prompt = f"{system_prompt}\n\n{input_prompt}"
    try:
        response = model.generate_content([prompt,])
        return response
    except Exception as e:
        st.error(f"An error occurred while generating summary and takeaways: {e}")
        return None

# Streamlit app
st.title("YouTube Video Analyzer")

video_url = st.text_input("Enter YouTube video URL:")

if st.button("Analyze"):
    if video_url:
        st.write("Fetching video metadata...")
        metadata = get_video_metadata(video_url)
        
        if metadata:
            st.write("Fetching video transcripts...")
            transcripts = get_transcript(metadata['Video_id'])
            
            if transcripts:
                st.write("Generating summary and key takeaways...")
                response = generate_summary_and_takeaways(metadata, transcripts)
                
                if response:
                    st.markdown(response.text)
            else:
                st.error("Failed to fetch transcripts.")
        else:
            st.error("Failed to fetch metadata.")
    else:
        st.error("Please enter a valid YouTube video URL.")
