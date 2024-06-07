from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiApp:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-pro")
        self.chat = self.model.start_chat(history=[])
        self.init_session_state()
        self.setup_streamlit()

    def init_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

    def get_gemini_response(self, question):
        response = self.chat.send_message(question, stream=True)
        return response

    def setup_streamlit(self):
        st.set_page_config(page_title="Conversational Q&A", layout="wide")
        
        with st.sidebar:
            st.title("Gemini LLM Application")
            st.markdown("### Instructions")
            st.write(
                """
                1. Enter your question in the input box.
                2. Click 'Ask the question' to get a response.
                3. View the response and the chat history below.
                """
            )

        st.title("Ask Your Questions")
        self.input = st.text_input("Enter your question here:", key="input")
        self.submit = st.button("Ask the question")

        if self.submit and self.input:
            self.process_submission()

        self.display_chat_history()

    def process_submission(self):
        response = self.get_gemini_response(self.input)
        st.session_state['chat_history'].append(("You", self.input))
        st.subheader("Response:")
        response_text = ""
        for chunk in response:
            response_text += chunk.text
        st.session_state['chat_history'].append(("Bot", response_text))

    def display_chat_history(self):
        st.subheader("Chat History:")
        for role, text in st.session_state['chat_history']:
            if role == "You":
                st.markdown(f"**{role}:** {text}")
            else:
                st.markdown(f"**{role}:** {text}", unsafe_allow_html=True)

# Instantiate and run the GeminiApp
if __name__ == "__main__":
    GeminiApp()
