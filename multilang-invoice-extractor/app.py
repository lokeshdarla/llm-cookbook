from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiApp:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro-vision')
        self.input_prompt = """
        You are an expert in understanding invoices. We will upload an image as an invoice and you will answer any questions based on the uploaded image.
        """
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(page_title="MultiLanguage Invoice Extractor")
        st.header("MultiLanguage Invoice Extractor")
        self.input_text = st.text_input("Input Prompt: ", key="input")
        self.uploaded_file = st.file_uploader("Choose an image of the invoice: ", type=["jpg", "jpeg", "png"])
        
        if self.uploaded_file:
            image = Image.open(self.uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

        self.submit = st.button("Submit")
        if self.submit:
            self.process_submission()

    def input_image_details(self):
        if self.uploaded_file:
            bytes_data = self.uploaded_file.getvalue()
            image_parts = [
                {
                    "mime_type": self.uploaded_file.type,
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

    def get_gemini_response(self, input_text, image_parts, prompt):
        response = self.model.generate_content([input_text, image_parts[0], prompt])
        return response.text

    def process_submission(self):
        try:
            image_data = self.input_image_details()
            response = self.get_gemini_response(self.input_text, image_data, self.input_prompt)
            st.subheader("The Response is: ")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing the submission: {e}")

# Instantiate and run the GeminiApp
if __name__ == "__main__":
    GeminiApp()
