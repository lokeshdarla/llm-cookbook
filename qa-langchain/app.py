from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import SequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate

class LangChainApp:
    def __init__(self):
        load_dotenv()
        self.llm_hf = HuggingFaceHub(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.3, "max_length": 64}
        )

        st.set_page_config(page_title="Q&A Demo")
        st.header("Langchain Application")

    def run(self):
        input = st.text_input("Input", key="input")
        submit_button = st.button("Ask the question")

        if submit_button:
            response = self.get_response(input)
            self.display_response(response)

    def get_response(self, input):
        # Modify this method to use example-driven prompting
        prompt = f"Question: {input}\nAnswer:"
        return self.llm_hf.invoke(prompt)

    def display_response(self, response):
        st.subheader("Response is")
        st.write(response)


if __name__ == "__main__":
    app = LangChainApp()
    app.run()
