import os
from dotenv import load_dotenv
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
            model_kwargs={"temperature": 0, "max_length": 64}
        )

        self.capital_template = PromptTemplate(input_variables=['country'],
                                                template="Please tell me the capital of the {country}")
        self.capital_chain = LLMChain(llm=self.llm_hf, prompt=self.capital_template, output_key="capital")

        self.famous_template = PromptTemplate(input_variables=['capital'],
                                                template="Suggest me some amazing places to visit in {capital}")
        self.famous_chain = LLMChain(llm=self.llm_hf, prompt=self.famous_template, output_key="places")

        self.chain = SequentialChain(chains=[self.capital_chain, self.famous_chain],
                                      input_variables=['country'],
                                      output_variables=['capital', "places"])

        st.set_page_config(page_title="Q&A Demo")
        st.header("Langchain Application")

    def run(self):
        input_country = st.text_input("Input", key="input")
        submit_button = st.button("Ask the question")

        if submit_button:
            response = self.get_response(input_country)
            self.display_response(response)

    def get_response(self, country):
        return self.chain.invoke(country)

    def display_response(self, response):
        st.subheader("Response")
        st.write(response)


if __name__ == "__main__":
    app = LangChainApp()
    app.run()
