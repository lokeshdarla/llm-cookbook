from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class PDFChatBot:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.vector_store_path = "faiss_index"
        
    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(self.vector_store_path)
    
    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, just say, "The answer is not available in the context". Do not provide an incorrect answer.\n\n
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def process_user_input(self, user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(self.vector_store_path, embeddings,allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    
    def run(self):
        st.set_page_config(page_title="Chat with PDF")
        st.header("Chat with PDF using Gemini 💁")
        
        user_question = st.text_input("Ask a question from the PDF files")
        if user_question:
            response = self.process_user_input(user_question)
            st.write(response["output_text"])  # Displaying the response
        
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit & Process button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = self.get_pdf_text(pdf_docs)
                    text_chunks = self.get_text_chunks(raw_text)
                    self.get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    pdf_chat_bot = PDFChatBot()
    pdf_chat_bot.run()
