import streamlit as st
from dotenv import load_dotenv
import pickle
import PyPDF2
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

with st.sidebar:
    st.title('untitleproj')
    add_vertical_space(5)
    st.write('under development')

def main():
    st.header("new")

    #upload pdf
    pdf = st.file_uploader("Upload Your PDF",type='pdf')

    if pdf is None:
        st.write("Please upload a PDF file")
        return
    
    #used to display the pdf det
    pdf_reader = PdfReader(pdf)
    
    text =""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    st.write(text)


if __name__ == '__main__':
    main()

