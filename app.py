import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit.components.v1 import html
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Load environment variables
load_dotenv()

# Initialize session state for query history and toggle state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_drag_drop' not in st.session_state:
    st.session_state.show_drag_drop = False

# Sidebar contents
def add_vertical_space(lines: int):
    for _ in range(lines):
        st.write("")

with st.sidebar:
    st.markdown("<h1>shaDocTalk</h1>", unsafe_allow_html=True)
    add_vertical_space(2)
    st.markdown("<p>This app uses OpenAI's GPT-3 to answer questions from a PDF file.</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .css-1d391kg {
            background-color: #f0f2f6;  /* Background color */
            padding: 20px;
            border-radius: 10px;
        }
        .css-1d391kg h1 {
            color: #333333;
        }
        .css-1d391kg p {
            color: #555555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Button to toggle the drag and drop section
    if st.button('Upload PDF'):
        st.session_state.show_drag_drop = not st.session_state.show_drag_drop
    
    # Display query history
    st.markdown("<h2>History</h2>", unsafe_allow_html=True)
    for entry in st.session_state.history:
        st.markdown(f"**Q:** {entry['query']}")
        st.markdown(f"**A:** {entry['response']}")
        add_vertical_space(1)
    
    add_vertical_space(15)
    st.write("Powered by OpenAI's GPT-3")
    st.write("Developed by [vaisx05](https://github.com/vaisx05)")

def main():
    st.header("I'm shaDocTalk!")
    
    # Check if the drag and drop section should be shown
    if st.session_state.show_drag_drop:
        st.markdown("<h2>Upload your PDF file</h2>", unsafe_allow_html=True)
        pdf = st.file_uploader("Drag and drop your PDF file here", type='pdf')
        
        if pdf is not None:
            with st.spinner('Processing PDF...'):
                pdf_reader = PdfReader(pdf)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)
        
                # Embeddings
                store_name = pdf.name[:-4]
                st.write(f'{store_name}')
        
                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        VectorStore = pickle.load(f)
                    st.write('Embeddings Loaded from the Disk')
                else:
                    embeddings = OpenAIEmbeddings()
                    
                    # Ensure embeddings is not empty and has valid content
                    if chunks:
                        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                        with open(f"{store_name}.pkl", "wb") as f:
                            pickle.dump(VectorStore, f)
                    else:
                        st.error("No chunks extracted from the PDF. Ensure the PDF is valid.")
                        return
    
            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
    
            if query:
                with st.spinner('Getting answer...'):
                    docs = VectorStore.similarity_search(query=query, k=3)
        
                    llm = OpenAI()
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)

                    # Save query and response to history
                    st.session_state.history.append({'query': query, 'response': response})
    else:
        st.write("Click the button in the sidebar to upload a PDF file.")

if __name__ == '__main__':
    main()
