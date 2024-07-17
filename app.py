import os
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()
google_api=os.getenv('GOOGLE_API')
import streamlit as st
st.title("Chat with a Document Using Gemma Model")
llm=ChatGoogleGenerativeAI(google_api_key="AIzaSyB8Y_IONFE3k_GYjQAUGu-1-BapJrhdLDI",model="gemini-pro-vision")

prompt=ChatPromptTemplate.from_template(
    "You are an excellent documnet reader and you will have to give the anser as according to context <context>{context} <context> question:{Question}"
)

def embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(google_api_key=google_api,model="models/embedding-001")
        st.session_state.documents=PyPDFDirectoryLoader("linalg.pdf")
        st.session_state.load_doc=st.session_state.documents.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.load_doc[:1])
        st.session_state.vectors=Chroma.from_documents(st.session_state.final_docs,st.session_state.embeddings)
input_text=st.text_input("Please enter the question")

if st.button("Documnet Embeddings"):
    embeddings()
    st.write("Object BOX is ready")
    

if input_text:
    documnet_chain=create_stuff_documents_chain(llm,prompt)
    retreiver=st.session_state.vectors.as_retriever()
    retreval_chain=create_retrieval_chain(retreiver,documnet_chain)
    
    response=retreval_chain.invoke({"input":{input_text}})
    
    st.write(response['answer'])