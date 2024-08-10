import os
from dotenv import load_dotenv

import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import chromadb

load_dotenv()


# https://diptimanrc.medium.com/rapid-q-a-on-multiple-pdfs-using-langchain-and-chromadb-as-local-disk-vector-store-60678328c0df
def load_hardcoded_pdf(pdf_folder_path) -> Chroma:
    # pdf_folder_path = "./consent_forms_cleaned"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_store/"
    )
    vectordb.persist()
    return vectordb


def get_current_vector_db() -> Chroma:
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_store/"
    )
    return vector_store


def save_file_on_disk(bytes_data, file_path):
    with open(file_path, 'wb') as f:  # Open file in binary write mode
        f.write(bytes_data)


def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    return chain


def get_llm_response_for_hardcoded_files(query):
    vectordb = load_hardcoded_pdf("./data")
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


def get_llm_response_for_attached_files(files, query):
    if (st.session_state.files_list):
        print("Reusing existing embeddings")
        vectordb = get_current_vector_db()
    else:
        print("Calculating new embeddings")
        for file in files:
            save_file_on_disk(file.read(), "./tmp/" + file.name)
            st.session_state.files_list.append(file.name)
        vectordb = load_hardcoded_pdf("./tmp")
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.title("📝 File Q&A")

if 'files_list' not in st.session_state:
    st.session_state.files_list = []

if not st.session_state.files_list:
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
else:
    uploaded_files = st.session_state.files_list
    for file_name in st.session_state.files_list:
        st.caption("📒 file: " + file_name)

st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your files?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask something about the files", disabled=not uploaded_files):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = get_llm_response_for_attached_files(uploaded_files, prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
