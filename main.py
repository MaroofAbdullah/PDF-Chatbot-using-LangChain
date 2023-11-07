import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    st.header("PDF reader")
    st.sidebar.title("LLM chatbot using langchain")
    st.sidebar.markdown('''
    chatbot using langchain
    ''')
    pdf = st.file_uploader("upload your file here", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks[0])
        embeddings = HuggingFaceEmbeddings()
        docsearch = FAISS.from_texts(chunks, embeddings)
        st.write("embeddings created")
        query = st.text_input("ask any question to the pdf")
        if query:
            docs = docsearch.similarity_search(query=query, k=1)
            llm = GooglePalm(google_api_key=st.secrets("google_api_key"), temperature=0.1)
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == "__main__":
    main()
