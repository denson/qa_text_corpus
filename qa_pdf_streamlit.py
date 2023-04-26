import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma




def main():
    st.set_page_config(page_title="QA with your PDF file", layout="wide")
    st.title("🤖! Question Answering with your PDF file")

    st.markdown("""
    Step 1: Upload a PDF file \n
    Step 2: Enter your OpenAI API key. This costs $$. You will need to set up billing info at [OpenAI](https://platform.openai.com/account). \n
    Step 3: Type your question at the bottom and click "Run" \n
    """)

    file_input = st.file_uploader("Upload your PDF file", type=["pdf"])
    openai_key = st.text_input("Enter your OpenAI API Key")
    query = st.text_area("Enter your question")
    run_button = st.button("Run!")

    chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])
    k = st.slider("Number of relevant chunks", 1, 5, 2)

    if run_button and file_input and openai_key and query:
        os.environ["OPENAI_API_KEY"] = openai_key
        temp_file = tempfile.NamedTemporaryFile(delete=False) 
        temp_file.write(file_input.getvalue())
        temp_file.close()
        result = qa(file=temp_file.name, query=query, chain_type=chain_type, k=k)
        st.write("🤖 Answer:", result["result"])
        st.write("Relevant source text:")
        st.write("--------------------------------------------------------------------\n".join(doc.page_content for doc in result["source_documents"]))

def qa(file, query, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    return result

if __name__ == "__main__":
    main()