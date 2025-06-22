import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

# Azure OpenAI configuration
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
GPT_DEPLOYMENT = os.getenv("AZURE_GPT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model="text-embedding-3-small",
        openai_api_key=AZURE_API_KEY,
        openai_api_base=AZURE_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_API_VERSION
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, say "answer is not available in the context", don't provide incorrect answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(
        deployment_name=GPT_DEPLOYMENT,
        model="gpt-4o",
        openai_api_key=AZURE_API_KEY,
        openai_api_base=AZURE_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_API_VERSION,
        temperature=0.3
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model="text-embedding-3-small",
        openai_api_key=AZURE_API_KEY,
        openai_api_base=AZURE_ENDPOINT,
        openai_api_type="azure",
        openai_api_version=AZURE_API_VERSION
    )

    if not os.path.exists("faiss_index"):
        st.error("No FAISS index found. Please upload and process a PDF first.")
        return

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.warning("⚠ No relevant content found in the PDF for your query.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response.get("output_text", "⚠ No response generated."))

    except Exception as e:
        st.error(f"❌ Error during response generation: {e}")
        print("\n[ERROR] Exception Traceback:\n", str(e))

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Azure OpenAI 💁")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.warning("No text could be extracted from the PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")

if __name__ == "__main__":
    main()
