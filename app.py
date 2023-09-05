import streamlit as st
import pickle
from PyPDF2 import PdfReader, PdfWriter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

from gsheet import push_messages_to_sheet
import boto3
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an smart document reader. It is made by using the below resources:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Vedant Somani](https://www.linkedin.com/in/vedantsomani/)')

def main():
    st.header("Chat with your document 💬")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        # Check if embeddings already exist.
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                # Embeddings loaded from disk.
                VectorStore = pickle.load(f)
        else:
            # Embeddings don't exist. Create the embeddings

            pdf_reader = PdfReader(pdf)
            pdf_writer = PdfWriter()
            text = ""
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
                text += page.extract_text() 

            # Save pdf file to local disk
            new_file = open(pdf.name, "wb")
            pdf_writer.write(new_file)
            new_file.close()

            session = boto3.Session(
                aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
            )
            s3 = session.resource('s3')
            s3.meta.client.upload_file(Filename=pdf.name, Bucket=st.secrets["S3_PDFREADER_BUCKETNAME"], Key=pdf.name)

            os.remove(pdf.name)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)
            
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key = st.secrets["OPENAI_API_KEY"], model_name='gpt-3.5-turbo'),
            chain_type='stuff',
            retriever=VectorStore.as_retriever()
        )

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
 
        # React to user input
        if prompt := st.chat_input("What do you want to know from the PDF?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = qa_chain({'query' : prompt})
            print(response)
            push_messages_to_sheet("https://" + st.secrets["S3_PDFREADER_BUCKETNAME"] + ".s3.amazonaws.com/" + pdf.name, prompt, response["result"])
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response["result"])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
 
if __name__ == '__main__':
    main()