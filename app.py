import streamlit as st
import pickle
from PyPDF2 import PdfReader, PdfWriter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

import os
import threading
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, source: str, query: str, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.query = query
        self.source = source

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print("Reached here")
        push_messages_to_sheet("https://" + st.secrets["S3_PDFREADER_BUCKETNAME"] + ".s3.amazonaws.com/" + self.source, self.query, self.text)
        
def create_vector_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    pdf_writer = PdfWriter()
    text = ""
    for page in pdf_reader.pages:
        pdf_writer.add_page(page)
        text += page.extract_text() 

    # Save pdf file to local disk temporarily
    new_file = open(pdf.name, "wb")
    pdf_writer.write(new_file)
    new_file.close()

    # Upload file to S3 bucket
    session = boto3.Session(
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=pdf.name, Bucket=st.secrets["S3_PDFREADER_BUCKETNAME"], Key=pdf.name)

    # Delete the temporary file
    os.remove(pdf.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def create_group_summary(llm, index, docs, summaries):
    summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary = summary_chain.run(docs)
    summaries[index] = summary

def generate_summary(pdf, message_history, llm):
    message_history.add_user_message("Summarize the document!")

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    num_chunks = len(chunks)
    docs = [Document(page_content=t) for t in chunks]

    group_chunks_count = 5
    num_groups = int(num_chunks / group_chunks_count)
    if not num_chunks % group_chunks_count == 0:
        num_groups += 1
    threads = []
    summaries = [None] * num_groups
    
    with st.spinner('Generating summary. Takes a little bit of time. Please stay tuned .....'):
        for i in range(num_groups):
            start = group_chunks_count*i
            if (group_chunks_count*(i+1) > num_chunks):
                end = num_chunks
            else:
                end = group_chunks_count*(i+1)
            thread = threading.Thread(target=create_group_summary, args=(llm, i, docs[start:end], summaries))
            thread.start()
            threads.append(thread)
        
        for thread in threads: 
            thread.join()

        summary = "\n\n".join(summaries)

    message_history.add_ai_message(summary)

    push_messages_to_sheet("https://" + st.secrets["S3_PDFREADER_BUCKETNAME"] + ".s3.amazonaws.com/" + pdf.name, "summary-button", summary)


def main():
    st.header("Chat with your document 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        # Check if embeddings already exist.
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                # Embeddings loaded from disk.
                vector_store = pickle.load(f)
        else:
            # Create embeddings
            vector_store = create_vector_embeddings(pdf)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        # Setup memory for contextual conversation
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        # Setup LLM
        llm = ChatOpenAI(openai_api_key = st.secrets["OPENAI_API_KEY"], model_name='gpt-3.5-turbo', streaming=True, temperature=0.5)
        
        st.button("Summarize this document!", on_click=generate_summary, args=[pdf, msgs, llm])

        # Setup QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        # Setup the display
        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)
 
        # React to user input
        if prompt := st.chat_input("What do you want to know from the PDF?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)

            with st.chat_message("assistant"):
                stream_handler = StreamHandler(st.empty(), pdf.name, prompt)
                response = qa_chain.run(prompt, callbacks = [stream_handler])
 
if __name__ == '__main__':
    main()