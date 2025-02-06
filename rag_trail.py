import streamlit as st
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from another_rag_trial import Models

st.title('🦜🔗 RAG Trials')

models = Models()
embeddings = models.embeddins_ollama
llm = models.model_llama

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def response_generator(llm_response):
    for word in llm_response.split():
        yield word + " "
        time.sleep(0.05)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the question based only the data provided."),
        ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question")
    ]
)

retriever = vector_store.as_retriever(kwargs={"k": 10})
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        result = retrieval_chain.invoke({"input": prompt})
        response = st.write(result["answer"])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})