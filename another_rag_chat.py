from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from another_rag_trial import Models

models = Models()
embeddings = models.embeddins_ollama
llm = models.model_llama

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",
)

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

def main():
    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break

        result = retrieval_chain.invoke({"input": query})
        print("Assistant: ", result["answer"], "\n\n")

if __name__ == "__main__":
    main()