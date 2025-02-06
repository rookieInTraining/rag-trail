import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

class Models:
    def __init__(self):
        self.embeddins_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        self.model_llama = ChatOllama(
            model="llama3.2-vision",
            temperature=0
        )

        # self.embeddins_openai = AzureOpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     dimensions=1536,
        #     azure_endpoint=os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        #     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        #     api_version=os.environ.get("AZURE_OPENAI_EMBEDDINS_API_VERSION")
        # )

        # self.model_openai = AzureChatOpenAI(
        #     azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        #     api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2
        # )