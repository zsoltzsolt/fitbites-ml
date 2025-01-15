from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from app.settings import settings
def load_vectorstore():
    vectorstore = FAISS.load_local("app/services/index", OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY), allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

retriever = load_vectorstore()
