import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from operator import itemgetter
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
# MODEL = "llama3"

if MODEL.startswith("gpt"):
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

loader = PyPDFLoader("data/IntroductiontoAlphas_WorldQuantBRAIN.pdf")
pages = loader.load_and_split()
# print(pages)

template = """
Answer the question basec on context below. If you can't 
answer the question, replay "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")


vector_store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vector_store.as_retriever()
# print(retriever.invoke("What is alpha?"))

# "context": itemgetter("question") | retriever - gets document with context that are close to question
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt | model | parser
)

# print("Ouput:")
# print(chain.invoke({"question": "What is alpha?"}))
questions = ["What is alpha?", "What is the platform?", "What is the data?"]
print(questions)
print("Ouput:")
print(chain.batch([{"question": q } for q in ["What is alpha?", "What is the platform?", "What is the data?"]]))
