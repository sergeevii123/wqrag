import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
# MODEL = "llama3"

model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
print(model.invoke("tell me a joke"))