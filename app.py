from flask import Flask, render_template, request
from src.helper import download_HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()

Pinecone_API_Key = os.getenv("PINECONE_API_KEY")
Gemini_medibot = os.getenv("GEMINI_medibot")

if not Pinecone_API_Key:
    raise RuntimeError("Missing environment variable: PINECONE_API_KEY")

if not Gemini_medibot:
    raise RuntimeError("Missing environment variable: GEMINI_medibot")

embeddings = download_HuggingFaceEmbeddings()

index_name = "medical-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("Gemini_medibot")
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)