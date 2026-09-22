import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Groq — reliable, fast, free-tier friendly, same as medibot.py)
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

def load_llm():
    llm=ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.5,
        groq_api_key=GROQ_API_KEY,
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain (LCEL style)

CUSTOM_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Use the pieces of information provided in the context to answer user's question.\n"
     "If you dont know the answer, just say that you dont know, dont try to make up an answer.\n"
     "Dont provide anything out of the given context\n\n"
     "Context: {context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={'k':3})

llm=load_llm()

# Build the RAG chain with LCEL
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | CUSTOM_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
source_documents=retriever.invoke(user_query)
result=qa_chain.invoke(user_query)

print("RESULT: ", result)
print("SOURCE DOCUMENTS: ", source_documents)