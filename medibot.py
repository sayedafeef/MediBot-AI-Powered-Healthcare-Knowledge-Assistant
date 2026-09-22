import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def format_docs(docs):
    """Combines retrieved documents into a single text block for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("MediBot: AI-Powered Healthcare Knowledge Assistant!")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display prior conversation history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Structured Chat Prompt
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Use the pieces of information provided in the context to answer the user's question.\n"
             "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
             "Don't provide anything out of the given context.\n"
             "Start the answer directly. No small talk please.\n\n"
             "Context:\n{context}"),
            ("human", "{input}")
        ])

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            
            llm = ChatGroq(
                model_name="openai/gpt-oss-20b",
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],
            )

            # Modern LCEL RAG Pipeline
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | custom_prompt
                | llm
                | StrOutputParser()
            )

            # Retrieve source documents separately to pass to display UI
            source_documents = retriever.invoke(prompt)

            # Run the generation chain
            result = rag_chain.invoke(prompt)
            
            # Format combined text response with source document outputs
            result_to_show = result + "\n\n**Source Docs:**\n" + str(source_documents)
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()