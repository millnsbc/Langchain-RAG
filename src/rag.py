"""RAG query module with conversational retrieval"""

import os
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY
)

#set api key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#initialise embeddings model
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

#load the persisted chroma vector store
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

#create retriever with mmr for diverse results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

#initialise llm
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

def contextualise_question(question: str, chat_history: list) -> str:
    """
    reformulate question based on chat history if needed
    
    args:
        question: user's question
        chat_history: list of previous messages
    
    returns:
        standalone question
    """
    if not chat_history:
        return question
    
    try:
        #prompt to reformulate question with history (from langchain docs)
        contextualise_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualise_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualise_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        chain = contextualise_q_prompt | llm
        result = chain.invoke({"input": question, "chat_history": chat_history})
        return result.content
    except Exception as e:
        logger.error(f"Error contextualising question: {e}")
        return question

def query_rag(question: str, chat_history: list = []):
    """
    query the rag system with optional chat history
    
    args:
        question: user's question
        chat_history: list of previous messages (alternating human/ai)
    
    returns:
        dict with 'answer' and 'context' (source documents)
    """
    try:
        logger.info(f"Processing query: {question[:50]}...")
        
        #contextualise question if there's history
        standalone_question = contextualise_question(question, chat_history)
        
        #retrieve relevant documents
        retrieved_docs = retriever.invoke(standalone_question)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        #format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # qa system prompt (from langchain rag patterns)
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"Context: {context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        #generate answer
        chain = qa_prompt | llm
        result = chain.invoke({"input": question, "chat_history": chat_history})
        
        logger.info("Query completed successfully")
        return {
            "answer": result.content,
            "context": retrieved_docs
        }
    
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "context": []
        }
