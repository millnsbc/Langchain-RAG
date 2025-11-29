"""Chainlit UI for RAG chatbot"""

import os
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from src.rag import query_rag, vectorstore 
from src.config import CHROMA_PATH

@cl.on_chat_start
async def start():
    """initialise chat session"""
    
    # Check if the vector store is actually populated
    # vectorstore.get()['ids'] returns a list of all document IDs. 
    # If it's empty, we need to ingest.
    try:
        existing_ids = vectorstore.get()["ids"]
        if not existing_ids:
            await cl.Message(
                content="⚠️ **Vector Store is Empty!**\n\n"
                        "The database folder exists, but it contains no documents.\n"
                        "Please run `python src/ingest.py` in your terminal to populate it, then restart this app."
            ).send()
            return
    except Exception:
        # Fallback if the DB is corrupted or unreadable
        await cl.Message(
            content="⚠️ **Vector Store Error!**\n\n"
                    "Could not read the vector database.\n"
                    "Please run `python src/ingest.py` to reset and populate it."
        ).send()
        return

    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="Welcome to the Bouldering Injury Research Assistant. This system provides evidence-based answers drawn from academic literature on climbing injuries. Ask a question to get started.",
    ).send()
    
@cl.on_message
async def main(message: cl.Message):
    """handle user messages"""
    # Get chat history
    chat_history = cl.user_session.get("chat_history", [])
    
    #query rag system (wrapped to prevent blocking event loop)
    response = await cl.make_async(query_rag)(message.content, chat_history=chat_history)
    
    # send answer
    await cl.Message(content=response["answer"]).send()
    
    # send sources as separate collapsed messages
    if response["context"]:
        sources_text = "**Sources:**\n\n"
        for idx, doc in enumerate(response["context"], 1):
            source_name = doc.metadata.get("source", "Unknown").split("\\")[-1]
            excerpt = doc.page_content[:300].replace("\n", " ").strip()
            sources_text += f'{idx}. "{excerpt}...": {source_name}\n\n'
        
        await cl.Message(content=sources_text).send()
    
    # update history
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=response["answer"]))
    cl.user_session.set("chat_history", chat_history)