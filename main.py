"""Chainlit UI for RAG chatbot"""

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from src.rag import query_rag

@cl.on_chat_start
async def start():
    """initialise chat session"""
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