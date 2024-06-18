import time
from typing import List

import chainlit as cl
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from chain import model, vector_search_as_retriever, with_message_history

welcome_message = """
    Welcome to Taylor's Lakehouse Apps demo! Ask anything about
    documents you vectorized and stored in your Databricks Vector Search Index.
"""
# message_history = ChatMessageHistory()

# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     output_key="answer",
#     chat_memory=message_history,
#     return_messages=True,
# )

# chain = ConversationalRetrievalChain.from_llm(
#     model,
#     chain_type="stuff",
#     retriever=vector_search_as_retriever,
#     memory=memory,
#     return_source_documents=True,
# )


@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()

    # chain = chain | StrOutputParser()
    cl.user_session.set("chain", with_message_history)
    cl.user_session.set("config", {"configurable": {"session_id": "unused"}})


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    config = cl.user_session.get("config")

    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)

    # res = await chain.acall(message.content, callbacks=[cb])

    msg = cl.Message(content="")

    # chunks = []
    # async for chunk in chain.astream(
    #     {"input": message.content}, config=config,
    # ):
    #     chunks.append(chunk)
    #     await msg.stream_token(str(chunk))

    chunks = {"context": None, "answer": ""}
    async for chunk in chain.astream({"input": "what is dlt"}, config):
        if "context" in chunk.keys():
            chunks["context"] = chunk["context"]
            continue
        await msg.stream_token(chunk["answer"])
        chunks["answer"] += chunk["answer"]

    source_documents = chunks["context"]

    text_elements = []  # type: List[cl.Text]

    def truncate(s):
        return s[:100] if len(s) > 100 else s

    if source_documents:
        for source_doc in source_documents.split("</context>"):
            if len(source_doc)==0:
                continue
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc,
                    # name=(truncate(source_doc.page_content) + " ..."),
                    name="blah",
                    # url=source_doc.metadata["url"],
                )
            )
    msg.elements = text_elements
    await msg.send()

