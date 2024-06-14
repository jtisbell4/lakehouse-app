from typing import List

import chainlit as cl
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from chain import model, vector_search_as_retriever

welcome_message = """
    Welcome to Taylor's Lakehouse Apps demo! Ask anything about
    documents you vectorized and stored in your Databricks Vector Search Index.
"""


@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        model,
        chain_type="stuff",
        retriever=vector_search_as_retriever,
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    def truncate(s):
        return s[:100] if len(s) > 100 else s

    if source_documents:
        for source_doc in source_documents[::-1]:

            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.metadata["url"],
                    name=(truncate(source_doc.page_content) + " ..."),
                    # url=source_doc.metadata["url"],
                )
            )

    await cl.Message(content=answer, elements=text_elements).send()
