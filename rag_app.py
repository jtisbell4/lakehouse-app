from typing import Dict, List

import chainlit as cl
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from numpy import source

from chain import with_message_history

welcome_message = """
    Welcome to Taylor's Lakehouse Apps demo! Ask anything about
    documents you vectorized and stored in your Databricks Vector Search Index.
"""


def string_to_dict_list(formatted_string: str) -> List[Dict[str, str]]:
    """Convert formatted string to a list of dictionaries."""
    docs = []
    for doc_string in formatted_string.strip().split("</context>"):
        if len(doc_string) == 0:
            continue
        doc_dict = {}
        doc_dict["doc_url"] = doc_string.split("Doc URL: ")[1].split("Doc Snippet: ")[0]
        doc_dict["doc_snippet"] = doc_string.split("Doc Snippet: ")[-1]
        docs.append(doc_dict)
    return docs


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

    msg = cl.Message(content="")

    chunks = {"context": None, "answer": ""}
    async for chunk in chain.astream({"input": message.content}, config):
        if "context" in chunk.keys():
            chunks["context"] = chunk["context"]
            continue
        await msg.stream_token(chunk["answer"])
        chunks["answer"] += chunk["answer"]

    source_documents = string_to_dict_list(chunks["context"])

    text_elements = []  # type: List[cl.Text]

    def truncate(s):
        return s[:100] if len(s) > 100 else s

    if source_documents:
        for source_doc in source_documents:
            if len(source_doc) == 0:
                continue
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=truncate(source_doc["doc_snippet"]),
                    # name=(truncate(source_doc.page_content) + " ..."),
                    name=source_doc["doc_url"],
                    # url=source_doc.metadata["url"],
                )
            )
    msg.elements = text_elements
    await msg.send()
