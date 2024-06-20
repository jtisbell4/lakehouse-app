from typing import Dict, List

import chainlit as cl
from langchain_community.chat_models import ChatDatabricks

from chain import load_chain

LLM_PARAMS = {"temperature": 0.01, "max_tokens": 1500}

welcome_message = """
    Welcome to Taylor's Lakehouse Apps demo! Ask anything about
    documents you vectorized and stored in your Databricks Vector Search Index.
"""


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="databricks-dbrx-instruct",
            markdown_description="The underlying LLM model is **DBRX**.",
            icon="https://media.istockphoto.com/id/678720240/vector/cute-cartoon-green-t-rex.jpg?s=2048x2048&w=is&k=20&c=bLXkBw_KagXGarKxyb0MWeSQCfaOz9wZHLvemAPQriQ=",
        ),
        cl.ChatProfile(
            name="databricks-meta-llama-3-70b-instruct",
            markdown_description="The underlying LLM model is **Llama3 70B**.",
            icon="https://media.istockphoto.com/id/1201041782/photo/alpaca.jpg?s=612x612&w=0&k=20&c=aHFfLZMuyEyyiJux4OghXfdcc40Oa6L7_cE0D7zvbtY=",
        ),
    ]


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

    chat_profile = cl.user_session.get("chat_profile")

    model = ChatDatabricks(
        endpoint=chat_profile,
        extra_params=LLM_PARAMS,
    )

    with_message_history = load_chain(model)
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
