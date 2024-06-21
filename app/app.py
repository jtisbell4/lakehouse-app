from operator import itemgetter

import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig, RunnablePassthrough
from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables.history import RunnableWithMessageHistory

from vector_search import vector_search_as_retriever

LLM_PARAMS = {"temperature": 0.01, "max_tokens": 1500}


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


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")

    human_input = """Given the following context answer the question below:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Below is a transcript of the current conversation between you and the person you are assisting, followed by a question and some relevant information for answering that question. When answering the question, you should pay attention to the relevant info first and foremost. However do not forget to review the history of the conversation in case there is helpful information there too.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", human_input),
        ]
    )

    retriever = vector_search_as_retriever

    model = ChatDatabricks(
        endpoint=chat_profile,
        extra_params=LLM_PARAMS,
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: demo_ephemeral_chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    cl.user_session.set("runnable", chain_with_message_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def _truncate_content(self, s):
            s = s.replace("\n", "")
            return s[:50] + "..." if len(s) > 50 else s + "..."

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (
                    d.metadata["url"],
                    self._truncate_content(d.page_content),
                    # "blah"
                )
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                for url, content in self.sources:
                    el = cl.Text(name=url, content=content, display="inline")
                    self.msg.elements.append(el)

    async for chunk in runnable.astream(
        {"input": message.content},
        RunnableConfig(
            {"configurable": {"session_id": "unused"}},
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)],
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
