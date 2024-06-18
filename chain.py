from operator import itemgetter
from typing import List

from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

VECTOR_SEARCH_ENDPOINT = "dbdemos_vs_endpoint"
VECTOR_SEARCH_INDEX = "field_demos.ssc_rag_chatbot.databricks_documentation_vs_index"
LLM_ENDPOINT = "databricks-dbrx-instruct"


STORE = {}

vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=VECTOR_SEARCH_INDEX,
)

vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=["id", "url", "content"],
).as_retriever(search_kwargs={"k": 3})


retriever: RunnableParallel = RunnableParallel(
    {
        "docs": itemgetter("input")
        # | RunnableLambda(retrieve_preprocess)
        | vector_search_as_retriever,
        "question": RunnablePassthrough(),
        "history": itemgetter("history"),
    }
)


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Doc URL: {doc.metadata['url']}\nDoc Snippet: {doc.page_content}</context>"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def load_chain(model):
    human_template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", human_template),
        ]
    )

    format = itemgetter("docs") | RunnableLambda(format_docs)
    # subchain for generating an answer once we've done retrieval
    answer = prompt | model | StrOutputParser()

    def parse_output(output: dict):
        return output["answer"]  # + format_docs(output['docs'])

    # complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the answer and retrieved docs.
    chain = (
        retriever.assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "context"])  # | RunnableLambda(parse_output)
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in STORE:
            STORE[session_id] = ChatMessageHistory()
        return STORE[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
