from operator import itemgetter

from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

VECTOR_SEARCH_ENDPOINT = "dbdemos_vs_endpoint"
VECTOR_SEARCH_INDEX = "field_demos.ssc_rag_chatbot.databricks_documentation_vs_index"
LLM_ENDPOINT = "databricks-dbrx-instruct"

SYSTEM_PROMPT = """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}""".strip()
LLM_PARAMS = {"temperature": 0.01, "max_tokens": 1500}


# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


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


def format_context(docs):
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata["url"],
        )
        for d in docs
    ]
    return "".join(chunk_contents)


prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            SYSTEM_PROMPT,
        ),
        # If there is history, provide it.
        # Note: This chain does not compress the history, so very long converastions can overflow the context window.
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        # User's most current question
        ("user", "{question}"),
    ]
)


# Format the converastion history to fit into the prompt template above.
def format_chat_history_for_prompt(chat_messages_array):
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(
                    HumanMessage(content=chat_message["content"])
                )
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(
                    AIMessage(content=chat_message["content"])
                )
    return formatted_chat_history


query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

query_rewrite_prompt = PromptTemplate(
    template=query_rewrite_template,
    input_variables=["chat_history", "question"],
)


############
# FM for generation
############
model = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    extra_params=LLM_PARAMS,
)

# model = ChatOpenAI(streaming=True)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages")
        | RunnableLambda(format_chat_history_for_prompt),
    }
    | RunnablePassthrough()
    | {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["chat_history"]) > 0,
                query_rewrite_prompt | model | StrOutputParser(),
            ),
            itemgetter("question"),
        )
        | vector_search_as_retriever
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "User's first question",
        },
        {
            "role": "assistant",
            "content": "Assistant's reply",
        },
        {
            "role": "user",
            "content": "User's next question",
        },
    ]
}

# chain.invoke(input_example)
