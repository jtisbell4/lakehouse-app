import os
from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from literalai.helper import utc_now
from openai import AsyncOpenAI

cl.instrument_openai()
now = utc_now()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}

access_token = os.getenv("DATABRICKS_TOKEN")
server_hostname = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
catalog = "main"
schema = "taylor_isbell"

conn_string = (
    f"databricks://token:{access_token}@{server_hostname}?"
    + f"http_path={http_path}&catalog={catalog}&schema={schema}"
)
cl_data._data_layer = SQLAlchemyDataLayer(conninfo=conn_string)


@cl.on_chat_start
async def main():
    await cl.Message("Hello, send me a message!", disable_feedback=True).send()


@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful bot, you always reply in a nice and concise manner.",
                "role": "system",
            },
            {"content": message.content, "role": "user"},
        ],
        **settings,
    )
    await cl.Message(content=response.choices[0].message.content).send()


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin")
    else:
        return None


@cl.on_chat_resume
async def on_chat_resume(thread: cl_data.ThreadDict):
    await cl.Message(f"Welcome back to {thread['name']}").send()
    if "metadata" in thread:
        await cl.Message(thread["metadata"], author="metadata", language="json").send()
    if "tags" in thread:
        await cl.Message(thread["tags"], author="tags", language="json").send()
