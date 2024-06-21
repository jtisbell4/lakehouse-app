import os

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

VECTOR_SEARCH_ENDPOINT = "dbdemos_vs_endpoint"
VECTOR_SEARCH_INDEX = "field_demos.ssc_rag_chatbot.databricks_documentation_vs_index"

CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")
WORKSPACE_URL = os.environ.get("DATABRICKS_HOST")


vs_client = VectorSearchClient(
    disable_notice=True,
    workspace_url=WORKSPACE_URL,
    service_principal_client_id=CLIENT_ID,
    service_principal_client_secret=CLIENT_SECRET,
)
vs_index = vs_client.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=VECTOR_SEARCH_INDEX,
)

vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=["id", "url", "content"],
).as_retriever(search_kwargs={"k": 3})
