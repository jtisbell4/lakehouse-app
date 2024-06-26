import os

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

WORKSPACE_URL = os.environ["DATABRICKS_HOST"]
if not WORKSPACE_URL.startswith("https://"):
    WORKSPACE_URL = "https://" + WORKSPACE_URL

VECTOR_SEARCH_ENDPOINT = os.environ["VECTOR_SEARCH_ENDPOINT"]
VECTOR_SEARCH_INDEX = os.environ["VECTOR_SEARCH_INDEX"]

CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

vs_client = VectorSearchClient(
    disable_notice=True,
    workspace_url=WORKSPACE_URL,
    personal_access_token=DATABRICKS_TOKEN,
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
