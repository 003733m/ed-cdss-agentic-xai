# observability/langsmith_setup.py
import os
from langsmith import Client

NUKE_KEYS = [
    "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
    "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT",
    "LANGSMITH_BASE_URL", "LANGCHAIN_BASE_URL",
    "LANGSMITH_API_URL", "LANGCHAIN_API_URL",
    "LANGSMITH_REGION", "LANGCHAIN_REGION",
    "LANGSMITH_HOST", "LANGCHAIN_HOST",
    "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
    "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
]

US = "https://api.smith.langchain.com"

def force_langsmith_us(project_name: str, api_key: str | None = None) -> Client:
    # 1) nuke endpoint drift vars
    for k in NUKE_KEYS:
        os.environ.pop(k, None)

    # 2) force US
    os.environ["LANGSMITH_ENDPOINT"] = US
    os.environ["LANGCHAIN_ENDPOINT"] = US
    os.environ["LANGSMITH_API_URL"] = US
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name

    # 3) key
    key = api_key or os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        raise RuntimeError("LangSmith key not set. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY.")
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGCHAIN_API_KEY"] = key

    # 4) forced client
    client = Client(api_url=US, api_key=key)

    # 5) auth sanity
    _ = list(client.list_projects(limit=1))
    return client
