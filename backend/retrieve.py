# tests/test_cyborg_insert_query.py - run inside venv in WSL
from services.cyborg_client import init_cyborg_client, create_or_load_index
import os, json, time

# ensure same env as your app
client = init_cyborg_client(api_key="cyborg_de8158fd38be4b97a400d4712fa77e3d")
index = create_or_load_index("embedded_index", index_key_path="models/cyborg_indexes/embedded_index.key", embedding_model=os.environ.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# semantic search by text (server will encode the query)
res = index.query(query_contents="Smartwatch Electronics", top_k=2)
print("Query results:", res)