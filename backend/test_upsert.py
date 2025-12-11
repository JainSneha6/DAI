from services.cyborg_client import init_cyborg_client, create_or_load_index
import logging, os
from services.cyborg_client import upsert_items

logging.basicConfig(level=logging.DEBUG)

api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
pg_uri = "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"
cyborg_index_table = "cyborg_index"
cyborg_items_table = "cyborg_items"
cyborg_config_table = "cyborg_config"


try:
    init_cyborg_client(api_key=api_key,
                      index_storage="postgres",
                      config_storage="postgres",
                      items_storage="postgres",
                      index_connection=pg_uri,
                      config_connection=pg_uri,
                      items_connection=pg_uri,
                      index_table=cyborg_index_table,
                items_table=cyborg_items_table,
                config_table=cyborg_config_table)
    idx = create_or_load_index("test_index_debug_v2", index_key_path="./models/cyborg_indexes/test_index_debug_v2.key",embedding_model="all-MiniLM-L6-v2")
    print("index:", bool(idx))
    upsert_items(idx, [{"id":"smoke-1","contents":"hello world","metadata":{"source":"smoke_test"}}])
    print("upsert called")
except Exception as e:
    print("Postgres-backed init failed:", e)
    print("Attempting in-memory fallback for further testing...")
    init_cyborg_client(api_key=api_key, index_storage="memory", config_storage="memory", items_storage="memory")
    idx = create_or_load_index("test_index_debug_inmem")
    print("in-memory index ok:", bool(idx))
