# tests/test_cyborg_insert_query.py - run inside venv in WSL
from services.cyborg_client import init_cyborg_client, create_or_load_index
import os
import secrets  # If needed for key generation, but likely not here
from datetime import datetime  # If needed

# ensure same env as your app

api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
pg_uri = "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"
cyborg_index_table = "cyborg_index"
cyborg_items_table = "cyborg_items"
cyborg_config_table = "cyborg_config"


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

index = create_or_load_index("embedded_index_v12", index_key_path="models/cyborg_indexes/embedded_index_v12.key", embedding_model=os.environ.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# semantic search by text (server will encode the query)
res = index.query(query_contents="Smartwatch Electronics", top_k=2)
print("Query results (metadata only):", res)

# Fetch full items including contents for each result
full_results = []
for hit in res:
    item_id = hit['id']
    try:
        # Use index.get(ids, include) to fetch contents (ids must be list, include specifies fields)
        full_items = index.get([item_id], ["contents"])
        if full_items and len(full_items) > 0:
            full_item = full_items[0]
            contents = full_item.get('contents', b'').decode('utf-8') if isinstance(full_item.get('contents'), bytes) else full_item.get('contents', '')
        else:
            contents = 'Item not found'
        full_results.append({
            'id': item_id,
            'distance': hit['distance'],
            'metadata': hit['metadata'],
            'contents': contents
        })
    except Exception as e:
        print(f"Failed to fetch item {item_id}: {e}")
        full_results.append({
            'id': item_id,
            'distance': hit['distance'],
            'metadata': hit['metadata'],
            'contents': 'Fetch failed'
        })

print("Full query results with contents:")
for result in full_results:
    print(f"ID: {result['id']}")
    print(f"Distance: {result['distance']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Contents (CSV row): {result['contents']}")
    print("---")