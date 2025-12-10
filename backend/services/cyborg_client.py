# services/cyborg_client.py
import os
import json
import secrets
import logging
from typing import List, Dict, Optional

import cyborgdb_core as cyborgdb  # embedded package
from cyborgdb_core import DBConfig  # sometimes useful for clarity

logger = logging.getLogger(__name__)

_CLIENT: Optional[cyborgdb.Client] = None

def init_cyborg_client(api_key: Optional[str] = None,
                       index_storage: str = "memory",
                       config_storage: str = "memory",
                       items_storage: str = "memory") -> cyborgdb.Client:
    """
    Initialize and return a cyborgdb_core Client singleton (embedded).
    index_storage/config_storage/items_storage: storage backends per quickstart ("memory" is easiest).
    """
    global _CLIENT
    if _CLIENT:
        return _CLIENT

    api_key = api_key or os.environ.get("CYBORG_API_KEY")
    if not api_key:
        # Embedded quickstart often still wants an API key placeholder; some versions require it.
        raise RuntimeError("CYBORG_API_KEY is required in env (set to the API key or a placeholder)")

    index_loc = cyborgdb.DBConfig(index_storage)
    config_loc = cyborgdb.DBConfig(config_storage)
    items_loc = cyborgdb.DBConfig(items_storage)

    _CLIENT = cyborgdb.Client(
        api_key=api_key,
        index_location=index_loc,
        config_location=config_loc,
        items_location=items_loc,
    )
    return _CLIENT


def create_or_load_index(index_name: str,
                         index_key_path: Optional[str] = None,
                         embedding_model: Optional[str] = None,
                         client: Optional[cyborgdb.Client] = None):
    """
    Load an existing index or create a new encrypted index (persist key if asked).
    If embedding_model is provided, the index will auto-generate embeddings for 'contents' fields.
    """
    client = client or init_cyborg_client()

    # Try load first
    try:
        if index_key_path and os.path.exists(index_key_path):
            with open(index_key_path, "rb") as fh:
                raw = fh.read()
            # if stored as hex
            if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw.strip()):
                index_key = bytes.fromhex(raw.decode().strip())
            else:
                index_key = raw
            index = client.load_index(index_name, index_key=index_key)
        else:
            # If we don't have key persisted, try load without explicit key (client may manage keys)
            index = client.load_index(index_name)
        logger.info("Loaded Cyborg index '%s'", index_name)
        return index
    except Exception:
        logger.info("Index '%s' not found or could not be loaded; creating a new index", index_name)

    # Generate index_key (32 bytes)
    try:
        index_key = client.generate_key()
    except Exception:
        index_key = secrets.token_bytes(32)

    # Persist key if path supplied
    if index_key_path:
        try:
            os.makedirs(os.path.dirname(index_key_path) or ".", exist_ok=True)
            with open(index_key_path, "wb") as fh:
                fh.write(index_key)
            with open(index_key_path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump({"index_name": index_name, "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}, fh)
        except Exception:
            logger.exception("Failed to persist index key to %s", index_key_path)

    # Create index with embedding_model if provided
    try:
        if embedding_model:
            index = client.create_index(index_name=index_name, index_key=index_key, embedding_model=embedding_model)
        else:
            index = client.create_index(index_name=index_name, index_key=index_key)
        logger.info("Created Cyborg index '%s' (embedding_model=%s)", index_name, embedding_model)
        return index
    except Exception as e:
        logger.exception("Failed to create index %s: %s", index_name, e)
        raise


def upsert_items(index, items: List[Dict]):
    """
    Upsert items to the encrypted index. Items can contain:
      - id: str
      - contents: str (recommended if embedding_model used)
      - vector: list[float] (optional if you compute locally)
      - metadata: dict
    """
    if not items:
        return
    try:
        index.upsert(items)
    except Exception as e:
        logger.exception("Index upsert failed: %s", e)
        raise
