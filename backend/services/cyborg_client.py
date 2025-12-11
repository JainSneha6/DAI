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


def _make_dbconfig(storage: str, connection_string: Optional[str] = None, table_name: Optional[str] = None) -> DBConfig:
    """
    Helper to construct DBConfig for supported storage backends.
    Recognized storage values: "memory", "redis", "postgres".
    For redis/postgres a connection_string is required.
    """
    storage = (storage or "memory").strip().lower()
    if storage == "memory":
        return cyborgdb.DBConfig("memory")
    if storage == "redis":
        if not connection_string:
            raise RuntimeError("Redis storage requested but no connection string supplied")
        # Example: connection_string="redis://localhost:6379/0"
        return cyborgdb.DBConfig(location="redis", connection_string=connection_string)
    if storage == "postgres":
        if not connection_string:
            raise RuntimeError("Postgres storage requested but no connection string supplied")
        # Provide table_name optionally (items/index may want different table names)
        if table_name:
            return cyborgdb.DBConfig(location="postgres", connection_string=connection_string, table_name=table_name)
        return cyborgdb.DBConfig(location="postgres", connection_string=connection_string)

    # fallback: pass through as a simple location string (library may support more)
    return cyborgdb.DBConfig(storage)


def init_cyborg_client(
    api_key: Optional[str] = None,
    index_storage: str = "postgres",
    config_storage: str = "postgres",
    items_storage: str = "postgres",
    index_connection: Optional[str] = None,
    config_connection: Optional[str] = None,
    items_connection: Optional[str] = None,
    index_table: Optional[str] = None,
    items_table: Optional[str] = None,
    config_table: Optional[str] = None,
) -> cyborgdb.Client:
    """
    Initialize and return a cyborgdb_core Client singleton (embedded).

    For Postgres/Redis provide connection strings and optional table names via the *_connection and *_table args
    or via environment variables:
      - CYBORG_INDEX_CONN, CYBORG_CONFIG_CONN, CYBORG_ITEMS_CONN
      - CYBORG_INDEX_TABLE, CYBORG_ITEMS_TABLE

    Default storage changed to 'postgres' (configurable).
    """
    global _CLIENT
    if _CLIENT:
        return _CLIENT

    api_key = api_key or os.environ.get("CYBORG_API_KEY")
    if not api_key:
        # Embedded quickstart often still wants an API key placeholder; some versions require it.
        raise RuntimeError("CYBORG_API_KEY is required in env (set to the API key or a placeholder)")

    # allow env vars to supply connection strings if not passed explicitly
    index_connection = index_connection or os.environ.get("CYBORG_INDEX_CONN")
    config_connection = config_connection or os.environ.get("CYBORG_CONFIG_CONN")
    items_connection = items_connection or os.environ.get("CYBORG_ITEMS_CONN")

    index_table = index_table or os.environ.get("CYBORG_INDEX_TABLE")
    items_table = items_table or os.environ.get("CYBORG_ITEMS_TABLE")
    config_table = config_table or os.environ.get("CYBORG_CONFIG_TABLE")

    # Build DBConfig objects using helper (will raise on missing connection for redis/postgres)
    index_loc = _make_dbconfig(index_storage, connection_string=index_connection, table_name=index_table)
    config_loc = _make_dbconfig(config_storage, connection_string=config_connection,table_name=config_table)
    items_loc = _make_dbconfig(items_storage, connection_string=items_connection, table_name=items_table)

    # Create the client
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
    Load an existing index or create a new encrypted index (persist key only after successful create).
    Recovery path will try available keys in this order:
      1) index_key (just generated during this call)
      2) persisted key file at index_key_path (if exists)
    """
    client = client or init_cyborg_client()

    # Try load first (prefer existing persisted key on disk)
    try:
        if index_key_path and os.path.exists(index_key_path):
            with open(index_key_path, "rb") as fh:
                raw = fh.read()
            if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw.strip()):
                index_key = bytes.fromhex(raw.decode().strip())
            else:
                index_key = raw
            index = client.load_index(index_name, index_key=index_key)
            logger.info("Loaded Cyborg index '%s' using persisted key at %s", index_name, index_key_path)
            return index
        else:
            # try load without explicit key (some clients may allow it) - wrap to show clear failure if not supported
            try:
                index = client.load_index(index_name)  # may raise TypeError if key mandatory
                logger.info("Loaded Cyborg index '%s' without explicit key", index_name)
                return index
            except TypeError:
                # load_index requires a key; continue to create path
                pass
    except Exception:
        logger.info("Index '%s' not found or could not be loaded; creating a new index", index_name)

    # Generate index_key (32 bytes) for creation
    index_key = None
    try:
        index_key = client.generate_key()
    except Exception:
        index_key = secrets.token_bytes(32)

    # Attempt to create index (do NOT persist the key to disk yet)
    try:
        if embedding_model:
            index = client.create_index(index_name=index_name, index_key=index_key, embedding_model=embedding_model)
        else:
            index = client.create_index(index_name=index_name, index_key=index_key)
        logger.info("Created Cyborg index '%s' (embedding_model=%s)", index_name, embedding_model)
    except Exception as create_exc:
        logger.exception("Failed to create index %s: %s", index_name, create_exc)

        # RECOVERY: try to load using a key we have (generated or persisted)
        # 1) Try the index_key we just generated (may succeed if create raced and used same key)
        if index_key is not None:
            try:
                index = client.load_index(index_name, index_key=index_key)
                logger.info("Index '%s' created by another process; loaded using generated key", index_name)
                return index
            except Exception as e:
                logger.debug("Loading with generated key failed: %s", e)

        # 2) Try persisted key file, if present
        if index_key_path and os.path.exists(index_key_path):
            try:
                with open(index_key_path, "rb") as fh:
                    raw = fh.read()
                if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw.strip()):
                    persisted_key = bytes.fromhex(raw.decode().strip())
                else:
                    persisted_key = raw
                index = client.load_index(index_name, index_key=persisted_key)
                logger.info("Index '%s' loaded using persisted key at %s", index_name, index_key_path)
                return index
            except Exception as e:
                logger.debug("Loading with persisted key failed: %s", e)

        # Nothing worked — surface a clear, actionable error to the caller
        err_msg = (
            f"Failed to create index '{index_name}': {create_exc}. "
            "Attempted recovery by loading with generated key and persisted key (if present) but both failed. "
            "Possible causes:\n"
            " - A stale local key file exists (models/cyborg_indexes/...), or\n"
            " - An index exists in another backing store with a different key, or\n"
            " - Concurrent creation raced and used a different key.\n\n"
            "Fixes:\n"
            " - If you want a fresh index, move or remove local key files (e.g. models/cyborg_indexes/embedded_index.key) and retry;\n"
            " - If you want to reuse an existing index, supply the correct key file via index_key_path;\n"
            " - Or choose a different index name to create a new index.\n"
        )
        raise RuntimeError(err_msg)

    # If create succeeded: persist key atomically (if requested)
    if index_key_path:
        try:
            os.makedirs(os.path.dirname(index_key_path) or ".", exist_ok=True)
            tmp_path = index_key_path + ".tmp"
            with open(tmp_path, "wb") as fh:
                fh.write(index_key)
            os.replace(tmp_path, index_key_path)
            meta = {"index_name": index_name, "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
            with open(index_key_path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
        except Exception:
            logger.exception("Failed to persist index key to %s", index_key_path)

    return index


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
