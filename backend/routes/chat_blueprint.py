# chat_blueprint.py
import os
import json
import logging
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import math

logger = logging.getLogger(__name__)

bp = Blueprint("chat", __name__)

# Try imports
try:
    import google.generativeai as genai
except Exception:
    genai = None
    logger.error("Gemini not installed")

try:
    from cyborgdb_core.integrations.langchain import CyborgVectorStore
    from services.cyborg_client import _make_dbconfig
except Exception:
    CyborgVectorStore = None
    logger.error("Cyborg integration not available")


# -------------------------------------------------------------------
# INIT GEMINI CLIENT
# -------------------------------------------------------------------
def init_gemini():
    api_key = "AIzaSyB_cMKuBZPux9FttkqZSFEsDJjcUlyukqY"
    if not api_key:
        logger.error("Missing GEMINI_API_KEY")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        logger.exception("Failed to init Gemini: %s", e)
        return None


# -------------------------------------------------------------------
# INIT CYBORG VECTOR STORE (EXACTLY AS IN UPLOAD_BLUEPRINT)
# -------------------------------------------------------------------
def get_vector_store():
    """Reproduce the EXACT cyborg initialization logic from upload_blueprint."""

    if CyborgVectorStore is None:
        logger.error("CyborgVectorStore not importable")
        return None

    try:
        cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
        if not cyborg_api_key:
            logger.error("Missing CYBORG_API_KEY")
            return None

        cyborg_index_name = current_app.config.get(
            "CYBORG_INDEX_NAME", "embedded_index_v15"
        )

        models_dir = current_app.config.get(
            "MODELS_FOLDER", os.path.join(os.getcwd(), "models")
        )
        keys_folder = os.path.join(models_dir, "cyborg_indexes")
        key_path = os.path.join(keys_folder, f"{cyborg_index_name}.key")

        if not os.path.exists(key_path):
            logger.error("Cyborg index key missing: %s", key_path)
            return None

        with open(key_path, "rb") as f:
            index_key = f.read()

        # STORAGE SETTINGS (must match your upload blueprint)
        storage_index = current_app.config.get("CYBORG_INDEX_STORAGE", "postgres")
        storage_config = current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres")
        storage_items = current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres")

        pg_uri = current_app.config.get(
            "CYBORG_PG_URI",
            os.environ.get("CYBORG_PG_URI", "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"),
        )

        tbl_index = current_app.config.get("CYBORG_INDEX_TABLE", "cyborg_index")
        tbl_config = current_app.config.get("CYBORG_CONFIG_TABLE", "cyborg_config")
        tbl_items = current_app.config.get("CYBORG_ITEMS_TABLE", "cyborg_items")

        index_loc = _make_dbconfig(
            storage_index,
            connection_string=pg_uri if storage_index == "postgres" else None,
            table_name=tbl_index,
        )
        config_loc = _make_dbconfig(
            storage_config,
            connection_string=pg_uri if storage_config == "postgres" else None,
            table_name=tbl_config,
        )
        items_loc = _make_dbconfig(
            storage_items,
            connection_string=pg_uri if storage_items == "postgres" else None,
            table_name=tbl_items,
        )

        embedding_model = (
            current_app.config.get("CYBORG_EMBEDDING_MODEL")
            or os.environ.get("CYBORG_EMBEDDING_MODEL")
            or "all-MiniLM-L6-v2"
        )

        logger.info("Initializing CyborgVectorStore…")

        vs = CyborgVectorStore(
            index_name=cyborg_index_name,
            index_key=index_key,
            api_key=cyborg_api_key,
            embedding=embedding_model,
            index_location=index_loc,
            config_location=config_loc,
            items_location=items_loc,
            metric="cosine",
        )

        logger.info("CyborgVectorStore READY")
        return vs

    except Exception as e:
        logger.exception("Failed to build vector store: %s", e)
        return None


# -------------------------------------------------------------------
# SEMANTIC SEARCH
# -------------------------------------------------------------------
def semantic_search(vs, query, k=5):
    try:
        docs = vs.similarity_search_with_score(query, k)
        formatted = []
        for d, score in docs:
            formatted.append({
                "text": d.page_content,
                "metadata": d.metadata,
                "score": float(score)
            })
        return formatted
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return []


# -------------------------------------------------------------------
# RAG PROMPT BUILDER
# -------------------------------------------------------------------
def build_prompt(docs, query):
    blocks = ["Use ONLY the documents below. If insufficient, say 'I don't know'.\n"]
    for i, d in enumerate(docs):
        snippet = d["text"][:2800]
        blocks.append(f"--- DOC {i+1} ---\n{snippet}\n")
    blocks.append(f"\nQUESTION:\n{query}\n")
    return "\n".join(blocks)


# -------------------------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------------------------
@bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = data.get("query")
    if not query:
        return jsonify({"success": False, "error": "query required"}), 400

    # 1. vector store
    vs = get_vector_store()
    if vs is None:
        return jsonify({"success": False, "error": "Vector store not available"}), 500

    # 2. semantic search
    docs = semantic_search(vs, query, k=5)

    # 3. build prompt
    prompt = build_prompt(docs, query)

    # 4. Gemini
    model = init_gemini()
    if model is None:
        return jsonify({"success": False, "error": "Gemini unavailable"}), 500

    try:
        resp = model.generate_content(prompt)
        text = resp.text
    except Exception as e:
        logger.exception("Gemini error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

    # 5. return
    return jsonify({
        "success": True,
        "answer": text,
        "sources": docs,
    })
