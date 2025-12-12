# services/chat_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import os
import json
import logging
from typing import List

logger = logging.getLogger(__name__)

from cyborgdb_core.integrations.langchain import CyborgVectorStore
from langchain.document_loaders import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from datetime import datetime

bp = Blueprint("chat", __name__)

def _get_vector_store():
    """
    Re-initialize or reuse: adapt to your app lifecycle (for clarity we rebuild similarly to upload_blueprint).
    """
    cyborg_api_key = current_app.config.get("CYBORG_API_KEY") or os.environ.get("CYBORG_API_KEY")
    index_name = current_app.config.get("CYBORG_INDEX_NAME", "embedded_index_v7")
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    keys_folder = os.path.join(models_dir, "cyborg_indexes")
    key_path = os.path.join(keys_folder, f"{index_name}.key")
    if not os.path.exists(key_path):
        logger.warning("Cyborg index key not found at %s; chat may be limited", key_path)
        return None

    with open(key_path, "rb") as fh:
        index_key = fh.read()

    # Build DBConfig locations (reuse your helper _make_dbconfig if importable)
    from services.cyborg_client import _make_dbconfig
    pg_uri = current_app.config.get("CYBORG_PG_URI") or os.environ.get("CYBORG_PG_URI")
    index_loc = _make_dbconfig(current_app.config.get("CYBORG_INDEX_STORAGE", "postgres"),
                               connection_string=pg_uri if current_app.config.get("CYBORG_INDEX_STORAGE", "postgres") == "postgres" else None,
                               table_name=current_app.config.get("CYBORG_INDEX_TABLE", "cyborg_index"))
    config_loc = _make_dbconfig(current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres"),
                               connection_string=pg_uri if current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres") == "postgres" else None,
                               table_name=current_app.config.get("CYBORG_CONFIG_TABLE", "cyborg_config"))
    items_loc = _make_dbconfig(current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres"),
                               connection_string=pg_uri if current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres") == "postgres" else None,
                               table_name=current_app.config.get("CYBORG_ITEMS_TABLE", "cyborg_items"))

    # Choose an embedding model - you may prefer OpenAI or SentenceTransformer
    emb = SentenceTransformerEmbeddings(model_name=current_app.config.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    try:
        vs = CyborgVectorStore(
            index_name=index_name,
            index_key=index_key,
            api_key=cyborg_api_key,
            embedding=emb,
            index_location=index_loc,
            config_location=config_loc,
            items_location=items_loc,
            metric="cosine"
        )
        return vs
    except Exception:
        logger.exception("Failed to init CyborgVectorStore for chat")
        return None

@bp.route("/api/files", methods=["GET"])
def list_uploaded_files():
    upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
    meta_dir = os.path.join(upload_folder, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    items = []
    try:
        for fname in os.listdir(meta_dir):
            if not fname.endswith(".meta.json"):
                continue
            path = os.path.join(meta_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                items.append(meta)
            except Exception:
                logger.exception("Failed to read meta file %s", path)
        # newest first
        items.sort(key=lambda i: i.get("uploaded_at", ""), reverse=True)
        return jsonify({"success": True, "files": items})
    except Exception as e:
        logger.exception("Failed to list uploaded metadata: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route("/api/chat", methods=["POST"])
def chat_with_data():
    payload = request.get_json(force=True)
    question = payload.get("question") if payload else None
    if not question:
        return jsonify({"success": False, "error": "Missing 'question' in JSON body"}), 400

    # Build vector store and retriever
    vs = _get_vector_store()
    if not vs:
        # fallback: return top-file summaries from metadata listing
        logger.warning("Vector store unavailable; returning file list as fallback")
        files_resp = list_uploaded_files().get_json()
        return jsonify({
            "success": True,
            "answer": "Vector store is not available. Here are the available files and metadata.",
            "fallback_files": files_resp.get("files", [])
        })

    retriever = vs.as_retriever(search_kwargs={"k": 6})

    # Build LLM: prefer OpenAI if key present, otherwise try local ChatOpenAI with model_name env var
    openai_key = os.environ.get("OPENAI_API_KEY") or current_app.config.get("OPENAI_API_KEY")
    llm = None
    try:
        if openai_key:
            llm = ChatOpenAI(temperature=0, model_name=current_app.config.get("LLM_MODEL", "gpt-4o") )
        else:
            # If no OpenAI key, try ChatOpenAI with local LLM or raise
            llm = ChatOpenAI(temperature=0, model_name=current_app.config.get("LLM_MODEL", "gpt-4o"))
    except Exception:
        logger.exception("Failed to init LLM; will use simple retrieval fallback")
        llm = None

    if llm:
        try:
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            resp = qa({"query": question})
            answer_text = resp.get("result") or resp.get("answer") or str(resp)
            sources = []
            for d in resp.get("source_documents", []):
                try:
                    md = getattr(d, "metadata", {}) or {}
                    sources.append({
                        "filename": md.get("filename"),
                        "category": md.get("category"),
                        "snippet": (d.page_content[:500] + "...") if len(d.page_content) > 500 else d.page_content
                    })
                except Exception:
                    continue

            return jsonify({"success": True, "answer": answer_text, "sources": sources})
        except Exception:
            logger.exception("LLM RetrievalQA failed; will fallback to raw retrieval")

    # Fallback: retrieve top docs and return their texts
    try:
        docs = retriever.get_relevant_documents(question)[:6]
        combined = "\n\n".join([d.page_content for d in docs])
        # simple best-effort "answer" by returning the retrieved context and filenames
        sources = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            sources.append({"filename": md.get("filename"), "category": md.get("category")})
        return jsonify({"success": True, "answer": combined, "sources": sources})
    except Exception as e:
        logger.exception("Retrieval fallback failed: %s", e)
        return jsonify({"success": False, "error": "Retrieval failed"}), 500
