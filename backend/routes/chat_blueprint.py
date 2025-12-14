# services/chat_blueprint.py
from flask import Blueprint, request, jsonify, current_app
import os
import json
import logging
from typing import List
from datetime import datetime

logger = logging.getLogger(__name__)

# Prefer native cyborg client index-based retrieval (query + get)
try:
    from services.cyborg_client import init_cyborg_client, create_or_load_index, _make_dbconfig
except Exception:
    init_cyborg_client = None
    create_or_load_index = None
    _make_dbconfig = None
    logger.debug("services.cyborg_client helpers not available; will fallback to CyborgVectorStore if present")

# cyborg vector store integration (LangChain-compatible)
from cyborgdb_core.integrations.langchain import CyborgVectorStore

# --- LangChain compatibility imports ---
# Document (moved across versions)
try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        Document = None

# Embeddings: prefer HuggingFaceEmbeddings, fallback to SentenceTransformerEmbeddings
HuggingFaceEmbeddings = None
SentenceTransformerEmbeddings = None

try:
    # preferred package if langchain-huggingface installed
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb
    HuggingFaceEmbeddings = _HFEmb
except Exception:
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings as _HFEmb
        HuggingFaceEmbeddings = _HFEmb
    except Exception:
        HuggingFaceEmbeddings = None

try:
    from langchain.embeddings import SentenceTransformerEmbeddings as _STEmb
    SentenceTransformerEmbeddings = _STEmb
except Exception:
    try:
        from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as _STEmb
        SentenceTransformerEmbeddings = _STEmb
    except Exception:
        SentenceTransformerEmbeddings = None

# Gemini (Google Generative AI) integration: try langchain-google-genai first
ChatGoogleGenerativeAI = None
try:
    # package: langchain-google-genai
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogle
    ChatGoogleGenerativeAI = _ChatGoogle
except Exception:
    try:
        # older or alternate naming (rare)
        from langchain.google_genai import ChatGoogleGenerativeAI as _ChatGoogle
        ChatGoogleGenerativeAI = _ChatGoogle
    except Exception:
        ChatGoogleGenerativeAI = None
        logger.warning(
            "ChatGoogleGenerativeAI not available. Install langchain-google-genai to use Gemini models."
        )

# RetrievalQA (stable)
try:
    from langchain.chains import RetrievalQA
except Exception:
    RetrievalQA = None
# -------------------------------------------------

bp = Blueprint("chat", __name__)


def _get_vector_store():
    """
    Older path: returns LangChain CyborgVectorStore which exposes as_retriever().
    Keep this for the LLM+RetrievalQA flow.
    """
    cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
    index_name = current_app.config.get("CYBORG_INDEX_NAME", "embedded_index_v9")
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    keys_folder = os.path.join(models_dir, "cyborg_indexes")
    key_path = os.path.join(keys_folder, f"{index_name}.key")
    if not os.path.exists(key_path):
        logger.warning("Cyborg index key not found at %s; chat may be limited", key_path)
        return None

    try:
        with open(key_path, "rb") as fh:
            index_key = fh.read()
    except Exception:
        logger.exception("Failed to read cyborg index key at %s", key_path)
        return None

    # Build DBConfig locations (reuse your helper _make_dbconfig if importable)
    try:
        if _make_dbconfig is None:
            from services.cyborg_client import _make_dbconfig as _make_dbconfig_local
            _make_dbconfig = _make_dbconfig_local
    except Exception:
        logger.exception("Failed to import _make_dbconfig from services.cyborg_client")
        return None

    pg_uri = pg_uri = "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"
    index_loc = _make_dbconfig(
        current_app.config.get("CYBORG_INDEX_STORAGE", "postgres"),
        connection_string=pg_uri if current_app.config.get("CYBORG_INDEX_STORAGE", "postgres") == "postgres" else None,
        table_name=current_app.config.get("CYBORG_INDEX_TABLE", "cyborg_index"),
    )
    config_loc = _make_dbconfig(
        current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres"),
        connection_string=pg_uri if current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres") == "postgres" else None,
        table_name=current_app.config.get("CYBORG_CONFIG_TABLE", "cyborg_config"),
    )
    items_loc = _make_dbconfig(
        current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres"),
        connection_string=pg_uri if current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres") == "postgres" else None,
        table_name=current_app.config.get("CYBORG_ITEMS_TABLE", "cyborg_items"),
    )

    # Choose and instantiate an embedding model - prefer HuggingFaceEmbeddings
    emb = None
    emb_model_cfg = current_app.config.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    emb_device = current_app.config.get("EMBEDDINGS_DEVICE", "cpu")

    if HuggingFaceEmbeddings is not None:
        try:
            emb = HuggingFaceEmbeddings(
                model_name=emb_model_cfg,
                model_kwargs={"device": emb_device},
            )
            logger.info("Using HuggingFaceEmbeddings with model %s (device=%s)", emb_model_cfg, emb_device)
        except TypeError:
            try:
                emb = HuggingFaceEmbeddings(embedding_model_name=emb_model_cfg)
                logger.info("Using HuggingFaceEmbeddings (alt ctor) with model %s", emb_model_cfg)
            except Exception:
                logger.exception("HuggingFaceEmbeddings constructor failed for model %s", emb_model_cfg)
                emb = None
        except Exception:
            logger.exception("Failed to instantiate HuggingFaceEmbeddings for model %s", emb_model_cfg)
            emb = None

    # Fallback to SentenceTransformerEmbeddings
    if emb is None and SentenceTransformerEmbeddings is not None:
        try:
            try:
                emb = SentenceTransformerEmbeddings(model_name=emb_model_cfg)
            except TypeError:
                emb = SentenceTransformerEmbeddings(model=emb_model_cfg)
            logger.info("Using SentenceTransformerEmbeddings with model %s", emb_model_cfg)
        except Exception:
            logger.exception("Failed to instantiate SentenceTransformerEmbeddings for model %s", emb_model_cfg)
            emb = None

    if emb is None:
        logger.error(
            "No suitable embedding class available. "
            "Install langchain-huggingface and sentence-transformers (or a compatible LangChain version)."
        )
        return None

    try:
        vs = CyborgVectorStore(
            index_name=index_name,
            index_key=index_key,
            api_key=cyborg_api_key,
            embedding=emb,
            index_location=index_loc,
            config_location=config_loc,
            items_location=items_loc,
            metric="cosine",
        )
        return vs
    except Exception:
        logger.exception("Failed to init CyborgVectorStore for chat")
        return None


def _get_cyborg_native_index():
    """
    Try to initialize the native Cyborg index object (the one returned by create_or_load_index in
    services.cyborg_client). This index supports .query(...) and .get(...), which the test file uses.
    Returns index or None.
    """
    if create_or_load_index is None or init_cyborg_client is None:
        logger.debug("Native cyborg client helpers not available")
        return None

    cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
    index_name = current_app.config.get("CYBORG_INDEX_NAME", "embedded_index_v9")
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    keys_folder = os.path.join(models_dir, "cyborg_indexes")
    key_path = os.path.join(keys_folder, f"{index_name}.key")

    if not os.path.exists(key_path):
        logger.warning("Cyborg index key not found at %s; native index not available", key_path)
        return None

    # ensure db init called (safe to call repeatedly)
    try:
        pg_uri = "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb"
        init_cyborg_client(
            api_key=cyborg_api_key,
            index_storage=current_app.config.get("CYBORG_INDEX_STORAGE", "postgres"),
            config_storage=current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres"),
            items_storage=current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres"),
            index_connection=pg_uri,
            config_connection=pg_uri,
            items_connection=pg_uri,
            index_table=current_app.config.get("CYBORG_INDEX_TABLE", "cyborg_index"),
            items_table=current_app.config.get("CYBORG_ITEMS_TABLE", "cyborg_items"),
            config_table=current_app.config.get("CYBORG_CONFIG_TABLE", "cyborg_config"),
        )
    except Exception:
        logger.exception("init_cyborg_client failed; continuing anyway")

    emb_model_cfg = current_app.config.get("CYBORG_EMBEDDING_MODEL", os.environ.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    try:
        index = create_or_load_index(
            index_name,
            index_key_path=key_path,
            embedding_model=emb_model_cfg,
        )
        return index
    except Exception:
        logger.exception("create_or_load_index failed for %s", index_name)
        return None


@bp.route("/api/chat", methods=["POST"])
def chat_with_data():
    payload = request.get_json(force=True)
    question = payload.get("question") if payload else None
    if not question:
        return jsonify({"success": False, "error": "Missing 'question' in JSON body"}), 400

    # Try native cyborg index first (query + get)
    native_index = _get_cyborg_native_index()
    if native_index is not None:
        try:
            # query the index (server-side encoding)
            hits = native_index.query(query_contents=question, top_k=6)
            full_results = []
            for hit in hits:
                item_id = hit.get('id') or hit.get('item_id') or hit.get('pk')
                # prepare default structure
                metadata = hit.get('metadata', {}) if isinstance(hit, dict) else {}
                distance = hit.get('distance') if isinstance(hit, dict) else None
                contents = ''
                if item_id is not None:
                    try:
                        # index.get expects list of ids and list of fields to include
                        full_items = native_index.get([item_id], ["contents"]) or []
                        if len(full_items) > 0:
                            full_item = full_items[0]
                            raw = full_item.get('contents')
                            if isinstance(raw, (bytes, bytearray)):
                                try:
                                    contents = raw.decode('utf-8')
                                except Exception:
                                    # fallback: latin-1 to preserve bytes
                                    contents = raw.decode('latin-1', errors='replace')
                            else:
                                contents = raw if raw is not None else ''
                        else:
                            contents = ''
                    except Exception:
                        logger.exception("Failed to fetch item contents for %s", item_id)
                        contents = ''

                full_results.append({
                    'id': item_id,
                    'distance': distance,
                    'metadata': metadata,
                    'contents': contents,
                })

            # Compose a simple answer: concatenated contents + brief metadata summary
            combined = "\n\n".join([r.get('contents', '') for r in full_results if r.get('contents')])
            sources = [
                {
                    'id': r.get('id'),
                    'distance': r.get('distance'),
                    'filename': (r.get('metadata') or {}).get('filename')
                }
                for r in full_results
            ]

            return jsonify({"success": True, "answer": combined, "results": full_results, "sources": sources})
        except Exception:
            logger.exception("Native cyborg index query/get failed; falling back to retriever path")

    # If native path failed / not available, try to use CyborgVectorStore as a retriever
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

    # Build LLM: prefer Gemini (ChatGoogleGenerativeAI) if api key present
    gemini_key = "AIzaSyBw1L2cJtEgDvsdeDaVHZ1cfLaaouGNjvs"
    llm = None
    if ChatGoogleGenerativeAI is None:
        logger.warning("ChatGoogleGenerativeAI class not available; LLM features will be disabled/fallback to retrieval-only.")
    else:
        try:
            model_id = current_app.config.get("LLM_MODEL", current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash"))
            try:
                if gemini_key:
                    llm = ChatGoogleGenerativeAI(model=model_id, api_key=gemini_key, temperature=0)
                else:
                    llm = ChatGoogleGenerativeAI(model=model_id, temperature=0)
                logger.info("Initialized ChatGoogleGenerativeAI model=%s", model_id)
            except TypeError:
                try:
                    if gemini_key:
                        llm = ChatGoogleGenerativeAI(model_name=model_id, api_key=gemini_key, temperature=0)
                    else:
                        llm = ChatGoogleGenerativeAI(model_name=model_id, temperature=0)
                    logger.info("Initialized ChatGoogleGenerativeAI (alt ctor) model=%s", model_id)
                except Exception:
                    logger.exception("Failed to init ChatGoogleGenerativeAI with alternate ctor")
                    llm = None
        except Exception:
            logger.exception("Failed to init ChatGoogleGenerativeAI; will use retrieval-only fallback")
            llm = None

    if llm and RetrievalQA is not None:
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

    # Fallback: retrieve top docs and return their texts via retriever
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
