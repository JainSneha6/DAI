# upload_blueprint.py
# Flask blueprint for file uploads, analysis, artifact serving and optional Cyborg embedding
# Modified to wire into services.gemini_analyzer trigger/dispatcher (analyze_and_trigger / trigger_models_for_file)

from flask import Blueprint, request, jsonify, current_app, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import logging
import json
from datetime import datetime
from services.gemini_analyzer import analyze_file_with_gemini, trigger_models_for_file, analyze_and_trigger
#from services.time_series_pipeline import analyze_file_and_run_pipeline  # STRICT runner is invoked via gemini_analyzer dispatcher now
from services.cyborg_client import init_cyborg_client, create_or_load_index, upsert_items, _make_dbconfig
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from cyborgdb_core.integrations.langchain import CyborgVectorStore
from cyborgdb_core import DBConfig
from services.data_enrichment import combine_classification, summarize_csv, write_upload_metadata
import math

# Basic logger configuration (don't rely on third-party logger.configure calls here)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return filename and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/api/upload", methods=["POST"])
def upload_files():
    """Receives uploaded CSV files, runs analysis + pipeline, stores artifacts and optionally embeds
    pipeline embeddings into an embedded Cyborg index (Postgres preferred, in-memory fallback).
    Uses LangChain integration to split CSV into row-level Documents for proper RAG retrieval.
    """
    model_type = request.form.get("model_type") or None
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    results = []

    # Configure folders from app config (or defaults)
    upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
    os.makedirs(upload_folder, exist_ok=True)

    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    os.makedirs(models_dir, exist_ok=True)

    # Cyborg configuration: prefer app config or env vars; default to Postgres URI payload used in working example
    cyborg_api_key = (
        current_app.config.get("CYBORG_API_KEY")
        or os.environ.get("CYBORG_API_KEY")
        or "cyborg_de8158fd38be4b97a400d4712fa77e3d"
    )

    # Storage backend choices (allow override via config)
    cyborg_index_storage = current_app.config.get("CYBORG_INDEX_STORAGE", "postgres")
    cyborg_config_storage = current_app.config.get("CYBORG_CONFIG_STORAGE", "postgres")
    cyborg_items_storage = current_app.config.get("CYBORG_ITEMS_STORAGE", "postgres")
    cyborg_index_name = current_app.config.get("CYBORG_INDEX_NAME", "embedded_index_v12")
    embedding_model = (
        current_app.config.get("CYBORG_EMBEDDING_MODEL")
        or os.environ.get("CYBORG_EMBEDDING_MODEL")
        or "all-MiniLM-L6-v2"
    )

    # Prefer a full postgres URI (works with the example init). If you use libpq connection string, ensure
    # the cyborg client accepts it. Here we prefer the URI form which matched your working example.
    pg_uri = current_app.config.get(
        "CYBORG_PG_URI",
        os.environ.get(
            "CYBORG_PG_URI",
            "postgresql://cyborg:cyborg123@localhost:5432/cyborgdb",
        ),
    )

    cyborg_index_table = current_app.config.get("CYBORG_INDEX_TABLE", "cyborg_index")
    cyborg_items_table = current_app.config.get("CYBORG_ITEMS_TABLE", "cyborg_items")
    cyborg_config_table = current_app.config.get("CYBORG_CONFIG_TABLE", "cyborg_config")

    cyborg_index = None
    cyborg_key_path = None
    vector_store = None

    # Initialize Cyborg client (try postgres storage if configured, otherwise fall back to memory)
    if cyborg_api_key:
        try:
            # If postgres storage is requested, pass the same URI for index/config/items connection params
            if cyborg_index_storage == "postgres" or cyborg_config_storage == "postgres" or cyborg_items_storage == "postgres":
                init_cyborg_client(
                    api_key=cyborg_api_key,
                    index_storage=cyborg_index_storage,
                    config_storage=cyborg_config_storage,
                    items_storage=cyborg_items_storage,
                    index_connection=pg_uri,
                    config_connection=pg_uri,
                    items_connection=pg_uri,
                    index_table=cyborg_index_table,
                    items_table=cyborg_items_table,
                    config_table=cyborg_config_table,
                )
            else:
                # In-memory init if configured
                init_cyborg_client(
                    api_key=cyborg_api_key,
                    index_storage=cyborg_index_storage,
                    config_storage=cyborg_config_storage,
                    items_storage=cyborg_items_storage,
                )

            # ensure keys folder exists and create/load index
            keys_folder = os.path.join(models_dir, "cyborg_indexes")
            os.makedirs(keys_folder, exist_ok=True)
            cyborg_key_path = os.path.join(keys_folder, f"{cyborg_index_name}.key")

            # create_or_load_index should return a truthy index object on success
            cyborg_index = create_or_load_index(
                cyborg_index_name, index_key_path=cyborg_key_path, embedding_model=embedding_model
            )

            if cyborg_index:
                logger.info("Cyborg index initialized: %s", cyborg_index_name)
            else:
                logger.warning("Cyborg create_or_load_index returned no index. Falling back to no-embed mode.")

        except Exception as e:
            # If postgres init fails (for example connection/config issues), attempt an in-memory fallback
            logger.exception("Embedded Cyborg init failed (will attempt in-memory fallback): %s", e)
            try:
                init_cyborg_client(api_key=cyborg_api_key, index_storage="memory", config_storage="memory", items_storage="memory")
                cyborg_index = create_or_load_index(cyborg_index_name, embedding_model=embedding_model)
                # Update storage vars for fallback
                cyborg_index_storage = "memory"
                cyborg_config_storage = "memory"
                cyborg_items_storage = "memory"
                logger.info("Cyborg in-memory index created: %s", bool(cyborg_index))
            except Exception as e2:
                logger.exception("Cyborg in-memory fallback also failed: %s", e2)
                cyborg_index = None

    # Initialize LangChain CyborgVectorStore if index available (for CSV row splitting and upsert)
    if cyborg_index and cyborg_key_path and os.path.exists(cyborg_key_path):
        try:
            with open(cyborg_key_path, "rb") as f:
                index_key = f.read()

            index_loc = _make_dbconfig(
                cyborg_index_storage,
                connection_string=pg_uri if cyborg_index_storage == "postgres" else None,
                table_name=cyborg_index_table
            )
            config_loc = _make_dbconfig(
                cyborg_config_storage,
                connection_string=pg_uri if cyborg_config_storage == "postgres" else None,
                table_name=cyborg_config_table
            )
            items_loc = _make_dbconfig(
                cyborg_items_storage,
                connection_string=pg_uri if cyborg_items_storage == "postgres" else None,
                table_name=cyborg_items_table
            )

            vector_store = CyborgVectorStore(
                index_name=cyborg_index_name,
                index_key=index_key,
                api_key=cyborg_api_key,
                embedding=embedding_model,
                index_location=index_loc,
                config_location=config_loc,
                items_location=items_loc,
                metric="cosine"
            )
            logger.info("CyborgVectorStore initialized for LangChain CSV row integration")
        except Exception as e:
            logger.exception("Failed to initialize CyborgVectorStore: %s", e)
            vector_store = None

    # Process uploaded files
    for f in uploaded_files:
        filename = secure_filename(f.filename or "")
        if not filename or not allowed_file(filename):
            results.append({"filename": filename, "success": False, "error": "Unsupported or missing file type"})
            continue

        save_path = os.path.join(upload_folder, filename)
        try:
            f.save(save_path)
        except Exception as e:
            logger.exception("Failed to save uploaded file %s: %s", filename, e)
            results.append({"filename": filename, "success": False, "error": f"Failed to save file: {e}"})
            continue

        # Run analysis
        try:
            gemini_response = analyze_file_with_gemini(save_path)
        except Exception as e:
            logger.exception("Gemini analysis failed for %s: %s", save_path, e)
            gemini_response = {"error": str(e)}

        # Load CSV rows as Documents and upsert via LangChain (replaces entire file upsert)
        csv_upsert_success = False
        if vector_store:
            try:
                loader = CSVLoader(file_path=save_path)
                docs = loader.load()

                # Enrich metadata with Gemini analysis if available
                analysis = gemini_response.get("analysis", {}) if isinstance(gemini_response, dict) and gemini_response.get("success") else {}
                for doc in docs:
                    doc.metadata["filename"] = filename
                    doc.metadata["source"] = "uploaded_csv"
                    doc.metadata["upload_time"] = datetime.utcnow().isoformat()
                    doc.metadata["model_type"] = analysis.get("model_type")
                    doc.metadata["target_column"] = analysis.get("target_column")
                    doc.metadata["key_features"] = analysis.get("key_features", [])
                    if analysis.get("explanation"):
                        doc.metadata["explanation"] = analysis.get("explanation")

                vector_store.add_documents(docs)
                logger.info("Upserted %d CSV row Documents into CyborgVectorStore for %s", len(docs), filename)
                csv_upsert_success = True
            except Exception as e:
                logger.exception("Failed to load/add CSV Documents to CyborgVectorStore for %s: %s", filename, e)
        else:
            logger.debug("Skipped CSV row upsert for %s: no vector_store", filename)

        if not csv_upsert_success:
            results.append({"filename": filename, "success": False, "error": "CSV embedding failed"})
            continue  
        try:
            summary = summarize_csv(save_path)
            category = combine_classification(gemini_response, summary.get("columns", []))
            file_meta = {
                "filename": filename,
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                "category": category,
                "columns": summary.get("columns", []),
                "dtypes": summary.get("dtypes", {}),
                "row_count": summary.get("row_count"),
                "sample_rows": summary.get("sample_rows", []),
                "aggregations": summary.get("aggregations", {}),
                "trend_last_30_days": summary.get("trend_last_30_days", []),
            }

            # Persist metadata next to uploads for quick listing
            try:
                write_upload_metadata(upload_folder, filename, file_meta)
            except Exception:
                logger.exception("Could not persist upload metadata for %s", filename)

            # Add a summary Document into your vector store to make category & summary retrievable
            if vector_store:
                # create a short textual summary to embed
                text_parts = [
                    f"Filename: {filename}",
                    f"Category: {category}",
                    f"Columns: {', '.join(file_meta['columns'])}",
                    f"Row count: {file_meta.get('row_count')}"
                ]
                # include top aggregations for retrieval
                for col, agg in (file_meta.get("aggregations") or {}).items():
                    try:
                        text_parts.append(f"{col} mean={agg.get('mean')}, sum={agg.get('sum')}")
                    except Exception:
                        continue
                text_summary = "\n".join(text_parts)

                doc = Document(page_content=text_summary, metadata={
                    "filename": filename,
                    "category": category,
                    "upload_time": file_meta["uploaded_at"],
                    "type": "file_summary",
                    "columns": file_meta["columns"]
                })
                try:
                    vector_store.add_documents([doc])
                    logger.info("Inserted summary doc for %s into vector store", filename)
                except Exception:
                    logger.exception("Failed to insert summary doc for %s into vector store", filename)

        except Exception as e:
            logger.exception("Data enrichment step failed for %s: %s", filename, e)
            
        logger.debug("Gemini response: %s", gemini_response)

        # ---------------------------
        # Trigger mapped models via gemini_analyzer dispatcher
        # - Try to provide a strict `target_col_hint` to the time-series pipeline when possible.
        # - We only use exact key_columns suggested by Gemini and verify they are numeric/convertible
        #   on a small CSV sample. This avoids fuzzy heuristics while still allowing the strict
        #   pipeline to run when Gemini provided an explicit target hint.
        try:
            target_col_hint = None
            try:
                # Prefer Gemini-provided key_columns (they must be exact column names per our strict rules)
                if isinstance(gemini_response, dict) and gemini_response.get("success"):
                    analysis = gemini_response.get("analysis", {})
                    key_cols = analysis.get("key_columns", []) or []
                else:
                    key_cols = []

                if key_cols:
                    # Inspect a small sample to verify numeric convertibility (non-heuristic: exact column names only)
                    try:
                        import pandas as _pd
                        sample_df = _pd.read_csv(save_path, nrows=100)
                        for kc in key_cols:
                            if kc in sample_df.columns:
                                coerced = _pd.to_numeric(sample_df[kc], errors="coerce")
                                if coerced.notna().sum() > 0:
                                    target_col_hint = kc
                                    logger.info("Using Gemini key_column '%s' as target_col_hint for %s", kc, filename)
                                    break
                    except Exception:
                        logger.exception("Failed to sample CSV to validate key_columns for %s", filename)
            except Exception:
                logger.exception("Failed to determine target_col_hint from gemini_response")

            if not target_col_hint:
                logger.info("No explicit target_col_hint available for %s; pipeline will not run in strict mode unless Gemini provides a target.", filename)

            # Use trigger_models_for_file to map data_domain -> models and invoke implemented runners (e.g. time_series pipeline)
            trigger_report = trigger_models_for_file(save_path, gemini_response, models_dir=models_dir, target_col_hint=target_col_hint)
            pipeline_result = trigger_report
        except Exception as e:
            logger.exception("Model trigger run failed for %s: %s", save_path, e)
            pipeline_result = {"error": str(e)}

        # Save artifact URLs if created by pipeline runner(s)
        artifact = None
        if isinstance(pipeline_result, dict):
            # Legacy pipelines may expose artifact at pipeline/artifact or artifact; preserve compatibility
            artifact = pipeline_result.get("pipeline", {}).get("artifact") if isinstance(pipeline_result.get("pipeline"), dict) else pipeline_result.get("artifact")
            # If we have multiple runners, look for first artifact produced
            if not artifact and pipeline_result.get("runners"):
                for r in pipeline_result.get("runners", {}).values():
                    res = r.get("result") if isinstance(r, dict) else r
                    if not isinstance(res, dict):
                        continue
                    a = res.get("pipeline", {}).get("artifact") if isinstance(res.get("pipeline"), dict) else res.get("artifact")
                    if a:
                        artifact = a
                        break

        artifact_urls = {}
        if artifact:
            model_path = artifact.get("model_path")
            meta_path = artifact.get("meta_path")
            if model_path and os.path.exists(model_path):
                artifact_urls["model_url"] = url_for("upload.serve_model", filename=os.path.basename(model_path), _external=True)
            if meta_path and os.path.exists(meta_path):
                artifact_urls["meta_url"] = url_for("upload.serve_model", filename=os.path.basename(meta_path), _external=True)

        # Upsert embeddings returned by pipeline_result into vector_store (via LangChain) or fallback to direct index
        try:
            items = []
            if isinstance(pipeline_result, dict):
                # legacy shape
                items = pipeline_result.get("pipeline", {}).get("embeddings") or pipeline_result.get("embeddings") or []

                # collect from runners if dispatcher returned one
                if pipeline_result.get("runners"):
                    for r in pipeline_result.get("runners", {}).values():
                        res = r.get("result") if isinstance(r, dict) else r
                        if isinstance(res, dict):
                            items_from_r = res.get("pipeline", {}).get("embeddings") or res.get("embeddings") or []
                            if items_from_r:
                                items.extend(items_from_r)

            else:
                items = []

            # Defensive logging to help debug why nothing gets upserted
            logger.debug("Cyborg index object: %s (type: %s)", bool(cyborg_index), type(cyborg_index).__name__ if cyborg_index else None)
            logger.debug("Found %d embedding items from pipeline for file %s", len(items) if items else 0, filename)

            if vector_store and items:
                docs = []
                # Enrich metadata with gemini hints if available
                gm = None
                if isinstance(gemini_response, dict):
                    gm = gemini_response.get("analysis") or gemini_response
                for it in items:
                    # ensure we always have an id (but LangChain generates if missing)
                    contents = it.get("contents") or it.get("text") or ""
                    metadata = it.get("metadata") or {}

                    if isinstance(gm, dict):
                        if gm.get("model_type"):
                            metadata.setdefault("model_type", gm.get("model_type"))
                        if gm.get("target_column"):
                            metadata.setdefault("target_column", gm.get("target_column"))

                    doc = Document(page_content=contents, metadata=metadata)
                    docs.append(doc)

                logger.debug("Prepared %d Documents from pipeline items for file %s", len(docs), filename)

                if docs:
                    vector_store.add_documents(docs)
                    logger.info("Upserted %d pipeline Documents into CyborgVectorStore for %s", len(docs), filename)
            elif cyborg_index and items:
                # Fallback to direct upsert if no vector_store
                items_to_upsert = []
                for it in items:
                    item_id = it.get("id") or it.get("uid") or f"auto-{datetime.utcnow().timestamp()}"
                    contents = it.get("contents") or it.get("text") or ""
                    metadata = it.get("metadata") or {}

                    # enrich as above
                    if isinstance(gm, dict):
                        if gm.get("model_type"):
                            metadata.setdefault("model_type", gm.get("model_type"))
                        if gm.get("target_column"):
                            metadata.setdefault("target_column", gm.get("target_column"))

                    item = {"id": item_id, "metadata": metadata}
                    if contents:
                        item["contents"] = contents

                    items_to_upsert.append(item)

                if items_to_upsert:
                    try:
                        res = upsert_items(cyborg_index, items_to_upsert)
                        logger.info("Fallback upserted %d items into Cyborg index '%s' (result: %s)", len(items_to_upsert), cyborg_index_name, repr(res))
                    except Exception as e_up:
                        logger.exception("Fallback upsert_items exception for file %s: %s", filename, e_up)
            else:
                logger.debug("Skipped pipeline item upsert for %s: no store or empty items", filename)
        except Exception as e:
            logger.exception("Failed to upsert pipeline items for file %s: %s", filename, e)


        results.append({
            "filename": filename,
            "analysis": gemini_response,
            "pipeline": pipeline_result,
            "artifacts": artifact_urls,
        })

    return jsonify({"success": True, "files": results})


@bp.route("/api/models", methods=["GET"])
def list_saved_models():
    """List saved model artifacts and their metadata from MODELS_FOLDER.
    Looks for files ending with .meta.json and pairs them with .pkl artifacts.
    """
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    os.makedirs(models_dir, exist_ok=True)

    items = []
    try:
        files = os.listdir(models_dir)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not list models dir: {e}"}), 500

    meta_files = [f for f in files if f.endswith(".meta.json")]
    for meta_fname in meta_files:
        meta_path = os.path.join(models_dir, meta_fname)
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue

        base = meta_fname.replace(".meta.json", "")
        pkl_match = next((candidate for candidate in files if candidate.startswith(base) and candidate.endswith(".pkl")), None)

        model_url = url_for("upload.serve_model", filename=pkl_match, _external=True) if pkl_match else None
        meta_url = url_for("upload.serve_model", filename=meta_fname, _external=True)

        items.append({
            "meta_filename": meta_fname,
            "metadata": meta,
            "model_url": model_url,
            "meta_url": meta_url,
        })

    def _created_key(it):
        try:
            return it["metadata"].get("created_at", "")
        except Exception:
            return ""

    items.sort(key=_created_key, reverse=True)
    return jsonify({"success": True, "models": items})


@bp.route("/models/<path:filename>", methods=["GET"])
def serve_model(filename):
    """Serve saved files from the models dir. Keeps path confined to MODELS_FOLDER."""
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    return send_from_directory(models_dir, filename, as_attachment=True)


def _sanitize_for_json(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _persist_meta(meta_dir: str, filename: str, meta_obj: dict):
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, f"{filename}.meta.json")
    safe = _sanitize_for_json(meta_obj)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(safe, fh, ensure_ascii=False, indent=2)


@bp.route("/api/files", methods=["GET"])
def list_uploaded_files():
    upload_folder = current_app.config.get(
        "UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads")
    )
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
            except Exception:
                logger.exception("Failed to read meta file %s", path)
                continue

            meta = _sanitize_for_json(meta)

            # ---- ENSURE CATEGORY STRING ----
            if not meta.get("category"):
                base_filename = fname.replace(".meta.json", "")
                csv_path = os.path.join(upload_folder, base_filename)

                if os.path.exists(csv_path):
                    try:
                        gemini = analyze_file_with_gemini(csv_path)
                        if gemini.get("success"):
                            analysis = gemini.get("analysis", {})
                            domain = analysis.get("data_domain") or analysis.get("category") or "Other"
                            row_count = analysis.get("row_count") or 0

                            meta["category"] = domain
                            meta["classification"] = _sanitize_for_json(analysis)
                            meta["row_count"] = row_count
                            meta.setdefault("uploaded_at", datetime.utcnow().isoformat() + "Z")

                            _persist_meta(meta_dir, base_filename, meta)
                        else:
                            meta["category"] = "Unknown"
                    except Exception:
                        logger.exception("Gemini classification failed")
                        meta["category"] = "Unknown"
                else:
                    meta["category"] = "Unknown"
                
            items.append(_sanitize_for_json(meta))

        items.sort(key=lambda i: i.get("uploaded_at", ""), reverse=True)
        return jsonify({"success": True, "files": items})

    except Exception as e:
        logger.exception("Failed to list uploaded files")
        return jsonify({"success": False, "error": str(e)}), 500
