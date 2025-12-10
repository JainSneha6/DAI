# server: update your blueprint file (where upload_files lives)
from flask import Blueprint, request, jsonify, current_app, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os, logging
from datetime import datetime
import csv
from services.gemini_analyzer import analyze_file_with_gemini
from services.time_series_pipeline import analyze_file_and_run_pipeline
from services.cyborg_client import init_cyborg_client, create_or_load_index, upsert_items

logger = logging.getLogger(__name__)
# add these imports near the top of your blueprint file if not already present
import json
from flask import send_from_directory, url_for

bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/api/upload", methods=["POST"])
def upload_files():
    model_type = request.form.get("model_type") or None
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    results = []
    upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
    os.makedirs(upload_folder, exist_ok=True)

    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    os.makedirs(models_dir, exist_ok=True)

    # Embedded Cyborg config (env or Flask config)
    cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
    cyborg_index_name = current_app.config.get("CYBORG_INDEX_NAME", "embedded_index")
    embedding_model = current_app.config.get("CYBORG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    cyborg_index = None
    cyborg_key_path = None
    if cyborg_api_key:
        try:
            init_cyborg_client(api_key=cyborg_api_key,
                               index_storage=current_app.config.get("CYBORG_INDEX_STORAGE", "memory"),
                               config_storage=current_app.config.get("CYBORG_CONFIG_STORAGE", "memory"),
                               items_storage=current_app.config.get("CYBORG_ITEMS_STORAGE", "memory"))

            keys_folder = os.path.join(models_dir, "cyborg_indexes")
            os.makedirs(keys_folder, exist_ok=True)
            cyborg_key_path = os.path.join(keys_folder, f"{cyborg_index_name}.key")

            cyborg_index = create_or_load_index(cyborg_index_name, index_key_path=cyborg_key_path, embedding_model=embedding_model)
        except Exception as e:
            logger.exception("Embedded Cyborg init failed: %s", e)
            cyborg_index = None

    for f in uploaded_files:
        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            results.append({"filename": filename, "success": False, "error": "Unsupported file type"})
            continue

        save_path = os.path.join(upload_folder, filename)
        f.save(save_path)

        gemini_response = analyze_file_with_gemini(save_path, model_type)
        pipeline_result = analyze_file_and_run_pipeline(save_path, gemini_response, models_dir=models_dir)

        # Save artifact URLs (same as your existing logic)
        artifact = pipeline_result.get("pipeline", {}).get("artifact") if isinstance(pipeline_result.get("pipeline"), dict) else pipeline_result.get("artifact")
        artifact_urls = {}
        if artifact:
            model_path = artifact.get("model_path")
            meta_path = artifact.get("meta_path")
            if model_path and os.path.exists(model_path):
                artifact_urls["model_url"] = url_for("upload.serve_model", filename=os.path.basename(model_path), _external=True)
            if meta_path and os.path.exists(meta_path):
                artifact_urls["meta_url"] = url_for("upload.serve_model", filename=os.path.basename(meta_path), _external=True)

        # Now upsert items returned by pipeline_result['embeddings'] (these contain 'contents')
        try:
            items = pipeline_result.get("embeddings") or []
            if cyborg_index and items:
                # add gemini hints to metadata and upsert
                items_to_upsert = []
                for it in items:
                    item_id = it.get("id")
                    contents = it.get("contents", "") or it.get("text", "")
                    metadata = it.get("metadata", {}) or {}
                    gm = gemini_response.get("analysis", {}) if isinstance(gemini_response, dict) else {}
                    if gm.get("model_type"):
                        metadata.setdefault("model_type", gm.get("model_type"))
                    if gm.get("target_column"):
                        metadata.setdefault("target_column", gm.get("target_column"))

                    item = {"id": item_id, "metadata": metadata}
                    if contents:
                        item["contents"] = contents
                    items_to_upsert.append(item)

                if items_to_upsert:
                    upsert_items(cyborg_index, items_to_upsert)
        except Exception as e:
            logger.exception("Failed to upsert to embedded Cyborg for file %s: %s", filename, e)

        results.append({
            "filename": filename,
            "analysis": gemini_response,
            "pipeline": pipeline_result,
            "artifacts": artifact_urls
        })

    return jsonify({"success": True, "files": results})



# ... existing blueprint code (upload_files, serve_model) ...

@bp.route("/api/models", methods=["GET"])
def list_saved_models():
    """
    List saved model artifacts and their metadata.
    Expects models to be stored in MODELS_FOLDER (app config) or ./models by default.
    Returns:
      {
        "success": true,
        "models": [
          {
            "meta_filename": "Prophet_20251201T123456Z.meta.json",
            "metadata": {...},
            "model_url": "https://.../models/Prophet_20251201T123456Z.pkl",
            "meta_url": "https://.../models/Prophet_20251201T123456Z.meta.json"
          },
          ...
        ]
      }
    """
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    os.makedirs(models_dir, exist_ok=True)

    items = []
    try:
        files = os.listdir(models_dir)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not list models dir: {e}"}), 500

    # find meta files
    meta_files = [f for f in files if f.endswith(".meta.json")]
    for meta_fname in meta_files:
        meta_path = os.path.join(models_dir, meta_fname)
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            # if meta cannot be read, skip it
            continue

        base = meta_fname.replace(".meta.json", "")
        # try to find matching model artifact (pkl)
        pkl_match = None
        for candidate in files:
            if candidate.startswith(base) and candidate.endswith(".pkl"):
                pkl_match = candidate
                break

        model_url = url_for("upload.serve_model", filename=pkl_match, _external=True) if pkl_match else None
        meta_url = url_for("upload.serve_model", filename=meta_fname, _external=True)

        items.append({
            "meta_filename": meta_fname,
            "metadata": meta,
            "model_url": model_url,
            "meta_url": meta_url,
        })

    # sort by created_at in metadata (if available) descending
    def _created_key(it):
        try:
            return it["metadata"].get("created_at", "")
        except Exception:
            return ""

    items.sort(key=_created_key, reverse=True)
    return jsonify({"success": True, "models": items})


# Serve saved models & metadata (downloadable)
@bp.route("/models/<path:filename>", methods=["GET"])
def serve_model(filename):
    """
    Serves files saved in the models directory. Make sure MODELS_FOLDER in app config
    points to the folder where models are saved (default: ./models).
    """
    models_dir = current_app.config.get("MODELS_FOLDER", os.path.join(os.getcwd(), "models"))
    # Security: ensure path is inside models_dir
    return send_from_directory(models_dir, filename, as_attachment=True)
