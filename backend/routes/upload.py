from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import csv
from services.gemini_analyzer import analyze_file_with_gemini

bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/api/upload", methods=["POST"])
def upload_files():
    # Expect multipart/form-data with file fields named "files" (multiple possible) and an optional "model_type"
    model_type = request.form.get("model_type") or None
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    results = []
    upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "backend", "uploads"))
    os.makedirs(upload_folder, exist_ok=True)

    for f in uploaded_files:
        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            results.append({"filename": filename, "success": False, "error": "Unsupported file type"})
            continue

        save_path = os.path.join(upload_folder, filename)
        f.save(save_path)

        # Run analysis on the saved CSV header columns
        analysis = analyze_file_with_gemini(save_path, model_type)
        results.append({"filename": filename, "analysis": analysis})

    return jsonify({"success": True, "files": results})