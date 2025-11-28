from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import csv
from services.gemini_analyzer import analyze_columns_with_gemini, get_available_models

upload_bp = Blueprint('upload', __name__, url_prefix='/api/upload')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_csv_columns(filepath):
    """Extract column names from CSV file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            columns = next(reader)
            return columns
    except Exception as e:
        return None

@upload_bp.route('/files', methods=['POST'])
def upload_files():
    """Handle file uploads from frontend"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                file.save(filepath)
                uploaded_files.append({
                    'filename': filename,
                    'original_name': file.filename,
                    'size': os.path.getsize(filepath)
                })
            else:
                errors.append(f"File '{file.filename}' has unsupported format")
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded', 'details': errors}), 400
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} file(s)',
            'files': uploaded_files,
            'errors': errors if errors else None
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/analyze/<filename>', methods=['POST'])
def analyze_file(filename):
    """Analyze uploaded file columns and recommend model type"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Extract columns from CSV
        columns = extract_csv_columns(filepath)
        if not columns:
            return jsonify({'error': 'Could not extract columns from file'}), 400
        
        # Analyze with Gemini
        analysis_result = analyze_columns_with_gemini(columns)
        
        if not analysis_result['success']:
            return jsonify({'error': analysis_result['error']}), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns,
            'analysis': analysis_result['analysis'],
            'model_info': analysis_result['model_recommendations']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/models', methods=['GET'])
def get_models():
    """Get all available model types and recommendations"""
    return jsonify({
        'success': True,
        'models': get_available_models()
    }), 200