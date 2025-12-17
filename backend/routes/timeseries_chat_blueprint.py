# chat_blueprint.py
# Enhanced chat interface with time series forecasting capabilities
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

bp = Blueprint("timeseries_chat_bp", __name__)

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

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None
    logger.error("Pandas/Numpy not available")


# -------------------------------------------------------------------
# INTENT CLASSIFICATION
# -------------------------------------------------------------------
class IntentClassifier:
    """Classify user intent for time series queries."""
    
    INTENTS = {
        'forecast': [
            r'forecast', r'predict', r'what will', r'future', r'next',
            r'upcoming', r'estimate', r'projection'
        ],
        'analyze': [
            r'analyze', r'analysis', r'tell me about', r'explain',
            r'describe', r'what is', r'show me', r'summary'
        ],
        'compare': [
            r'compare', r'difference', r'versus', r'vs', r'better',
            r'which model', r'accuracy'
        ],
        'model_info': [
            r'model', r'trained', r'parameters', r'performance',
            r'metrics', r'how does', r'what model'
        ],
        'data_info': [
            r'data', r'columns', r'features', r'target', r'dataset',
            r'uploaded', r'files', r'what data'
        ]
    }
    
    @classmethod
    def classify(cls, query):
        """Return primary intent and confidence."""
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in cls.INTENTS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return 'general', 0.0
        
        primary = max(scores.items(), key=lambda x: x[1])
        return primary[0], primary[1] / len(cls.INTENTS[primary[0]])


# -------------------------------------------------------------------
# ENTITY EXTRACTION
# -------------------------------------------------------------------
class EntityExtractor:
    """Extract relevant entities from user queries."""
    
    @staticmethod
    def extract_time_steps(query):
        """Extract forecast horizon from query."""
        # Pattern: "next 10 days", "5 weeks", "30 steps"
        patterns = [
            r'next\s+(\d+)\s*(day|week|month|year|step|period)s?',
            r'(\d+)\s*(day|week|month|year|step|period)s?\s+ahead',
            r'for\s+(\d+)\s*(day|week|month|year|step|period)s?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                
                # Convert to steps (default to days)
                multipliers = {
                    'day': 1, 'week': 7, 'month': 30,
                    'year': 365, 'step': 1, 'period': 1
                }
                return num * multipliers.get(unit, 1)
        
        return None
    
    @staticmethod
    def extract_model_name(query, available_models):
        """Find if user mentions specific model."""
        query_lower = query.lower()
        for model in available_models:
            if model.lower() in query_lower:
                return model
        return None
    
    @staticmethod
    def extract_target_column(query, available_columns):
        """Find if user mentions specific column."""
        query_lower = query.lower()
        for col in available_columns:
            if col.lower() in query_lower:
                return col
        return None


# -------------------------------------------------------------------
# MODEL MANAGER
# -------------------------------------------------------------------
class ModelManager:
    """Manage loaded models and their metadata."""
    
    @staticmethod
    def list_models(models_dir):
        """List all available models with metadata."""
        models = []
        try:
            files = os.listdir(models_dir)
            meta_files = [f for f in files if f.endswith('.meta.json')]
            
            for meta_file in meta_files:
                meta_path = os.path.join(models_dir, meta_file)
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    base = meta_file.replace('.meta.json', '')
                    pkl_file = f"{base}.pkl"
                    
                    if pkl_file in files:
                        models.append({
                            'name': meta.get('model_name'),
                            'pkl_file': pkl_file,
                            'meta_file': meta_file,
                            'metadata': meta
                        })
                except Exception as e:
                    logger.warning(f"Failed to load metadata {meta_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return models
    
    @staticmethod
    def load_model(models_dir, pkl_file):
        """Load a specific model."""
        path = os.path.join(models_dir, pkl_file)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_latest_model(models_dir):
        """Get the most recently created model."""
        models = ModelManager.list_models(models_dir)
        if not models:
            return None
        
        models.sort(
            key=lambda m: m['metadata'].get('created_at', ''),
            reverse=True
        )
        return models[0]


# -------------------------------------------------------------------
# FORECAST GENERATOR
# -------------------------------------------------------------------
class ForecastGenerator:
    """Generate forecasts from trained models."""
    
    @staticmethod
    def _get_latest_date(meta, date_column='date'):
        """Get the actual latest date from data."""
        latest_str = meta.get('data_end')
        print(f"Latest date from metadata: {latest_str}")
        
        try:
            return pd.to_datetime(latest_str)
        except Exception as e:
            logger.warning(f"Failed to parse latest_date from metadata: {e}")
            return pd.to_datetime(meta.get('train_end'))
    
    @staticmethod
    def generate_forecast(model_info, steps=10):
        """Generate forecast using loaded model."""
        try:
            models_dir = current_app.config.get('MODELS_FOLDER', 'models')
            model = ModelManager.load_model(models_dir, model_info['pkl_file'])
            meta = model_info['metadata']
            
            model_name = meta.get('model_name')
            exogs = meta.get('exogenous_features', [])
            last_exog_values = meta.get('last_exog_values')
            inferred_freq = meta.get('inferred_freq', 'D')
            
            # Get actual latest date from CSV or fallback to train_end
            latest_date = ForecastGenerator._get_latest_date(meta)
            
            # Generate future dates starting from the day after latest_date
            future_dates = pd.date_range(
                latest_date + pd.tseries.frequencies.to_offset(inferred_freq),
                periods=steps,
                freq=inferred_freq
            )
            
            if model_name == 'SARIMAX_ARIMA':
                # Build exogenous features for forecast with dates as index
                if exogs and last_exog_values:
                    future_exog = pd.DataFrame(
                        data=[last_exog_values] * steps,
                        index=future_dates,
                        columns=exogs
                    )
                else:
                    future_exog = None
                
                forecast_obj = model.get_forecast(
                    steps=steps,
                    exog=future_exog
                )
                forecast = forecast_obj.predicted_mean.tolist()
                conf_int = forecast_obj.conf_int()
                lower_bound = conf_int.iloc[:, 0].tolist()
                upper_bound = conf_int.iloc[:, 1].tolist()
                
                return {
                    'success': True,
                    'forecast': forecast,
                    'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'steps': steps,
                    'model': model_name,
                    'target': meta.get('target_column')
                }
            
            elif model_name == 'Prophet':
                # Prophet forecasting
                future = pd.DataFrame({'ds': future_dates})
                
                if exogs and last_exog_values:
                    for idx, col in enumerate(exogs):
                        future[col] = last_exog_values[idx]
                
                forecast_df = model.predict(future)
                
                return {
                    'success': True,
                    'forecast': forecast_df['yhat'].values.tolist(),
                    'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'lower_bound': forecast_df.get('yhat_lower', [None]*steps).values.tolist(),
                    'upper_bound': forecast_df.get('yhat_upper', [None]*steps).values.tolist(),
                    'steps': steps,
                    'model': model_name,
                    'target': meta.get('target_column')
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported model type: {model_name}'
                }
        
        except Exception as e:
            logger.exception(f"Forecast generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# -------------------------------------------------------------------
# RESPONSE FORMATTER
# -------------------------------------------------------------------
class ResponseFormatter:
    """Format responses for different intents."""
    
    @staticmethod
    def format_forecast_response(forecast_result, query):
        """Format a forecast result into natural language."""
        if not forecast_result.get('success'):
            return f"I couldn't generate a forecast: {forecast_result.get('error')}"
        
        forecast = forecast_result['forecast']
        steps = forecast_result['steps']
        model = forecast_result['model']
        target = forecast_result['target']
        
        response = [
            f"Based on the {model} model, here's the forecast for {target}:",
            ""
        ]
        
        # Show first 10 values
        display_count = min(10, len(forecast))
        for i in range(display_count):
            if 'forecast_dates' in forecast_result:
                date_str = forecast_result['forecast_dates'][i]
                value = forecast[i]
                response.append(f"  {date_str}: {value:.2f}")
            else:
                response.append(f"  Step {i+1}: {forecast[i]:.2f}")
        
        if len(forecast) > display_count:
            response.append(f"  ... and {len(forecast) - display_count} more steps")
        
        # Summary statistics
        response.extend([
            "",
            f"Summary:",
            f"  • Mean forecasted value: {np.mean(forecast):.2f}",
            f"  • Min: {np.min(forecast):.2f}",
            f"  • Max: {np.max(forecast):.2f}",
            f"  • Trend: {'Increasing' if forecast[-1] > forecast[0] else 'Decreasing'}"
        ])
        
        return "\n".join(response)
    
    @staticmethod
    def format_model_info(model_info):
        """Format model information."""
        meta = model_info['metadata']
        
        response = [
            f"Model: {meta.get('model_name')}",
            f"Target Column: {meta.get('target_column')}",
            f"Training Period: {meta.get('train_start')} to {meta.get('train_end')}",
            ""
        ]
        
        if meta.get('exogenous_features'):
            response.append("Exogenous Features:")
            for feat in meta['exogenous_features']:
                response.append(f"  • {feat}")
            response.append("")
        
        response.append(f"Created: {meta.get('created_at')}")
        
        return "\n".join(response)
    
    @staticmethod
    def format_data_info(meta):
        """Format dataset information."""
        response = [
            f"Dataset: {meta.get('filename', 'Unknown')}",
            f"Category: {meta.get('category', 'Unknown')}",
            f"Rows: {meta.get('row_count', 'Unknown')}",
            ""
        ]
        
        if meta.get('columns'):
            response.append("Columns:")
            for col in meta['columns'][:20]:  # Show first 20
                response.append(f"  • {col}")
            if len(meta['columns']) > 20:
                response.append(f"  ... and {len(meta['columns']) - 20} more")
        
        return "\n".join(response)


# -------------------------------------------------------------------
# INIT GEMINI CLIENT
# -------------------------------------------------------------------
def init_gemini():
    api_key = "AIzaSyDFQjtwbWaxVRhEIHZVqiRByg4GS9gW0z0"
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
# INIT CYBORG VECTOR STORE
# -------------------------------------------------------------------
def get_vector_store():
    """Initialize CyborgVectorStore."""
    if CyborgVectorStore is None:
        logger.error("CyborgVectorStore not importable")
        return None

    try:
        cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
        if not cyborg_api_key:
            logger.error("Missing CYBORG_API_KEY")
            return None

        cyborg_index_name = current_app.config.get('CYBORG_INDEX_NAME', 'embedded_index_v15')
        models_dir = current_app.config.get('MODELS_FOLDER', os.path.join(os.getcwd(), 'models'))
        keys_folder = os.path.join(models_dir, 'cyborg_indexes')
        key_path = os.path.join(keys_folder, f"{cyborg_index_name}.key")

        if not os.path.exists(key_path):
            logger.error("Cyborg index key missing: %s", key_path)
            return None

        with open(key_path, 'rb') as f:
            index_key = f.read()

        storage_index = current_app.config.get('CYBORG_INDEX_STORAGE', 'postgres')
        storage_config = current_app.config.get('CYBORG_CONFIG_STORAGE', 'postgres')
        storage_items = current_app.config.get('CYBORG_ITEMS_STORAGE', 'postgres')

        pg_uri = current_app.config.get(
            'CYBORG_PG_URI',
            os.environ.get('CYBORG_PG_URI', 'postgresql://cyborg:cyborg123@localhost:5432/cyborgdb'),
        )

        tbl_index = current_app.config.get('CYBORG_INDEX_TABLE', 'cyborg_index')
        tbl_config = current_app.config.get('CYBORG_CONFIG_TABLE', 'cyborg_config')
        tbl_items = current_app.config.get('CYBORG_ITEMS_TABLE', 'cyborg_items')

        index_loc = _make_dbconfig(
            storage_index,
            connection_string=pg_uri if storage_index == 'postgres' else None,
            table_name=tbl_index,
        )
        config_loc = _make_dbconfig(
            storage_config,
            connection_string=pg_uri if storage_config == 'postgres' else None,
            table_name=tbl_config,
        )
        items_loc = _make_dbconfig(
            storage_items,
            connection_string=pg_uri if storage_items == 'postgres' else None,
            table_name=tbl_items,
        )

        embedding_model = (
            current_app.config.get('CYBORG_EMBEDDING_MODEL')
            or os.environ.get('CYBORG_EMBEDDING_MODEL')
            or 'all-MiniLM-L6-v2'
        )

        vs = CyborgVectorStore(
            index_name=cyborg_index_name,
            index_key=index_key,
            api_key=cyborg_api_key,
            embedding=embedding_model,
            index_location=index_loc,
            config_location=config_loc,
            items_location=items_loc,
            metric='cosine',
        )

        logger.info("CyborgVectorStore initialized")
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
                'text': d.page_content,
                'metadata': d.metadata,
                'score': float(score)
            })
        return formatted
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return []


# -------------------------------------------------------------------
# ENHANCED RAG PROMPT BUILDER
# -------------------------------------------------------------------
def build_enhanced_prompt(docs, query, context_info):
    """Build prompt with model and data context."""
    blocks = [
        "You are an AI assistant specialized in time series forecasting and data analysis.",
        "Answer the user's question based on the provided documents and context.",
        "If the information is insufficient, clearly state what's missing.",
        ""
    ]
    
    # Add context
    if context_info:
        blocks.append("CONTEXT:")
        if context_info.get('available_models'):
            blocks.append(f"Available Models: {', '.join(context_info['available_models'])}")
        if context_info.get('latest_model'):
            blocks.append(f"Latest Model: {context_info['latest_model']}")
        if context_info.get('target_columns'):
            blocks.append(f"Target Columns: {', '.join(context_info['target_columns'])}")
        blocks.append("")
    
    # Add retrieved documents
    blocks.append("RELEVANT DOCUMENTS:")
    for i, d in enumerate(docs):
        snippet = d['text'][:2000]
        blocks.append(f"--- Document {i+1} (Score: {d['score']:.3f}) ---")
        blocks.append(snippet)
        blocks.append("")
    
    blocks.append(f"USER QUESTION:\n{query}\n")
    blocks.append("Provide a clear, concise answer:")
    
    return "\n".join(blocks)


# -------------------------------------------------------------------
# MAIN CHAT ENDPOINT
# -------------------------------------------------------------------
@bp.route('/api/timeseries/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with forecasting capabilities."""
    data = request.get_json() or {}
    query = data.get('query')
    
    if not query:
        return jsonify({'success': False, 'error': 'query required'}), 400
    
    try:
        # 1. Classify intent
        intent, confidence = IntentClassifier.classify(query)
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
        
        # 2. Get models directory
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        
        # 3. Handle forecast intent directly
        if intent == 'forecast':
            # Extract entities
            steps = EntityExtractor.extract_time_steps(query) or 10
            
            # Get latest model
            model_info = ModelManager.get_latest_model(models_dir)
            
            if not model_info:
                return jsonify({
                    'success': False,
                    'error': 'No trained models available. Please upload and train data first.'
                }), 404
            
            # Generate forecast
            forecast_result = ForecastGenerator.generate_forecast(model_info, steps)
            
            if not forecast_result.get('success'):
                return jsonify({
                    'success': False,
                    'error': forecast_result.get('error')
                }), 500
            
            # Format response
            response_text = ResponseFormatter.format_forecast_response(forecast_result, query)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'forecast_data': forecast_result,
                'model_info': model_info['metadata']
            })
        
        # 4. Handle model_info intent
        if intent == 'model_info':
            models = ModelManager.list_models(models_dir)
            
            if not models:
                return jsonify({
                    'success': True,
                    'answer': 'No models have been trained yet.',
                    'intent': intent
                })
            
            # Get latest or specific model
            model_name = EntityExtractor.extract_model_name(
                query,
                [m['name'] for m in models]
            )
            
            if model_name:
                model_info = next(
                    (m for m in models if m['name'] == model_name),
                    None
                )
            else:
                model_info = models[0]  # Latest
            
            response_text = ResponseFormatter.format_model_info(model_info)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'model_info': model_info['metadata']
            })
        
        # 5. For other intents, use RAG
        vs = get_vector_store()
        
        if vs is None:
            return jsonify({
                'success': False,
                'error': 'Vector store not available'
            }), 500
        
        # Semantic search
        docs = semantic_search(vs, query, k=5)
        
        # Build context
        models = ModelManager.list_models(models_dir)
        context_info = {
            'available_models': [m['name'] for m in models],
            'latest_model': models[0]['name'] if models else None,
            'target_columns': list(set(
                m['metadata'].get('target_column')
                for m in models
                if m['metadata'].get('target_column')
            ))
        }
        
        # Build enhanced prompt
        prompt = build_enhanced_prompt(docs, query, context_info)
        
        # Get Gemini response
        model = init_gemini()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Gemini unavailable'
            }), 500
        
        response = model.generate_content(prompt)
        answer = response.text
        
        return jsonify({
            'success': True,
            'answer': answer,
            'intent': intent,
            'sources': docs,
            'context': context_info
        })
    
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# -------------------------------------------------------------------
# ADDITIONAL ENDPOINTS
# -------------------------------------------------------------------

@bp.route('/api/timeseries/chat/models', methods=['GET'])
def list_available_models():
    """List all available models for chat context."""
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        models = ModelManager.list_models(models_dir)
        
        return jsonify({
            'success': True,
            'models': [
                {
                    'name': m['name'],
                    'target': m['metadata'].get('target_column'),
                    'created': m['metadata'].get('created_at'),
                    'features': m['metadata'].get('exogenous_features', [])
                }
                for m in models
            ]
        })
    except Exception as e:
        logger.exception("Failed to list models")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/timeseries/chat/forecast', methods=['POST'])
def direct_forecast():
    """Direct forecast endpoint (bypassing NLP)."""
    data = request.get_json() or {}
    steps = data.get('steps', 10)
    model_name = data.get('model_name')
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        
        if model_name:
            models = ModelManager.list_models(models_dir)
            model_info = next(
                (m for m in models if m['name'] == model_name),
                None
            )
        else:
            model_info = ModelManager.get_latest_model(models_dir)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404
        
        forecast_result = ForecastGenerator.generate_forecast(model_info, steps)
        
        return jsonify(forecast_result)
    
    except Exception as e:
        logger.exception("Forecast error")
        return jsonify({'success': False, 'error': str(e)}), 500