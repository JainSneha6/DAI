# chat_blueprint_marketing_mmm.py
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

bp = Blueprint("marketing_mmm_chat_bp", __name__)

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None
    logger.error("Pandas/Numpy not available")

try:
    import google.generativeai as genai
except Exception:
    genai = None
    logger.error("Gemini not installed")


# -------------------------------------------------------------------
# INTENT CLASSIFIER
# -------------------------------------------------------------------
class IntentClassifier:
    INTENTS = {
        'roi_attribution': [
            r'roi', r'return', r'attribution', r'contribution', r'which channel',
            r'best channel', r'most effective', r'impact', r'drive revenue'
        ],
        'model_performance': [
            r'performance', r'r2', r'accuracy', r'metrics', r'how good', r'model score'
        ],
        'channel_spend': [
            r'spend', r'budget', r'actual spend', r'how much spent', r'channel cost'
        ],
        'recommendation': [
            r'recommend', r'suggest', r'optimize', r'where to spend', r'allocation'
        ],
        'model_info': [
            r'model', r'trained', r'elasticnet', r'ridge', r'lasso', r'adstock'
        ],
        'data_info': [
            r'data', r'channels?', r'campaigns', r'time period', r'revenue'
        ]
    }

    @classmethod
    def classify(cls, query):
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
# MODEL MANAGER
# -------------------------------------------------------------------
class ModelManager:
    @staticmethod
    def list_models(models_dir):
        models = []
        try:
            files = os.listdir(models_dir)
            pkl_files = [f for f in files if f.endswith('.pkl') and ('ElasticNet' in f or 'Ridge' in f or 'Lasso' in f)]
            for pkl_file in pkl_files:
                meta_file = pkl_file.replace('.pkl', '.meta.json')
                if meta_file in files:
                    meta_path = os.path.join(models_dir, meta_file)
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    models.append({
                        'name': f"{meta.get('model_name')} ({meta.get('created_at', '')[:10]})",
                        'pkl_file': pkl_file,
                        'meta_file': meta_file,
                        'metadata': meta
                    })
            models.sort(key=lambda m: m['metadata'].get('created_at', ''), reverse=True)
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return models

    @staticmethod
    def load_model(models_dir, pkl_file):
        path = os.path.join(models_dir, pkl_file)
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_latest_model(models_dir):
        models = ModelManager.list_models(models_dir)
        return models[0] if models else None


# -------------------------------------------------------------------
# INSIGHT & ATTRIBUTION FORMATTING
# -------------------------------------------------------------------
class ResponseFormatter:
    @staticmethod
    def format_roi_attribution(attribution, spend_cols):
        if not attribution:
            return "No attribution results available."

        lines = [
            "Channel Attribution & ROI (Approximate from Model Coefficients)",
            "──────────────────────────────────────────────────────",
            f"{'Channel':<25} {'Spend':>12} {'Contribution':>15} {'ROI':>10}",
            "──────────────────────────────────────────────────────",
        ]

        total_spend = sum(info['total_spend'] for info in attribution.values())
        total_contrib = sum(info['approx_contribution'] for info in attribution.values())

        sorted_channels = sorted(
            attribution.items(),
            key=lambda x: x[1]['approx_roi'] if x[1]['approx_roi'] is not None else 0,
            reverse=True
        )

        for channel, info in sorted_channels:
            spend = info['total_spend']
            contrib = info['approx_contribution']
            roi = info['approx_roi']
            roi_str = f"{roi:.2f}x" if roi is not None else "N/A"
            percent_contrib = (contrib / total_contrib * 100) if total_contrib > 0 else 0
            lines.append(
                f"{channel:<25} ${spend:>11.0f} ${contrib:>14.0f}  {roi_str:>9} ({percent_contrib:>5.1f}%)"
            )

        lines.extend([
            "──────────────────────────────────────────────────────",
            f"Total Spend: ${total_spend:,.0f} | Total Attributed Revenue: ${total_contrib:,.0f}"
        ])

        return "\n".join(lines)

    @staticmethod
    def format_model_performance(metrics, model_name):
        r2 = metrics.get('r2', 0)
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        mape = metrics.get('mape', 0)

        quality = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Moderate" if r2 > 0.4 else "Poor"

        return "\n".join([
            f"Model Performance: {model_name}",
            f"• R² Score: {r2:.3f} ({quality})",
            f"• RMSE: {rmse:,.0f}",
            f"• MAE: {mae:,.0f}",
            f"• MAPE: {mape:.1f}%",
            "",
            "Higher R² means better fit. This model explains revenue drivers reliably." if r2 > 0.6 else "Model fit is weak — consider more data or features."
        ])

    @staticmethod
    def format_recommendation(attribution):
        if not attribution:
            return "Not enough data for recommendations."

        valid_roi = {k: v['approx_roi'] for k, v in attribution.items() if v['approx_roi'] is not None}
        if not valid_roi:
            return "ROI could not be calculated reliably."

        best = max(valid_roi.items(), key=lambda x: x[1])
        worst = min(valid_roi.items(), key=lambda x: x[1])

        return "\n".join([
            "Optimization Recommendations:",
            f"• Increase budget for **{best[0]}** (ROI: {best[1]:.2f}x) — highest return",
            f"• Review or reduce **{worst[0]}** (ROI: {worst[1]:.2f}x) — lowest efficiency",
            "• Reallocate from low-ROI to high-ROI channels for better results."
        ])

    @staticmethod
    def format_model_info(meta):
        channels = meta.get('unique_channels', meta.get('spend_columns_detected', []))
        return "\n".join([
            "Marketing Mix Model Details",
            f"• Model: {meta.get('model_name')}",
            f"• Target: {meta.get('target_column')}",
            f"• Channels: {', '.join(channels) if channels else 'Detected automatically'}",
            f"• Adstock: {'Yes' if meta.get('adstock_applied') else 'No'} (decay={meta.get('adstock_decay')})",
            f"• Data Period: {meta.get('data_start_date', '?')} to {meta.get('data_end_date', '?')}",
            f"• Campaigns: {meta.get('row_count_original', '?')} → Daily rows: {meta.get('row_count_aggregated', '?')}",
            f"• Created: {meta.get('created_at', '')[:10]}",
        ])


# -------------------------------------------------------------------
# GEMINI SETUP (optional fallback)
# -------------------------------------------------------------------
def init_gemini():
    api_key = "AIzaSyB_cMKuBZPux9FttkqZSFEsDJjcUlyukqY"  # Replace with env in production
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logger.exception("Gemini init failed")
        return None


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@bp.route('/api/marketing_mmm/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'query required'}), 400

    models_dir = current_app.config.get('MODELS_FOLDER', 'models/marketing')
    intent, _ = IntentClassifier.classify(query)

    try:
        latest_model = ModelManager.get_latest_model(models_dir)

        if not latest_model:
            return jsonify({
                'success': False,
                'error': 'No MMM model trained yet. Please upload campaign data and run the Marketing ROI pipeline.'
            })

        meta = latest_model['metadata']
        
        # FIXED: Load attribution from metadata file - this should be saved during model training
        attribution = meta.get('attribution', {})
        best_metrics = meta.get('metrics', {})
        
        # Fallback: If no attribution in metadata, we need to compute it
        # This requires the training data which may not be available
        if not attribution:
            logger.warning("Attribution data not found in model metadata. Please ensure training pipeline saves attribution.")
            # Try to extract spend columns to show helpful error
            spend_cols = meta.get('spend_columns_detected', [])
            unique_channels = meta.get('unique_channels', [])
            
            # Create placeholder message explaining the issue
            return jsonify({
                'success': True,
                'answer': f"""⚠️ Attribution data is not available in the saved model.

The model was trained on these channels: {', '.join(unique_channels) if unique_channels else 'Unknown'}

**To fix this issue:**
1. Ensure the training pipeline saves 'attribution' data in the .meta.json file
2. Attribution should include for each channel:
   - total_spend: Total money spent
   - approx_contribution: Revenue attributed to the channel
   - approx_roi: Return on investment (contribution/spend)

3. Re-train the model to populate attribution data

**What you can still do:**
- Ask about model performance metrics
- View model configuration details
- Check training data information""",
                'intent': intent,
                'model_info': meta,
                'needs_retraining': True
            })

        if intent == 'roi_attribution':
            spend_cols = meta.get('spend_columns_detected', [])
            answer = ResponseFormatter.format_roi_attribution(attribution, spend_cols)
            return jsonify({
                'success': True,
                'answer': answer,
                'intent': intent,
                'attribution_data': attribution,
                'model_info': meta
            })

        elif intent == 'model_performance':
            model_name = meta.get('model_name', 'Unknown')
            answer = ResponseFormatter.format_model_performance(best_metrics, model_name)
            return jsonify({
                'success': True,
                'answer': answer,
                'intent': intent,
                'metrics': best_metrics
            })

        elif intent == 'recommendation':
            answer = ResponseFormatter.format_recommendation(attribution)
            return jsonify({
                'success': True,
                'answer': answer,
                'intent': intent,
                'attribution_data': attribution
            })

        elif intent == 'model_info':
            answer = ResponseFormatter.format_model_info(meta)
            return jsonify({
                'success': True,
                'answer': answer,
                'intent': intent,
                'model_info': meta
            })

        elif intent == 'channel_spend':
            if attribution:
                spend_summary = []
                for ch, info in attribution.items():
                    spend_summary.append(f"• {ch}: ${info['total_spend']:,.0f}")
                answer = "Channel Spend Summary:\n" + "\n".join(spend_summary)
            else:
                answer = "Spend data not available. Please retrain the model with attribution enabled."
            
            return jsonify({
                'success': True,
                'answer': answer,
                'intent': intent
            })

        # Fallback to Gemini for complex queries
        model = init_gemini()
        if model:
            prompt = f"""
You are a marketing analytics expert specializing in Marketing Mix Modeling (MMM).
Explain the user's question using standard MMM concepts like adstock, saturation, channel ROI, and attribution.

User question: {query}

Keep response professional, concise, and insightful.
"""
            response = model.generate_content(prompt)
            answer = response.text
        else:
            answer = "I can help with ROI, channel performance, and budget recommendations. Try asking about specific channels or model metrics."

        return jsonify({
            'success': True,
            'answer': answer,
            'intent': intent
        })

    except Exception as e:
        logger.exception("MMM Chat error")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/marketing_mmm/chat/models', methods=['GET'])
def list_models():
    models_dir = current_app.config.get('MODELS_FOLDER', 'models/marketing')
    models = ModelManager.list_models(models_dir)
    return jsonify({
        'success': True,
        'models': [{
            'name': m['name'],
            'target': m['metadata'].get('target_column'),
            'channels': len(m['metadata'].get('unique_channels') or m['metadata'].get('spend_columns_detected', [])),
            'created': m['metadata'].get('created_at'),
            'r2': m['metadata'].get('metrics', {}).get('r2')
        } for m in models]
    })