# chat_blueprint_customer_segmentation.py
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

bp = Blueprint("customer_seg_chat_bp", __name__)

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
# INTENT CLASSIFICATION
# -------------------------------------------------------------------
class IntentClassifier:
    INTENTS = {
        'segment_summary': [
            r'segment', r'cluster', r'group', r'customer type', r'profile',
            r'who are', r'describe segment', r'tell me about segment'
        ],
        'rfm_analysis': [
            r'rfm', r'recency', r'frequency', r'monetary', r'ltv', r'value'
        ],
        'churn_risk': [
            r'churn', r'attrition', r'risk', r'leaving', r'likely to leave'
        ],
        'model_info': [
            r'model', r'trained', r'methods?', r'kmeans', r'gmm', r'autoencoder'
        ],
        'data_info': [
            r'data', r'customers?', r'how many', r'columns', r'uploaded'
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
            pkl_files = [f for f in files if f.endswith('.pkl') and 'Customer_Segmentation' in f]
            for pkl_file in pkl_files:
                meta_file = pkl_file.replace('.pkl', '.meta.json')
                if meta_file in files:
                    meta_path = os.path.join(models_dir, meta_file)
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    models.append({
                        'name': f"Segmentation Model ({meta.get('created_at', 'Unknown')[:10]})",
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
# INSIGHT GENERATOR
# -------------------------------------------------------------------
class InsightGenerator:
    @staticmethod
    def generate_segment_insights(model_data):
        results = model_data.get('results', {})
        customer_summary = results.get('customer_summary_sample', [])
        if not customer_summary:
            return None

        df = pd.DataFrame(customer_summary)
        insights = []

        # Preferred method: try kmeans first, then gmm, then autoencoder
        seg_col = None
        for col in ['kmeans_segment', 'gmm_segment', 'autoencoder_segment']:
            if col in df.columns and df[col].notna().any():
                seg_col = col
                method = col.split('_')[0].upper()
                break
        if not seg_col:
            return None

        segments = df.groupby(seg_col).agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'LTV': 'mean',
            'RFM_score': 'mean'
        }).round(2).to_dict('index')

        segment_profiles = {}
        for seg_id, stats in segments.items():
            profile = {
                'size': int(df[seg_col].value_counts().get(seg_id, 0)),
                'avg_recency_days': stats['recency_days'],
                'avg_frequency': stats['frequency'],
                'avg_monetary': stats['monetary'],
                'avg_ltv': stats['LTV'],
                'avg_rfm': stats['RFM_score'],
                'description': InsightGenerator._describe_segment(stats)
            }
            segment_profiles[int(seg_id)] = profile

        return {
            'method': method,
            'n_segments': len(segments),
            'profiles': segment_profiles,
            'total_customers': len(df)
        }

    @staticmethod
    def _describe_segment(stats):
        r = stats['recency_days']
        f = stats['frequency']
        m = stats['monetary']

        if r < 60 and f > 5 and m > stats['monetary'] * 0.7:
            return "Champions – Loyal & High Value"
        elif r < 90 and f > 3:
            return "Loyal Customers – Frequent Buyers"
        elif r > 180:
            return "At Risk – Inactive High Value"
        elif r > 120 and f < 2:
            return "Hibernating – Lost Customers"
        elif m > stats['monetary'] * 0.8:
            return "Big Spenders – High Monetary"
        else:
            return "Potential Loyalists – Growing"


# -------------------------------------------------------------------
# RESPONSE FORMATTER
# -------------------------------------------------------------------
class ResponseFormatter:
    @staticmethod
    def format_segment_response(insights):
        if not insights:
            return "No segmentation results available yet."

        lines = [
            f"Customer Segmentation Analysis ({insights['method']})",
            f"Total Customers Analyzed: {insights['total_customers']:,}",
            f"Number of Segments: {insights['n_segments']}",
            ""
        ]

        for seg_id, profile in sorted(insights['profiles'].items()):
            lines.extend([
                f"Segment {seg_id} – {profile['description']}",
                f"  • Customers: {profile['size']:,}",
                f"  • Avg Recency: {profile['avg_recency_days']:.0f} days",
                f"  • Avg Frequency: {profile['avg_frequency']:.1f}",
                f"  • Avg Monetary: ${profile['avg_monetary']:.2f}",
                f"  • Avg LTV: ${profile['avg_ltv']:.2f}",
                ""
            ])

        lines.append("Use these segments for targeted marketing, retention campaigns, and personalized offers.")

        return "\n".join(lines)

    @staticmethod
    def format_model_info(meta):
        return "\n".join([
            f"Customer Segmentation Model",
            f"Created: {meta.get('created_at', 'Unknown')}",
            f"Segmentation Methods: {', '.join(meta.get('segmentation_methods', []))}",
            f"Number of Segments: {meta.get('n_segments', 4)}",
            f"Input File: {meta.get('input_file', 'Unknown')}",
        ])

    @staticmethod
    def format_rfm_summary(results):
        sample = results.get('rfm_sample', [])
        if not sample:
            return "No RFM data available."
        df = pd.DataFrame(sample)
        return "\n".join([
            "RFM Overview (Top 5 Customers):",
            f"Avg Recency: {df['recency_days'].mean():.1f} days",
            f"Avg Frequency: {df['frequency'].mean():.1f}",
            f"Avg Monetary: ${df['monetary'].mean():.2f}",
            f"Avg RFM Score: {df['RFM_score'].mean():.1f}",
        ])


# -------------------------------------------------------------------
# GEMINI SETUP
# -------------------------------------------------------------------
def init_gemini():
    api_key = "AIzaSyB_cMKuBZPux9FttkqZSFEsDJjcUlyukqY"  # Replace or use env
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
@bp.route('/api/customer_segmentation/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'query required'}), 400

    models_dir = current_app.config.get('MODELS_FOLDER', 'models/customer')
    intent, _ = IntentClassifier.classify(query)

    try:
        latest_model = ModelManager.get_latest_model(models_dir)

        if intent == 'segment_summary' or intent == 'general':
            if not latest_model:
                return jsonify({'success': False, 'error': 'No segmentation model trained yet.'})

            model_obj = ModelManager.load_model(models_dir, latest_model['pkl_file'])
            insights = InsightGenerator.generate_segment_insights(model_obj)

            if insights:
                answer = ResponseFormatter.format_segment_response(insights)
                return jsonify({
                    'success': True,
                    'answer': answer,
                    'intent': intent,
                    'segment_insights': insights,
                    'model_info': latest_model['metadata']
                })
            else:
                answer = "Segmentation completed but no clear clusters found."

        elif intent == 'rfm_analysis':
            if not latest_model:
                return jsonify({'success': False, 'error': 'No model available.'})
            model_obj = ModelManager.load_model(models_dir, latest_model['pkl_file'])
            answer = ResponseFormatter.format_rfm_summary(model_obj.get('results', {}))
            return jsonify({'success': True, 'answer': answer, 'intent': intent})

        elif intent == 'model_info':
            if not latest_model:
                return jsonify({'success': True, 'answer': 'No segmentation model trained yet.'})
            answer = ResponseFormatter.format_model_info(latest_model['metadata'])
            return jsonify({'success': True, 'answer': answer, 'intent': intent, 'model_info': latest_model['metadata']})

        elif intent == 'churn_risk':
            if not latest_model:
                return jsonify({'success': False, 'error': 'No model available.'})
            model_obj = ModelManager.load_model(models_dir, latest_model['pkl_file'])
            predictive = model_obj.get('results', {}).get('predictive', {})
            if predictive.get('trained'):
                probs = predictive.get('churn_probability_per_customer', {})
                if probs:
                    high_risk = {k: v for k, v in probs.items() if v > 0.5}
                    count = len(high_risk)
                    answer = f"{count} customers have >50% churn risk.\nHighest risk: {max(high_risk.values()):.1%}"
                else:
                    answer = "Churn model trained but no probabilities available."
            else:
                answer = "No predictive churn model was trained (no churn label in data)."
            return jsonify({'success': True, 'answer': answer, 'intent': intent})

        # Fallback: use Gemini for complex questions
        model = init_gemini()
        if model:
            prompt = f"""
You are an expert in customer segmentation and RFM analysis.
Answer the user's question based on typical segmentation practices.

User question: {query}

Provide a helpful, professional response.
"""
            response = model.generate_content(prompt)
            answer = response.text
        else:
            answer = "I'm having trouble connecting to my reasoning engine. Please try a specific question about segments, RFM, or churn."

        return jsonify({
            'success': True,
            'answer': answer,
            'intent': intent
        })

    except Exception as e:
        logger.exception("Chat error")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/customer_segmentation/chat/models', methods=['GET'])
def list_models():
    models_dir = current_app.config.get('MODELS_FOLDER', 'models/customer')
    models = ModelManager.list_models(models_dir)
    return jsonify({
        'success': True,
        'models': [{
            'name': m['name'],
            'created': m['metadata'].get('created_at'),
            'methods': m['metadata'].get('segmentation_methods'),
            'n_segments': m['metadata'].get('n_segments')
        } for m in models]
    })