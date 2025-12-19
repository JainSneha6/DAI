# chat_blueprint.py
# Enhanced chat interface with inventory optimization capabilities
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime
import pickle
import csv
from io import StringIO, BytesIO
logger = logging.getLogger(__name__)
bp = Blueprint("inventory_chat_bp", __name__)
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
try:
    import pulp as pl
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False
    logger.error("Pulp not available")
# Import the inventory pipeline
try:
    from services.inventory_optimization_pipeline import run_inventory_optimization_pipeline
except Exception:
    run_inventory_optimization_pipeline = None
    logger.error("Inventory pipeline not available")
# -------------------------------------------------------------------
# INTENT CLASSIFICATION
# -------------------------------------------------------------------
class IntentClassifier:
    """Classify user intent for inventory queries."""
    
    INTENTS = {
        'optimize': [
            r'optimize', r'schedule', r'replenish', r'plan', r'order',
            r'eoq', r'poq', r'inventory plan'
        ],
        'analyze': [
            r'analyze', r'analysis', r'tell me about', r'explain',
            r'describe', r'what is', r'show me', r'summary'
        ],
        'compare': [
            r'compare', r'difference', r'versus', r'vs', r'better',
            r'which plan', r'cost'
        ],
        'plan_info': [
            r'plan', r'generated', r'parameters', r'performance',
            r'metrics', r'how does', r'what plan'
        ],
        'data_info': [
            r'data', r'skus', r'features', r'demand', r'dataset',
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
    def extract_time_horizon(query):
        """Extract time horizon from query."""
        # Pattern: "next 10 periods", "30 weeks", etc.
        patterns = [
            r'(\d+)\s*(day|week|month|year|period)s?',
            r'for\s+(\d+)\s*(day|week|month|year|period)s?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                num = int(match.group(1))
                return num
        
        return None
    
    @staticmethod
    def extract_sku_name(query, available_skus):
        """Find if user mentions specific SKU."""
        query_lower = query.lower()
        for sku in available_skus:
            if sku.lower() in query_lower:
                return sku
        return None
# -------------------------------------------------------------------
# MODEL MANAGER
# -------------------------------------------------------------------
class ModelManager:
    """Manage loaded inventory plans and their metadata."""
    
    @staticmethod
    def list_models(models_dir):
        """List all available plans with metadata."""
        plans = []
        try:
            files = os.listdir(models_dir)
            meta_files = [f for f in files if f.endswith('.meta.json')]
            
            for meta_file in meta_files:
                if 'inventory' not in meta_file.lower():
                    continue  # Filter for inventory plans
                meta_path = os.path.join(models_dir, meta_file)
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    base = meta_file.replace('.meta.json', '')
                    pkl_file = f"{base}.pkl"
                    
                    if pkl_file in files:
                        plans.append({
                            'name': meta.get('model', 'Inventory Plan'),
                            'pkl_file': pkl_file,
                            'meta_file': meta_file,
                            'metadata': meta
                        })
                except Exception as e:
                    logger.warning(f"Failed to load metadata {meta_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to list plans: {e}")
        
        return plans
    
    @staticmethod
    def load_model(models_dir, pkl_file):
        """Load a specific plan."""
        path = os.path.join(models_dir, pkl_file)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_latest_model(models_dir):
        """Get the most recently created plan."""
        plans = ModelManager.list_models(models_dir)
        if not plans:
            return None
        
        plans.sort(
            key=lambda m: m['metadata'].get('created_at', ''),
            reverse=True
        )
        return plans[0]
# -------------------------------------------------------------------
# PLAN GENERATOR
# -------------------------------------------------------------------
class PlanGenerator:
    """Generate or load inventory plans."""
    
    @staticmethod
    def generate_plan(file_path, horizon=None, timeout=30):
        """Run the pipeline if available."""
        if run_inventory_optimization_pipeline is None:
            return {'success': False, 'error': 'Pipeline not available'}
        
        try:
            kwargs = {
                'time_horizon_limit': horizon,
                'milp_timeout_seconds': timeout,
                'use_milp': _HAS_PULP
            }
            return run_inventory_optimization_pipeline(file_path, gemini_analysis={}, **kwargs)
        except Exception as e:
            logger.exception(f"Plan generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_schedule(plan_info, sku=None):
        """Extract schedule from loaded plan."""
        try:
            models_dir = current_app.config.get('MODELS_FOLDER', 'models/inventory')
            artifact = ModelManager.load_model(models_dir, plan_info['pkl_file'])
            schedule = artifact.get('schedule', {})
            inventory = artifact.get('inventory_projection', {})
            metadata = plan_info['metadata']
            
            # Round order quantities to nearest integer
            rounded_schedule = {}
            for s, orders in schedule.items():
                rounded_schedule[s] = {p: round(q) for p, q in orders.items()}
            
            if sku and sku in rounded_schedule:
                return {
                    'success': True,
                    'schedule': rounded_schedule[sku],
                    'inventory': [round(i, 2) for i in inventory.get(sku, [])],  # Round inventory to 2 decimals
                    'sku': sku,
                    'skus': metadata.get('skus', []),
                    'solver': metadata.get('solver_used', 'unknown')
                }
            else:
                return {
                    'success': True,
                    'schedule': rounded_schedule,
                    'inventory': {s: [round(i, 2) for i in inv_list] for s, inv_list in inventory.items()},
                    'skus': metadata.get('skus', []),
                    'solver': metadata.get('solver_used', 'unknown')
                }
        except Exception as e:
            logger.exception(f"Schedule extraction failed: {e}")
            return {'success': False, 'error': str(e)}
# -------------------------------------------------------------------
# RESPONSE FORMATTER
# -------------------------------------------------------------------
class ResponseFormatter:
    """Format responses for different intents."""
    
    @staticmethod
    def format_plan_response(plan_result, query):
        """Format a plan result into natural language."""
        if not plan_result.get('success'):
            return f"I couldn't generate a plan: {plan_result.get('error')}"
        
        schedule = plan_result['schedule']
        skus = plan_result['skus']
        solver = plan_result['solver']
        
        response = [
            f"Based on the {solver} plan, here's the replenishment schedule (quantities rounded to nearest integer):",
            ""
        ]
        
        # Show orders for first few SKUs or specific
        for sku in list(skus)[:3]:  # Limit to first 3 SKUs
            if sku in schedule and schedule[sku]:
                response.append(f"{sku}:")
                for period, qty in list(schedule[sku].items())[:5]:  # First 5 orders
                    response.append(f"  Period {period}: Order {qty}")
                if len(schedule[sku]) > 5:
                    response.append(f"  ... and {len(schedule[sku]) - 5} more orders")
                response.append("")
        
        if len(skus) > 3:
            response.append(f"... and {len(skus) - 3} more SKUs")
            response.append("")
            response.append("Download the full plan (CSV) for all SKUs and orders.")
        
        # Summary
        total_orders = sum(len(orders) for orders in schedule.values())
        response.extend([
            "",
            f"Summary:",
            f" • Total SKUs: {len(skus)}",
            f" • Total Orders: {total_orders}",
            f" • Solver: {solver}"
        ])
        
        return "\n".join(response)
    
    @staticmethod
    def format_eoq_response(plan_result, query):
        """Format EOQ/POQ info."""
        if not plan_result.get('success'):
            return f"Could not retrieve EOQ data: {plan_result.get('error')}"
        
        eoq = plan_result.get('EOQ', {})
        skus = plan_result.get('skus', [])
        
        response = ["Economic Order Quantities (EOQ):", ""]
        for sku in list(skus)[:5]:
            if sku in eoq:
                response.append(f" • {sku}: {eoq[sku]:.2f}")
        if len(skus) > 5:
            response.append(f"... and more")
        
        return "\n".join(response)
    
    @staticmethod
    def format_plan_info(plan_info):
        """Format plan information."""
        meta = plan_info['metadata']
        
        response = [
            f"Plan: {meta.get('model', 'Inventory Plan')}",
            f"Solver: {meta.get('solver_used', 'Unknown')}",
            f"SKUs: {len(meta.get('skus', []))}",
            f"Created: {meta.get('created_at')}",
            ""
        ]
        
        if meta.get('EOQ'):
            response.append("Sample EOQ:")
            for sku, val in list(meta['EOQ'].items())[:3]:
                response.append(f" • {sku}: {val:.2f}")
            response.append("")
        
        return "\n".join(response)
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
        cyborg_index_name = current_app.config.get('CYBORG_INDEX_NAME', 'embedded_index_v16')
        models_dir = current_app.config.get('MODELS_FOLDER', os.path.join(os.getcwd(), 'models/inventory'))
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
    """Build prompt with plan and data context."""
    blocks = [
        "You are an AI assistant specialized in inventory optimization and replenishment planning.",
        "Answer the user's question based on the provided documents and context.",
        "If the information is insufficient, clearly state what's missing.",
        ""
    ]
    
    # Add context
    if context_info:
        blocks.append("CONTEXT:")
        if context_info.get('available_plans'):
            blocks.append(f"Available Plans: {', '.join(context_info['available_plans'])}")
        if context_info.get('latest_plan'):
            blocks.append(f"Latest Plan: {context_info['latest_plan']}")
        if context_info.get('skus'):
            blocks.append(f"SKUs: {', '.join(context_info['skus'][:10])}")
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
@bp.route('/api/inventory/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with inventory optimization capabilities."""
    data = request.get_json() or {}
    query = data.get('query')
    file_path = data.get('file_path')  # Optional for generation
    
    if not query:
        return jsonify({'success': False, 'error': 'query required'}), 400
    
    try:
        # 1. Classify intent
        intent, confidence = IntentClassifier.classify(query)
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
        
        # 2. Get models directory
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/inventory')
        
        # 3. Handle optimize intent
        if intent == 'optimize':
            # Extract entities
            horizon = EntityExtractor.extract_time_horizon(query)
            
            if file_path and run_inventory_optimization_pipeline:
                # Run pipeline if file provided
                plan_result = PlanGenerator.generate_plan(file_path, horizon=horizon)
                if plan_result.get('success'):
                    response_text = ResponseFormatter.format_plan_response(plan_result, query)
                    return jsonify({
                        'success': True,
                        'answer': response_text,
                        'intent': intent,
                        'plan_data': plan_result,
                        'solver': plan_result.get('solver_used')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': plan_result.get('error')
                    }), 500
            
            # Otherwise, load latest plan
            plan_info = ModelManager.get_latest_model(models_dir)
            if not plan_info:
                return jsonify({
                    'success': False,
                    'error': 'No plans available. Provide a file or upload data first.'
                }), 404
            
            schedule_result = PlanGenerator.get_schedule(plan_info, sku=None)
            if not schedule_result.get('success'):
                return jsonify({
                    'success': False,
                    'error': schedule_result.get('error')
                }), 500
            
            response_text = ResponseFormatter.format_plan_response(schedule_result, query)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'plan_data': schedule_result,
                'plan_info': plan_info['metadata']
            })
        
        # 4. Handle plan_info or eoq
        if intent in ['plan_info', 'compare']:
            plans = ModelManager.list_models(models_dir)
            
            if not plans:
                return jsonify({
                    'success': True,
                    'answer': 'No plans have been generated yet.',
                    'intent': intent
                })
            
            # Get latest plan
            plan_info = plans[0]
            
            if intent == 'plan_info':
                response_text = ResponseFormatter.format_plan_info(plan_info)
            else:  # compare or eoq
                artifact = ModelManager.load_model(models_dir, plan_info['pkl_file'])
                eoq_data = artifact.get('metadata', {}).get('EOQ', {})
                response_text = ResponseFormatter.format_eoq_response(
                    {'success': True, 'EOQ': eoq_data, 'skus': plan_info['metadata'].get('skus', [])},
                    query
                )
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'plan_info': plan_info['metadata']
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
        plans = ModelManager.list_models(models_dir)
        context_info = {
            'available_plans': [p['name'] for p in plans],
            'latest_plan': plans[0]['name'] if plans else None,
            'skus': plans[0]['metadata'].get('skus', []) if plans else []
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
@bp.route('/api/inventory/chat/plans', methods=['GET'])
def list_available_plans():
    """List all available plans for chat context."""
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/inventory')
        plans = ModelManager.list_models(models_dir)
        
        return jsonify({
            'success': True,
            'plans': [
                {
                    'name': p['name'],
                    'skus_count': len(p['metadata'].get('skus', [])),
                    'solver': p['metadata'].get('solver_used'),
                    'created': p['metadata'].get('created_at'),
                }
                for p in plans
            ]
        })
    except Exception as e:
        logger.exception("Failed to list plans")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/inventory/chat/schedule', methods=['POST'])
def direct_schedule():
    """Direct schedule endpoint (bypassing NLP)."""
    data = request.get_json() or {}
    file_path = data.get('file_path')
    horizon = data.get('horizon')
    plan_name = data.get('plan_name')
    sku = data.get('sku')
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/inventory')
        
        if file_path and run_inventory_optimization_pipeline:
            plan_result = PlanGenerator.generate_plan(file_path, horizon=horizon)
            if plan_result.get('success'):
                return jsonify(plan_result)
            else:
                return jsonify(plan_result), 500
        
        if plan_name:
            plans = ModelManager.list_models(models_dir)
            plan_info = next(
                (p for p in plans if p['name'] == plan_name),
                None
            )
        else:
            plan_info = ModelManager.get_latest_model(models_dir)
        
        if not plan_info:
            return jsonify({
                'success': False,
                'error': 'Plan not found'
            }), 404
        
        schedule_result = PlanGenerator.get_schedule(plan_info, sku=sku)
        
        return jsonify(schedule_result)
    
    except Exception as e:
        logger.exception("Schedule error")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/inventory/chat/download_plan/<plan_name>', methods=['GET'])
def download_plan(plan_name):
    """Download the full plan as CSV."""
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/inventory')
        plans = ModelManager.list_models(models_dir)
        plan_info = next((p for p in plans if p['name'] == plan_name), None)
        
        if not plan_info:
            return jsonify({'success': False, 'error': 'Plan not found'}), 404
        
        schedule_result = PlanGenerator.get_schedule(plan_info, sku=None)
        if not schedule_result.get('success'):
            return jsonify({'success': False, 'error': schedule_result.get('error')}), 500
        
        # Flatten schedule to CSV rows: sku, period, order_qty
        rows = []
        for sku, orders in schedule_result['schedule'].items():
            for period, qty in orders.items():
                rows.append([sku, period, qty])
        
        if not rows:
            return jsonify({'success': False, 'error': 'No orders in plan'}), 404
        
        # Create CSV in memory using StringIO first
        csv_output = StringIO()
        writer = csv.writer(csv_output)
        writer.writerow(['SKU', 'Period', 'Order_Qty'])
        writer.writerows(rows)
        csv_content = csv_output.getvalue().encode('utf-8')
        
        # Now use BytesIO for send_file
        output = BytesIO(csv_content)
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'inventory_plan_{plan_name}_{timestamp}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.exception("Download failed")
        return jsonify({'success': False, 'error': str(e)}), 500