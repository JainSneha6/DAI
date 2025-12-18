# routes/logistics_chat_blueprint.py
# Enhanced chat interface with supplier risk and routing optimization capabilities
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

bp = Blueprint("logistics_chat_bp", __name__)

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
    """Classify user intent for logistics queries."""
    
    INTENTS = {
        'optimize': [
            r'optimize', r'allocation', r'assign', r'route', r'plan',
            r'schedule', r'distribute', r'what should', r'best suppliers'
        ],
        'risk': [
            r'risk', r'score', r'reliability', r'performance',
            r'on_time', r'defect', r'safety', r'assess'
        ],
        'analyze': [
            r'analyze', r'summary', r'insights', r'trends',
            r'describe', r'what is', r'show me', r'overview'
        ],
        'supplier_info': [
            r'supplier', r'vendor', r'capacity', r'cost',
            r'lead_time', r'location', r'details', r'which supplier'
        ],
        'data_info': [
            r'data', r'columns', r'features', r'demand', r'dataset',
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
    def extract_suppliers(query, available_suppliers):
        """Extract mentioned suppliers from query."""
        query_lower = query.lower()
        mentioned = []
        for supp in available_suppliers:
            if supp.lower() in query_lower:
                mentioned.append(supp)
        return mentioned if mentioned else None
    
    @staticmethod
    def extract_demands(query):
        """Extract demand-related quantities."""
        # Simple pattern for quantities
        match = re.search(r'(\d+(?:\.\d+)?)\s*(units?|qty|demand)', query.lower())
        return float(match.group(1)) if match else None
    
    @staticmethod
    def extract_time_horizon(query):
        """Extract time horizon limit."""
        patterns = [
            r'(\d+)\s*(days?|weeks?|months?|periods?)',
            r'for\s+(\d+)\s*(days?|weeks?|months?|periods?)',
            r'next\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None


# -------------------------------------------------------------------
# ARTIFACT MANAGER
# -------------------------------------------------------------------
class ArtifactManager:
    """Manage loaded artifacts and their metadata for logistics plans."""
    
    @staticmethod
    def list_artifacts(models_dir):
        """List all available artifacts with metadata."""
        artifacts = []
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
                        artifacts.append({
                            'name': meta.get('model', 'Logistics Plan'),
                            'pkl_file': pkl_file,
                            'meta_file': meta_file,
                            'metadata': meta
                        })
                except Exception as e:
                    logger.warning(f"Failed to load metadata {meta_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
        
        return artifacts
    
    @staticmethod
    def load_artifact(models_dir, pkl_file):
        """Load a specific artifact."""
        path = os.path.join(models_dir, pkl_file)
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            return loaded.get('results'), loaded.get('metadata')
    
    @staticmethod
    def get_latest_artifact(models_dir):
        """Get the most recently created artifact."""
        artifacts = ArtifactManager.list_artifacts(models_dir)
        if not artifacts:
            return None
        
        artifacts.sort(
            key=lambda a: a['metadata'].get('created_at', ''),
            reverse=True
        )
        return artifacts[0]


# -------------------------------------------------------------------
# PLAN GENERATOR
# -------------------------------------------------------------------
class PlanGenerator:
    """Generate or query logistics plans from artifacts."""
    
    @staticmethod
    def generate_plan_summary(artifact_results, metadata):
        """Extract and summarize key plan elements."""
        try:
            suppliers = artifact_results.get('suppliers', [])
            combined_risk = artifact_results.get('combined_risk', {})
            allocation = artifact_results.get('allocation', {})
            routes = artifact_results.get('routes', {})
            solver_used = artifact_results.get('solver_used', 'Unknown')
            
            # Use floats for risk_scores
            risk_scores = {s: float(combined_risk.get(s, 0.5)) for s in suppliers[:5]}  # Top 5, float
            
            summary = {
                'success': True,
                'suppliers': suppliers,
                'risk_scores': risk_scores,  # floats
                'allocation_summary': {
                    s: len(alloc.get(s, {})) for s, alloc in allocation.items()
                },
                'routes_summary': {
                    s: len(routes.get(s, [])) for s in suppliers[:5]
                },
                'solver': solver_used,
                'total_suppliers': len(suppliers),
                'risk_weight': metadata.get('risk_weight', 10.0),
                'vehicle_capacity': metadata.get('vehicle_capacity', 100.0),
                'total_demands': sum(len(supplier_alloc) for supplier_alloc in allocation.values())  # Fixed: use values()
            }
            return summary
        except Exception as e:
            logger.exception(f"Plan summary failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def query_allocation(artifact_results, supplier_query=None):
        """Query allocation for specific supplier or all."""
        allocation = artifact_results.get('allocation', {})
        if supplier_query:
            return allocation.get(supplier_query, {})
        return allocation
    
    @staticmethod
    def query_routes(artifact_results, supplier_query=None):
        """Query routes for specific supplier or all."""
        routes = artifact_results.get('routes', {})
        if supplier_query:
            return routes.get(supplier_query, [])
        return routes


# -------------------------------------------------------------------
# RESPONSE FORMATTER
# -------------------------------------------------------------------
class ResponseFormatter:
    """Format responses for different intents."""
    
    @staticmethod
    def format_optimize_response(plan_summary, query):
        """Format optimization plan into natural language."""
        if not plan_summary.get('success'):
            return f"I couldn't generate an optimization plan: {plan_summary.get('error')}"
        
        suppliers = plan_summary['suppliers'][:3]
        response = [
            f"Based on the latest logistics plan, here's the optimization summary:",
            ""
        ]
        
        response.extend([
            f"• Solver used: {plan_summary['solver']}",
            f"• Total suppliers: {plan_summary['total_suppliers']}",
            f"• Risk weight: {plan_summary['risk_weight']}",
            f"• Vehicle capacity: {plan_summary['vehicle_capacity']}",
            ""
        ])
        
        response.append("Top supplier risk scores:")
        for s in suppliers:
            score = plan_summary['risk_scores'].get(s, 0.5)
            response.append(f"  • {s}: {score:.3f}")
        
        response.append("")
        response.append("Allocation (assigned demands per supplier):")
        for s, count in list(plan_summary['allocation_summary'].items())[:5]:
            response.append(f"  • {s}: {count} demands")
        
        response.append("")
        response.append("Routes (vehicles per supplier):")
        for s, num_routes in list(plan_summary['routes_summary'].items())[:3]:
            response.append(f"  • {s}: {num_routes} routes")
        
        return "\n".join(response)
    
    @staticmethod
    def format_risk_response(artifact_results, query):
        """Format risk scores."""
        combined_risk = artifact_results.get('combined_risk', {})
        suppliers = list(combined_risk.keys())[:5]
        
        response = [
            "Supplier Risk Scores (0=low risk, 1=high risk):",
            ""
        ]
        for s in suppliers:
            score = combined_risk.get(s, 0.5)
            response.append(f"• {s}: {score:.3f} ({'High' if score > 0.7 else 'Medium' if score > 0.3 else 'Low'})")
        
        return "\n".join(response)
    
    @staticmethod
    def format_supplier_info(artifact_results, supplier_query):
        """Format supplier details."""
        if not supplier_query:
            return "Please specify a supplier."
        
        stats = artifact_results.get('stats', {})
        combined_risk = artifact_results.get('combined_risk', {})
        allocation = artifact_results.get('allocation', {})
        
        s_stats = stats.get(supplier_query, {})
        risk = combined_risk.get(supplier_query, 0.5)
        alloc_count = len(allocation.get(supplier_query, {}))
        
        response = [
            f"Supplier: {supplier_query}",
            f"Risk Score: {risk:.3f}",
            f"Total Supply: {s_stats.get('total_supply', 0):.2f}",
            f"Avg Lead Time: {s_stats.get('avg_lead_time', 'N/A'):.2f} days",
            f"On-Time Rate: {s_stats.get('on_time_rate', 'N/A'):.1%}",
            f"Defect Rate: {s_stats.get('defect_rate', 'N/A'):.1%}",
            f"Assigned Demands: {alloc_count}",
            ""
        ]
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

        cyborg_index_name = current_app.config.get('CYBORG_INDEX_NAME', 'embedded_index_v16')
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
    """Build prompt with plan and data context."""
    blocks = [
        "You are an AI assistant specialized in logistics, supplier risk assessment, and routing optimization.",
        "Answer the user's question based on the provided documents and context.",
        "If the information is insufficient, clearly state what's missing.",
        ""
    ]
    
    # Add context
    if context_info:
        blocks.append("CONTEXT:")
        if context_info.get('available_artifacts'):
            blocks.append(f"Available Plans: {', '.join(context_info['available_artifacts'])}")
        if context_info.get('latest_artifact'):
            blocks.append(f"Latest Plan: {context_info['latest_artifact']}")
        if context_info.get('suppliers'):
            blocks.append(f"Suppliers: {', '.join(context_info['suppliers'][:5])}")
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
@bp.route('/api/logistics/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with logistics optimization capabilities."""
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
        
        # 3. Handle optimize intent directly
        if intent == 'optimize':
            # Extract entities
            time_horizon = EntityExtractor.extract_time_horizon(query) or None
            
            # Get latest artifact
            artifact_info = ArtifactManager.get_latest_artifact(models_dir)
            
            if not artifact_info:
                return jsonify({
                    'success': False,
                    'error': 'No optimization plans available. Please upload and run the pipeline first.'
                }), 404
            
            results, metadata = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
            plan_summary = PlanGenerator.generate_plan_summary(results, metadata)
            
            if not plan_summary.get('success'):
                return jsonify({
                    'success': False,
                    'error': plan_summary.get('error')
                }), 500
            
            # Format response
            response_text = ResponseFormatter.format_optimize_response(plan_summary, query)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'plan_data': plan_summary,
                'artifact_info': artifact_info['metadata']
            })
        
        # 4. Handle risk intent
        if intent == 'risk':
            artifact_info = ArtifactManager.get_latest_artifact(models_dir)
            
            if not artifact_info:
                return jsonify({
                    'success': False,
                    'error': 'No plans available for risk assessment.'
                }), 404
            
            results, _ = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
            response_text = ResponseFormatter.format_risk_response(results, query)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'risk_data': results.get('combined_risk', {}),
                'artifact_info': artifact_info['metadata']
            })
        
        # 5. Handle supplier_info intent
        if intent == 'supplier_info':
            artifact_info = ArtifactManager.get_latest_artifact(models_dir)
            
            if not artifact_info:
                return jsonify({
                    'success': False,
                    'error': 'No plans available for supplier details.'
                }), 404
            
            results, _ = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
            supplier_query = EntityExtractor.extract_suppliers(query, results.get('suppliers', []))
            if not supplier_query:
                supplier_query = results.get('suppliers', [None])[0] if results.get('suppliers') else None
            
            response_text = ResponseFormatter.format_supplier_info(results, supplier_query[0] if isinstance(supplier_query, list) and supplier_query else supplier_query)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'supplier_data': PlanGenerator.query_allocation(results, supplier_query[0] if isinstance(supplier_query, list) and supplier_query else supplier_query),
                'artifact_info': artifact_info['metadata']
            })
        
        # 6. For other intents, use RAG
        vs = get_vector_store()
        
        if vs is None:
            return jsonify({
                'success': False,
                'error': 'Vector store not available'
            }), 500
        
        # Semantic search
        docs = semantic_search(vs, query, k=5)
        
        # Build context
        artifacts = ArtifactManager.list_artifacts(models_dir)
        context_info = {
            'available_artifacts': [a['name'] for a in artifacts],
            'latest_artifact': artifacts[0]['name'] if artifacts else None,
            'suppliers': list(set(
                a['metadata'].get('suppliers', [])
                for a in artifacts
                if a['metadata'].get('suppliers')
            ))[0][:5] if artifacts else []
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

@bp.route('/api/logistics/chat/plans', methods=['GET'])
def list_available_plans():
    """List all available plans for chat context."""
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        artifacts = ArtifactManager.list_artifacts(models_dir)
        
        return jsonify({
            'success': True,
            'plans': [
                {
                    'name': a['name'],
                    'suppliers_count': len(a['metadata'].get('suppliers', [])),
                    'created': a['metadata'].get('created_at'),
                    'solver': a['metadata'].get('solver_used', 'Unknown')
                }
                for a in artifacts
            ]
        })
    except Exception as e:
        logger.exception("Failed to list plans")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/logistics/chat/optimize', methods=['POST'])
def direct_optimize():
    """Direct optimization query endpoint (bypassing NLP)."""
    data = request.get_json() or {}
    time_horizon = data.get('time_horizon')
    plan_name = data.get('plan_name')
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        
        if plan_name:
            artifacts = ArtifactManager.list_artifacts(models_dir)
            artifact_info = next(
                (a for a in artifacts if a['name'] == plan_name),
                None
            )
        else:
            artifact_info = ArtifactManager.get_latest_artifact(models_dir)
        
        if not artifact_info:
            return jsonify({
                'success': False,
                'error': 'Plan not found'
            }), 404
        
        results, metadata = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
        plan_summary = PlanGenerator.generate_plan_summary(results, metadata)
        
        return jsonify(plan_summary)
    
    except Exception as e:
        logger.exception("Optimization query error")
        return jsonify({'success': False, 'error': str(e)}), 500