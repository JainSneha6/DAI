# routes/logistics_chat_blueprint_enhanced.py
# Enhanced chat interface with supplier risk and routing optimization capabilities
# Compatible with the enhanced pipeline v2.0
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime
import pickle
import io
import math

logger = logging.getLogger(__name__)

bp = Blueprint("logistics_chat_bp", __name__)

# -------------------------------------------------------------------
# JSON SANITIZER - Handles NaN, Infinity, and other non-JSON values
# -------------------------------------------------------------------
def sanitize_for_json(obj):
    """
    Recursively sanitize objects for JSON serialization.
    Converts NaN, Infinity, -Infinity to None or appropriate values.
    """
    if obj is None:
        return None
    
    # Handle pandas/numpy types
    if hasattr(obj, 'item'):
        try:
            obj = obj.item()
        except Exception:
            pass
    
    # Handle numbers
    if isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Handle strings
    if isinstance(obj, str):
        return obj
    
    # Handle booleans
    if isinstance(obj, bool):
        return obj
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    # Handle other iterables
    try:
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return [sanitize_for_json(item) for item in obj]
    except Exception:
        pass
    
    # Convert to string as fallback
    return str(obj)


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
            r'schedule', r'distribute', r'what should', r'best suppliers',
            r'recommend', r'suggest'
        ],
        'risk': [
            r'risk', r'score', r'reliability', r'performance',
            r'on_time', r'on-time', r'defect', r'safety', r'assess',
            r'delay', r'quality', r'hazard'
        ],
        'analyze': [
            r'analyze', r'summary', r'insights', r'trends',
            r'describe', r'what is', r'show me', r'overview',
            r'statistics', r'metrics', r'performance'
        ],
        'supplier_info': [
            r'supplier', r'vendor', r'capacity', r'cost',
            r'lead_time', r'location', r'details', r'which supplier',
            r'carrier', r'who', r'tell me about'
        ],
        'data_quality': [
            r'quality', r'completeness', r'missing', r'data issue',
            r'data problem', r'validity', r'accuracy'
        ],
        'routes': [
            r'route', r'routing', r'vehicle', r'delivery route',
            r'path', r'trip', r'journey', r'vrp'
        ],
        'cost': [
            r'cost', r'price', r'expensive', r'cheap', r'budget',
            r'efficiency', r'cost per', r'total cost'
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
    def extract_top_n(query):
        """Extract 'top N' requests."""
        match = re.search(r'top\s+(\d+)', query.lower())
        return int(match.group(1)) if match else 5
    
    @staticmethod
    def extract_threshold(query):
        """Extract risk threshold."""
        match = re.search(r'(?:risk|score).*?(?:above|over|greater|>)\s*(\d+\.?\d*)', query.lower())
        return float(match.group(1)) if match else None


# -------------------------------------------------------------------
# ARTIFACT MANAGER (Enhanced)
# -------------------------------------------------------------------
class ArtifactManager:
    """Manage loaded artifacts and their metadata for logistics plans."""
    
    @staticmethod
    def list_artifacts(models_dir):
        """List all available artifacts with enhanced metadata."""
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
                            'name': meta.get('model_name', 'Logistics Plan'),
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
# PLAN GENERATOR (Enhanced)
# -------------------------------------------------------------------
class PlanGenerator:
    """Generate or query logistics plans from artifacts."""
    
    @staticmethod
    def generate_plan_summary(artifact_results, metadata):
        """Extract and summarize key plan elements with enhanced metrics."""
        try:
            suppliers = artifact_results.get('suppliers', [])
            combined_risk = artifact_results.get('risk_scores', {}).get('combined', {})
            allocation = artifact_results.get('allocation', {}).get('assignments', {})
            routes = artifact_results.get('routes', {})
            solver_used = artifact_results.get('allocation', {}).get('solver_used', 'Unknown')
            
            # Enhanced: Get supplier statistics
            supplier_stats = artifact_results.get('supplier_statistics', {})
            
            # Enhanced: Get data quality
            data_quality = artifact_results.get('data_quality', {})
            
            # Enhanced: Get ML metrics if available
            ml_metrics = artifact_results.get('ml_model_metrics')
            
            # Top suppliers by risk (sorted)
            sorted_suppliers = sorted(
                suppliers,
                key=lambda s: combined_risk.get(s, 0.5)
            )
            
            summary = {
                'success': True,
                'suppliers': suppliers,
                'top_suppliers_by_risk': sorted_suppliers[:5],
                'risk_scores': {s: float(combined_risk.get(s, 0.5)) for s in suppliers},
                'allocation_summary': {
                    s: len(allocation.get(s, {})) for s in suppliers
                },
                'routes_summary': {
                    s: len(routes.get(s, [])) for s in suppliers
                },
                'solver': solver_used,
                'total_suppliers': len(suppliers),
                'configuration': artifact_results.get('configuration', {}),
                'data_quality_score': data_quality.get('overall_quality_score', 0.0),
                'data_summary': metadata.get('data_summary', {}),
                'ml_metrics': ml_metrics,
                'supplier_stats_preview': {
                    s: supplier_stats.get(s, {}) for s in sorted_suppliers[:3]
                }
            }
            return summary
        except Exception as e:
            logger.exception(f"Plan summary failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def query_allocation(artifact_results, supplier_query=None):
        """Query allocation for specific supplier or all."""
        allocation = artifact_results.get('allocation', {}).get('assignments', {})
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
    
    @staticmethod
    def query_supplier_details(artifact_results, supplier_name):
        """Get comprehensive supplier details."""
        stats = artifact_results.get('supplier_statistics', {}).get(supplier_name, {})
        risk_scores = artifact_results.get('risk_scores', {})
        combined_risk = risk_scores.get('combined', {}).get(supplier_name, 0.5)
        stat_risk = risk_scores.get('statistical', {}).get(supplier_name, 0.5)
        ml_risk = risk_scores.get('ml_based', {}).get(supplier_name)
        survival_risk = risk_scores.get('survival_based', {}).get(supplier_name)
        
        attributes = artifact_results.get('supplier_attributes', {})
        unit_cost = attributes.get('unit_cost', {}).get(supplier_name, 0.0)
        capacity = attributes.get('capacity', {}).get(supplier_name, 0.0)
        location = attributes.get('locations', {}).get(supplier_name, (0.0, 0.0))
        
        allocation = artifact_results.get('allocation', {}).get('assignments', {})
        routes = artifact_results.get('routes', {})
        
        return {
            'supplier_name': supplier_name,
            'statistics': stats,
            'risk_scores': {
                'combined': float(combined_risk),
                'statistical': float(stat_risk),
                'ml_based': float(ml_risk) if ml_risk is not None else None,
                'survival_based': float(survival_risk) if survival_risk is not None else None
            },
            'attributes': {
                'unit_cost_usd': float(unit_cost),
                'capacity': float(capacity),
                'location': location
            },
            'allocation': {
                'assigned_shipments': len(allocation.get(supplier_name, {})),
                'shipment_ids': list(allocation.get(supplier_name, {}).keys())[:10]
            },
            'routing': {
                'total_routes': len(routes.get(supplier_name, [])),
                'routes_preview': routes.get(supplier_name, [])[:3]
            }
        }


# -------------------------------------------------------------------
# RESPONSE FORMATTER (Enhanced)
# -------------------------------------------------------------------
class ResponseFormatter:
    """Format responses for different intents."""
    
    @staticmethod
    def safe_format(value, format_spec=".2f", default="N/A"):
        """Safely format numbers, handling None, NaN, and Infinity."""
        if value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return default
                if format_spec.endswith('%'):
                    return f"{value * 100:{format_spec[:-1]}}%"
                return f"{value:{format_spec}}"
            return str(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def format_optimize_response(plan_summary, query):
        """Format optimization plan into natural language."""
        if not plan_summary.get('success'):
            return f"I couldn't generate an optimization plan: {plan_summary.get('error')}"
        
        suppliers = plan_summary['top_suppliers_by_risk'][:5]
        config = plan_summary.get('configuration', {})
        data_summary = plan_summary.get('data_summary', {})
        
        response = [
            f"Here's your logistics optimization plan:",
            ""
        ]
        
        # Data quality
        quality_score = plan_summary.get('data_quality_score', 0.0)
        response.append(f"Data Quality Score: {ResponseFormatter.safe_format(quality_score, '.1%', '0%')}")
        response.append(f"Total Shipments Analyzed: {data_summary.get('total_rows', 0)}")
        response.append("")
        
        # Optimization details
        response.append("Optimization Configuration:")
        response.append(f"  Solver: {plan_summary['solver']}")
        response.append(f"  Suppliers Evaluated: {plan_summary['total_suppliers']}")
        response.append(f"  Risk Weight: {ResponseFormatter.safe_format(config.get('risk_weight', 10.0), '.1f')}")
        response.append(f"  Vehicle Capacity: {ResponseFormatter.safe_format(config.get('vehicle_capacity', 10000.0), '.1f')} kg")
        response.append("")
        
        # ML Model performance
        ml_metrics = plan_summary.get('ml_metrics')
        if ml_metrics:
            response.append("ML Model Performance:")
            response.append(f"  Model Type: {ml_metrics.get('model_type', 'N/A')}")
            response.append(f"  Prediction Accuracy: {ResponseFormatter.safe_format(ml_metrics.get('accuracy'), '.1%')}")
            if ml_metrics.get('roc_auc'):
                response.append(f"  ROC-AUC Score: {ResponseFormatter.safe_format(ml_metrics['roc_auc'], '.3f')}")
            response.append("")
        
        # Top suppliers by risk (lowest first)
        response.append("Top 5 Suppliers (ranked by risk, lowest first):")
        for i, s in enumerate(suppliers, 1):
            score = plan_summary['risk_scores'].get(s, 0.5)
            stats = plan_summary.get('supplier_stats_preview', {}).get(s, {})
            on_time = stats.get('on_time_rate')
            alloc_count = plan_summary['allocation_summary'].get(s, 0)
            
            if score < 0.3:
                risk_label = "Low Risk"
            elif score < 0.7:
                risk_label = "Medium Risk"
            else:
                risk_label = "High Risk"
            
            response.append(f"  {i}. {s} - Risk: {ResponseFormatter.safe_format(score, '.3f')} ({risk_label})")
            if on_time is not None:
                response.append(f"     On-time Rate: {ResponseFormatter.safe_format(on_time, '.1%')}")
            response.append(f"     Assigned Shipments: {alloc_count}")
        
        response.append("")
        response.append("Download the full plan for complete routing and allocation details.")
        
        return "\n".join(response)
    
    @staticmethod
    def format_risk_response(artifact_results, query, top_n=5):
        """Format risk scores."""
        combined_risk = artifact_results.get('risk_scores', {}).get('combined', {})
        stat_risk = artifact_results.get('risk_scores', {}).get('statistical', {})
        ml_risk = artifact_results.get('risk_scores', {}).get('ml_based', {})
        survival_risk = artifact_results.get('risk_scores', {}).get('survival_based', {})
        
        suppliers = list(combined_risk.keys())
        sorted_suppliers = sorted(suppliers, key=lambda s: combined_risk.get(s, 0.5))
        
        response = [
            "Supplier Risk Assessment",
            "",
            "Risk Scores (0=low risk, 1=high risk):",
            ""
        ]
        
        for i, s in enumerate(sorted_suppliers[:top_n], 1):
            score = combined_risk.get(s, 0.5)
            if score < 0.3:
                risk_label = "Low Risk"
            elif score < 0.7:
                risk_label = "Medium Risk"
            else:
                risk_label = "High Risk"
            
            response.append(f"{i}. {s}")
            response.append(f"   Combined Score: {ResponseFormatter.safe_format(score, '.3f')} ({risk_label})")
            
            if stat_risk:
                response.append(f"   Statistical Model: {ResponseFormatter.safe_format(stat_risk.get(s), '.3f')}")
            if ml_risk and ml_risk.get(s) is not None:
                response.append(f"   ML Model: {ResponseFormatter.safe_format(ml_risk.get(s), '.3f')}")
            if survival_risk and survival_risk.get(s) is not None:
                response.append(f"   Survival Model: {ResponseFormatter.safe_format(survival_risk.get(s), '.3f')}")
            response.append("")
        
        return "\n".join(response)
    
    @staticmethod
    def format_supplier_info(details):
        """Format comprehensive supplier details."""
        if not details:
            return "Supplier information not available."
        
        stats = details.get('statistics', {})
        risk = details.get('risk_scores', {})
        attrs = details.get('attributes', {})
        allocation = details.get('allocation', {})
        routing = details.get('routing', {})
        
        response = [
            f"Supplier Profile: {details['supplier_name']}",
            ""
        ]
        
        # Risk assessment
        combined_risk = risk.get('combined', 0.5)
        if combined_risk < 0.3:
            risk_label = "Low Risk"
        elif combined_risk < 0.7:
            risk_label = "Medium Risk"
        else:
            risk_label = "High Risk"
        response.append(f"Risk Score: {ResponseFormatter.safe_format(combined_risk, '.3f')} ({risk_label})")
        response.append("")
        
        # Performance metrics
        response.append("Performance Metrics:")
        if stats.get('total_shipments'):
            response.append(f"  Total Shipments: {ResponseFormatter.safe_format(stats['total_shipments'], '.0f')}")
        if stats.get('on_time_rate') is not None:
            response.append(f"  On-Time Rate: {ResponseFormatter.safe_format(stats['on_time_rate'], '.1%')}")
        if stats.get('delay_rate') is not None:
            response.append(f"  Delay Rate: {ResponseFormatter.safe_format(stats['delay_rate'], '.1%')}")
        if stats.get('avg_delay_hours') is not None:
            response.append(f"  Average Delay: {ResponseFormatter.safe_format(stats['avg_delay_hours'], '.1f')} hours")
        if stats.get('defect_rate') is not None:
            response.append(f"  Defect Rate: {ResponseFormatter.safe_format(stats['defect_rate'], '.1%')}")
        response.append("")
        
        # Cost & efficiency
        response.append("Cost & Efficiency:")
        if stats.get('avg_cost_usd'):
            response.append(f"  Average Cost: ${ResponseFormatter.safe_format(stats['avg_cost_usd'], '.2f')}")
        if stats.get('cost_per_kg'):
            response.append(f"  Cost per kg: ${ResponseFormatter.safe_format(stats['cost_per_kg'], '.2f')}")
        if stats.get('cost_per_km'):
            response.append(f"  Cost per km: ${ResponseFormatter.safe_format(stats['cost_per_km'], '.2f')}")
        response.append("")
        
        # Volume
        response.append("Volume Handled:")
        if stats.get('total_weight_kg'):
            response.append(f"  Total Weight: {ResponseFormatter.safe_format(stats['total_weight_kg'], '.1f')} kg")
        if stats.get('avg_weight_kg'):
            response.append(f"  Average Weight per Shipment: {ResponseFormatter.safe_format(stats['avg_weight_kg'], '.1f')} kg")
        if stats.get('total_volume_m3'):
            response.append(f"  Total Volume: {ResponseFormatter.safe_format(stats['total_volume_m3'], '.2f')} m³")
        response.append("")
        
        # Special handling
        if stats.get('temp_controlled_pct') or stats.get('hazardous_pct'):
            response.append("Special Handling Capabilities:")
            if stats.get('temp_controlled_pct'):
                response.append(f"  Temperature-Controlled: {ResponseFormatter.safe_format(stats['temp_controlled_pct'], '.1%')}")
            if stats.get('hazardous_pct'):
                response.append(f"  Hazardous Materials: {ResponseFormatter.safe_format(stats['hazardous_pct'], '.1%')}")
            response.append("")
        
        # Current allocation
        response.append("Current Allocation:")
        response.append(f"  Assigned Shipments: {allocation.get('assigned_shipments', 0)}")
        response.append(f"  Active Routes: {routing.get('total_routes', 0)}")
        
        return "\n".join(response)
    
    @staticmethod
    def format_data_quality_response(artifact_results):
        """Format data quality information."""
        quality = artifact_results.get('data_quality', {})
        
        if not quality:
            return "Data quality information not available."
        
        score = quality.get('overall_quality_score', 0.0)
        completeness = quality.get('column_completeness', {})
        issues = quality.get('data_issues', [])
        
        response = [
            "Data Quality Report",
            "",
            f"Overall Quality Score: {ResponseFormatter.safe_format(score, '.1%')}",
            f"Total Rows: {quality.get('total_rows', 0)}",
            f"Total Columns: {quality.get('total_columns', 0)}",
            ""
        ]
        
        if issues:
            response.append("Issues Detected:")
            for issue in issues:
                response.append(f"  - {issue}")
            response.append("")
        
        response.append("Column Completeness (top 10):")
        sorted_cols = sorted(
            completeness.items(),
            key=lambda x: x[1].get('completeness', 0),
            reverse=True
        )[:10]
        
        for col_type, info in sorted_cols:
            comp = info.get('completeness', 0)
            response.append(f"  {col_type}: {ResponseFormatter.safe_format(comp, '.1%')} ({info.get('column_name')})")
        
        return "\n".join(response)
    
    @staticmethod
    def format_routes_response(artifact_results, supplier_query=None):
        """Format routing information."""
        routes = artifact_results.get('routes', {})
        
        if supplier_query:
            supplier_routes = routes.get(supplier_query, [])
            if not supplier_routes:
                return f"No routes found for {supplier_query}"
            
            response = [
                f"Routes for {supplier_query}",
                "",
                f"Total Routes: {len(supplier_routes)}",
                ""
            ]
            
            for i, route in enumerate(supplier_routes[:5], 1):
                response.append(f"Route {i}: {len(route)} stops")
                response.append(f"  Shipments: {', '.join(map(str, route[:10]))}")
                if len(route) > 10:
                    response.append(f"  ... and {len(route) - 10} more")
                response.append("")
            
            if len(supplier_routes) > 5:
                response.append(f"... and {len(supplier_routes) - 5} more routes")
            
            return "\n".join(response)
        
        else:
            # Summary for all suppliers
            response = [
                "Routing Summary",
                ""
            ]
            
            sorted_suppliers = sorted(
                routes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            
            total_routes = sum(len(r) for r in routes.values())
            response.append(f"Total Routes: {total_routes}")
            response.append("")
            response.append("Routes per Supplier (top 10):")
            
            for supplier, supplier_routes in sorted_suppliers:
                total_stops = sum(len(route) for route in supplier_routes)
                response.append(f"  {supplier}: {len(supplier_routes)} routes, {total_stops} stops")
            
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
        return genai.GenerativeModel("gemini-2.0-flash-exp")
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
        models_dir = current_app.config.get('MODELS_FOLDER', os.path.join(os.getcwd(), 'models/supplier'))
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
        "You are an AI assistant specialized in logistics, supply chain management, supplier risk assessment, and routing optimization.",
        "You have access to detailed supplier performance data, risk scores, and optimization plans.",
        "Answer the user's question based on the provided documents and context.",
        "Use specific numbers and metrics when available.",
        "If the information is insufficient, clearly state what's missing.",
        ""
    ]
    
    # Add context
    if context_info:
        blocks.append("CONTEXT:")
        if context_info.get('data_quality'):
            blocks.append(f"Data Quality Score: {context_info['data_quality']:.1%}")
        if context_info.get('latest_plan'):
            blocks.append(f"Latest Plan: {context_info['latest_plan']}")
        if context_info.get('suppliers'):
            blocks.append(f"Suppliers ({len(context_info['suppliers'])}): {', '.join(context_info['suppliers'][:5])}")
        if context_info.get('total_shipments'):
            blocks.append(f"Total Shipments: {context_info['total_shipments']}")
        blocks.append("")
    
    # Add retrieved documents
    if docs:
        blocks.append("RELEVANT DOCUMENTS:")
        for i, d in enumerate(docs):
            snippet = d['text'][:2000]
            blocks.append(f"--- Document {i+1} (Relevance: {d['score']:.3f}) ---")
            blocks.append(snippet)
            blocks.append("")
    
    blocks.append(f"USER QUESTION:\n{query}\n")
    blocks.append("Provide a clear, data-driven answer with specific metrics:")
    
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
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/supplier')
        
        # 3. Get latest artifact for context
        artifact_info = ArtifactManager.get_latest_artifact(models_dir)
        
        if not artifact_info:
            return jsonify({
                'success': False,
                'error': 'No optimization plans available. Please upload data and run the pipeline first.'
            }), 404
        
        results, metadata = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
        
        # 4. Handle optimize intent
        if intent == 'optimize':
            plan_summary = PlanGenerator.generate_plan_summary(results, metadata)
            
            if not plan_summary.get('success'):
                return jsonify({
                    'success': False,
                    'error': plan_summary.get('error')
                }), 500
            
            response_text = ResponseFormatter.format_optimize_response(plan_summary, query)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'plan_data': plan_summary,
                'artifact_info': metadata
            }))
        
        # 5. Handle risk intent
        if intent == 'risk':
            top_n = EntityExtractor.extract_top_n(query)
            response_text = ResponseFormatter.format_risk_response(results, query, top_n)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'risk_data': results.get('risk_scores', {}),
                'artifact_info': metadata
            }))
        
        # 6. Handle supplier_info intent
        if intent == 'supplier_info':
            suppliers = results.get('suppliers', [])
            supplier_query = EntityExtractor.extract_suppliers(query, suppliers)
            
            if not supplier_query:
                supplier_query = suppliers[0] if suppliers else None
            else:
                supplier_query = supplier_query[0]
            
            if not supplier_query:
                return jsonify({
                    'success': False,
                    'error': 'No supplier specified or found.'
                }), 400
            
            details = PlanGenerator.query_supplier_details(results, supplier_query)
            response_text = ResponseFormatter.format_supplier_info(details)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'supplier_details': details,
                'artifact_info': metadata
            }))
        
        # 7. Handle data_quality intent
        if intent == 'data_quality':
            response_text = ResponseFormatter.format_data_quality_response(results)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'data_quality': results.get('data_quality', {}),
                'artifact_info': metadata
            }))
        
        # 8. Handle routes intent
        if intent == 'routes':
            suppliers = results.get('suppliers', [])
            supplier_query = EntityExtractor.extract_suppliers(query, suppliers)
            supplier_query = supplier_query[0] if supplier_query else None
            
            response_text = ResponseFormatter.format_routes_response(results, supplier_query)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'routes_data': PlanGenerator.query_routes(results, supplier_query),
                'artifact_info': metadata
            }))
        
        # 9. For other intents, use RAG
        vs = get_vector_store()
        
        if vs is None:
            # Fallback: Use Gemini without RAG
            model = init_gemini()
            if model is None:
                return jsonify({
                    'success': False,
                    'error': 'No AI services available'
                }), 500
            
            prompt = f"Based on logistics and supply chain knowledge, answer: {query}"
            response = model.generate_content(prompt)
            
            return jsonify(sanitize_for_json({
                'success': True,
                'answer': response.text,
                'intent': intent,
                'mode': 'direct_llm'
            }))
        
        # Semantic search
        docs = semantic_search(vs, query, k=5)
        
        # Build context
        context_info = {
            'data_quality': results.get('data_quality', {}).get('overall_quality_score', 0.0),
            'latest_plan': metadata.get('model_name', 'Unknown'),
            'suppliers': results.get('suppliers', [])[:10],
            'total_shipments': metadata.get('data_summary', {}).get('total_rows', 0)
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
        
        return jsonify(sanitize_for_json({
            'success': True,
            'answer': answer,
            'intent': intent,
            'sources': docs,
            'context': context_info,
            'artifact_info': metadata
        }))
    
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
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/supplier')
        artifacts = ArtifactManager.list_artifacts(models_dir)
        
        return jsonify(sanitize_for_json({
            'success': True,
            'plans': [
                {
                    'name': a['name'],
                    'suppliers_count': a['metadata'].get('data_summary', {}).get('total_suppliers', 0),
                    'total_rows': a['metadata'].get('data_summary', {}).get('total_rows', 0),
                    'quality_score': a['metadata'].get('data_summary', {}).get('quality_score', 0),
                    'created': a['metadata'].get('created_at'),
                    'solver': a['metadata'].get('optimization', {}).get('solver', 'Unknown'),
                    'model_version': a['metadata'].get('model_version', '1.0')
                }
                for a in artifacts
            ]
        }))
    except Exception as e:
        logger.exception("Failed to list plans")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/logistics/chat/optimize', methods=['POST'])
def direct_optimize():
    """Direct optimization query endpoint (bypassing NLP)."""
    data = request.get_json() or {}
    plan_name = data.get('plan_name')
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/supplier')
        
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
        
        return jsonify(sanitize_for_json(plan_summary))
    
    except Exception as e:
        logger.exception("Optimization query error")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/logistics/chat/download_plan/<plan_name>', methods=['GET'])
def download_plan(plan_name):
    """Download plan as CSV export."""
    try:
        if not pd:
            return jsonify({'success': False, 'error': 'Pandas not available'}), 500
        
        models_dir = current_app.config.get('MODELS_FOLDER', 'models/supplier')
        artifacts = ArtifactManager.list_artifacts(models_dir)
        
        artifact_info = next(
            (a for a in artifacts if a['name'] == plan_name or a['pkl_file'].startswith(plan_name.replace(' ', '_'))),
            None
        )
        
        if not artifact_info:
            artifact_info = ArtifactManager.get_latest_artifact(models_dir)
        
        if not artifact_info:
            return jsonify({'success': False, 'error': 'Plan not found'}), 404
        
        results, metadata = ArtifactManager.load_artifact(models_dir, artifact_info['pkl_file'])
        
        # Build CSV export
        rows = []
        suppliers = results.get('suppliers', [])
        allocation = results.get('allocation', {}).get('assignments', {})
        risk_scores = results.get('risk_scores', {}).get('combined', {})
        stats = results.get('supplier_statistics', {})
        
        for supplier in suppliers:
            supplier_stats = stats.get(supplier, {})
            row = {
                'Supplier': supplier,
                'Risk_Score': risk_scores.get(supplier, 0.5),
                'Assigned_Shipments': len(allocation.get(supplier, {})),
                'Total_Shipments': supplier_stats.get('total_shipments', 0),
                'On_Time_Rate': supplier_stats.get('on_time_rate', 0),
                'Delay_Rate': supplier_stats.get('delay_rate', 0),
                'Avg_Cost_USD': supplier_stats.get('avg_cost_usd', 0),
                'Cost_Per_Kg': supplier_stats.get('cost_per_kg', 0),
                'Total_Weight_Kg': supplier_stats.get('total_weight_kg', 0),
                'Defect_Rate': supplier_stats.get('defect_rate', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create in-memory CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'logistics_plan_{plan_name.replace(" ", "_")}.csv'
        )
    
    except Exception as e:
        logger.exception("Download failed")
        return jsonify({'success': False, 'error': str(e)}), 500