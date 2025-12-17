"""
Backend API for Model Details Pages - Integrated with Pipeline Artifacts
Uses actual metadata and model files created by the pipelines
"""

from flask import Blueprint, jsonify, request, send_file
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)


# ============================
# Helper Functions
# ============================

def get_models_directory() -> str:
    """Get the models directory path from app config."""
    from flask import current_app
    return current_app.config.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))


def get_upload_folder() -> str:
    """Get upload folder path."""
    from flask import current_app
    return current_app.config.get("UPLOAD_FOLDER", "uploads")


def load_json_metadata(filepath: str) -> Dict:
    """Load JSON metadata file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load JSON metadata {filepath}: {e}")
        return {}


def load_pickle_artifact(filepath: str) -> Any:
    """Load a pickled model artifact."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to load pickle artifact {filepath}: {e}")
        return None


def find_model_artifacts_for_file(filename: str) -> Dict[str, List[Dict]]:
    """
    Find all model artifacts (.pkl) and their metadata (.meta.json) for a given file.
    Scans domain-specific subdirectories under models/
    Returns: {model_type: [{"pkl_path": ..., "meta_path": ..., "metadata": {...}, "timestamp": ...}, ...]}
    """
    from flask import current_app
    base_models_dir = current_app.config.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
    
    if not os.path.exists(base_models_dir):
        return {}
    
    # Group artifacts by model type
    artifacts_by_type: Dict[str, List[Dict]] = {}
    
    # Scan all subdirectories (sales, inventory, marketing, etc.)
    try:
        for domain_dir in os.listdir(base_models_dir):
            domain_path = os.path.join(base_models_dir, domain_dir)
            
            # Skip non-directories and special folders
            if not os.path.isdir(domain_path) or domain_dir.startswith('.') or domain_dir == 'cyborg_indexes':
                continue
            
            # Scan files in this domain directory
            try:
                files_in_domain = os.listdir(domain_path)
            except Exception:
                continue
            
            for fname in files_in_domain:
                if not fname.endswith(".pkl"):
                    continue
                
                pkl_path = os.path.join(domain_path, fname)
                
                # Get corresponding metadata
                base = os.path.splitext(fname)[0]
                meta_file = f"{base}.meta.json"
                meta_path = os.path.join(domain_path, meta_file)
                
                if not os.path.exists(meta_path):
                    continue
                
                # Load metadata
                metadata = load_json_metadata(meta_path)
                
                # Check if this artifact is for the requested file
                input_file = metadata.get("input_file") or metadata.get("file_name")
                
                # If metadata doesn't specify file, or matches our file
                if input_file == filename or input_file is None:
                    
                    # Determine model type from metadata or filename
                    model_type = metadata.get("model") or metadata.get("model_name")
                    
                    if not model_type:
                        # Try to infer from filename patterns
                        if "SARIMAX" in fname or "Prophet" in fname:
                            model_type = "Sales, Demand & Financial Forecasting Model"
                        elif "Marketing" in fname or "ElasticNetCV" in fname or "RidgeCV" in fname:
                            model_type = "Marketing ROI & Attribution Model"
                        elif "Inventory" in fname:
                            model_type = "Inventory & Replenishment Optimization Model"
                        elif "Supplier" in fname or "Routing" in fname:
                            model_type = "Logistics & Supplier Risk Model"
                        elif "Customer" in fname or "Segmentation" in fname:
                            model_type = "Customer Segmentation & Modeling"
                        else:
                            model_type = "Unknown Model"
                    
                    # Get timestamp
                    created_at = metadata.get("created_at")
                    if created_at:
                        try:
                            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        except:
                            timestamp = datetime.fromtimestamp(os.path.getmtime(pkl_path))
                    else:
                        timestamp = datetime.fromtimestamp(os.path.getmtime(pkl_path))
                    
                    artifact_info = {
                        "pkl_path": pkl_path,
                        "meta_path": meta_path,
                        "metadata": metadata,
                        "timestamp": timestamp,
                        "filename": fname,
                        "domain": domain_dir  # Track which domain directory this is in
                    }
                    
                    artifacts_by_type.setdefault(model_type, [])
                    artifacts_by_type[model_type].append(artifact_info)
    
    except Exception as e:
        logger.exception("Error scanning model directories: %s", e)
    
    # Sort each model type's artifacts by timestamp (newest first)
    for model_type in artifacts_by_type:
        artifacts_by_type[model_type].sort(key=lambda x: x["timestamp"], reverse=True)
    
    return artifacts_by_type


def extract_results_summary(artifact_obj: Any, metadata: Dict, model_type: str) -> Dict:
    """
    Extract key results from the loaded artifact object based on model type.
    Returns a JSON-serializable summary dict.
    """
    summary = {
        "model_type": model_type,
        "created_at": metadata.get("created_at"),
        "available": True
    }
    
    try:
        # The artifact might be a dict with "results" or "pipeline" keys, or the model object itself
        if isinstance(artifact_obj, dict):
            # Try to get results from common keys
            results = artifact_obj.get("results") or artifact_obj.get("pipeline") or artifact_obj
        else:
            results = {"model_object": "Loaded"}
        
        if model_type == "Sales, Demand & Financial Forecasting Model":
            # Time series forecasting
            summary["target_column"] = metadata.get("target_column")
            summary["datetime_column"] = metadata.get("datetime_column")
            summary["exogenous_features"] = metadata.get("exogenous_features", [])
            summary["train_period"] = metadata.get("train_start") + " to " + metadata.get("train_end") if metadata.get("train_start") else None
            
            if isinstance(results, dict):
                best_model = results.get("best_model", {})
                summary["best_model_name"] = best_model.get("name")
                summary["metrics"] = best_model.get("metrics", {})
                summary["forecast_horizon"] = len(best_model.get("future_forecast", []))
        
        elif model_type == "Marketing ROI & Attribution Model":
            # Marketing MMM
            summary["target_column"] = metadata.get("target_column")
            summary["spend_columns"] = metadata.get("spend_columns_detected", [])
            summary["adstock_applied"] = metadata.get("adstock_applied", False)
            
            if isinstance(results, dict):
                best_model = results.get("best_model", {})
                summary["best_model_name"] = best_model.get("name")
                summary["metrics"] = best_model.get("metrics", {})
                
                # ROI by channel
                attribution = best_model.get("attribution", {})
                roi_list = []
                for channel, data in attribution.items():
                    if isinstance(data, dict):
                        roi_list.append({
                            "channel": channel,
                            "roi": data.get("approx_roi"),
                            "spend": data.get("total_spend"),
                            "contribution": data.get("approx_contribution")
                        })
                summary["top_channels"] = sorted(roi_list, key=lambda x: x.get("roi") or 0, reverse=True)[:5]
        
        elif model_type == "Inventory & Replenishment Optimization Model":
            # Inventory optimization
            summary["solver_used"] = metadata.get("solver_used") or results.get("solver_used")
            summary["skus"] = metadata.get("skus", [])
            summary["total_skus"] = len(metadata.get("skus", []))
            
            if isinstance(results, dict):
                eoq = results.get("EOQ", {})
                summary["avg_eoq"] = float(np.mean([v for v in eoq.values() if v])) if eoq else None
                
                schedule = results.get("schedule", {})
                total_orders = sum(len(periods) for periods in schedule.values())
                summary["total_orders_planned"] = total_orders
        
        elif model_type == "Logistics & Supplier Risk Model":
            # Supplier risk
            summary["suppliers"] = metadata.get("suppliers", [])
            summary["solver_used"] = metadata.get("solver_used") or results.get("solver_used")
            
            if isinstance(results, dict):
                risk_scores = results.get("combined_risk", {})
                summary["avg_risk_score"] = float(np.mean([v for v in risk_scores.values() if v])) if risk_scores else None
                
                # High risk suppliers
                high_risk = [(s, v) for s, v in risk_scores.items() if v > 0.6]
                summary["high_risk_suppliers_count"] = len(high_risk)
        
        elif model_type == "Customer Segmentation & Modeling":
            # Customer segmentation
            summary["customers_count"] = results.get("customers_count", 0) if isinstance(results, dict) else 0
            
            if isinstance(results, dict):
                segmentation = results.get("segmentation", {})
                summary["segmentation_methods"] = [k for k in segmentation.keys() if "labels" in k]
                
                predictive = results.get("predictive", {})
                summary["predictive_trained"] = predictive.get("trained", False)
                if predictive.get("metrics"):
                    summary["predictive_auc"] = predictive["metrics"].get("auc")
        
    except Exception as e:
        logger.exception(f"Error extracting results summary: {e}")
        summary["extraction_error"] = str(e)
    
    return summary


# ============================
# API Endpoints
# ============================

@models_bp.route("/api/files/<filename>/models", methods=["GET"])
def get_models_for_file(filename):
    """
    Get all models that have been run on a specific file.
    Returns model history with results summaries.
    Only returns models from the file's domain subdirectory.
    """
    try:
        # First, get the file's metadata to determine its domain
        upload_folder = get_upload_folder()
        meta_dir = os.path.join(upload_folder, "metadata")
        meta_path = os.path.join(meta_dir, f"{filename}.meta.json")
        
        file_domain = None
        if os.path.exists(meta_path):
            try:
                file_meta = load_json_metadata(meta_path)
                file_domain = file_meta.get("data_domain") or file_meta.get("model_directory")
            except Exception:
                pass
        
        # Find artifacts for this file (will scan all domains but we'll filter)
        artifacts_by_type = find_model_artifacts_for_file(filename)
        
        models_list = []
        
        for model_type, artifacts in artifacts_by_type.items():
            if not artifacts:
                continue
            
            # Take the latest artifact for this model type
            latest = artifacts[0]
            
            # Filter: only include if artifact is in the same domain as the file
            artifact_domain = latest.get("domain")
            if file_domain and artifact_domain and artifact_domain != file_domain:
                logger.debug(f"Skipping model {model_type} for {filename} - domain mismatch ({artifact_domain} != {file_domain})")
                continue
            
            metadata = latest["metadata"]
            
            # Try to load artifact and extract summary
            artifact_obj = load_pickle_artifact(latest["pkl_path"])
            results_summary = extract_results_summary(artifact_obj, metadata, model_type)
            
            models_list.append({
                "model_type": model_type,
                "last_run": latest["timestamp"].isoformat(),
                "has_results": True,
                "model_path": latest["pkl_path"],
                "meta_path": latest["meta_path"],
                "domain": artifact_domain,
                "results_summary": results_summary,
                "run_count": len(artifacts)
            })
        
        return jsonify({
            "success": True,
            "filename": filename,
            "file_domain": file_domain,
            "models": models_list
        })
        
    except Exception as e:
        logger.exception("Error getting models for file")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@models_bp.route("/api/models/<model_id>/results/<filename>", methods=["GET"])
def get_model_results(model_id, filename):
    """
    Get detailed results for a specific model and file.
    """
    try:
        # Model ID to name mapping
        model_names = {
            "sales-forecasting": "Sales, Demand & Financial Forecasting Model",
            "marketing-roi": "Marketing ROI & Attribution Model",
            "inventory-optimization": "Inventory & Replenishment Optimization Model",
            "supplier-risk": "Logistics & Supplier Risk Model",
            "customer-segmentation": "Customer Segmentation & Modeling",
        }
        
        model_type = model_names.get(model_id)
        if not model_type:
            return jsonify({
                "success": False,
                "error": "Invalid model ID"
            }), 400
        
        # Find artifacts for this file and model type
        artifacts_by_type = find_model_artifacts_for_file(filename)
        
        if model_type not in artifacts_by_type or not artifacts_by_type[model_type]:
            return jsonify({
                "success": False,
                "error": "No results found for this model and file"
            }), 404
        
        # Get latest artifact
        latest = artifacts_by_type[model_type][0]
        metadata = latest["metadata"]
        
        # Load full artifact
        artifact_obj = load_pickle_artifact(latest["pkl_path"])
        
        if artifact_obj is None:
            return jsonify({
                "success": False,
                "error": "Failed to load model artifact"
            }), 500
        
        # Extract detailed results
        detailed_results = extract_detailed_results(artifact_obj, metadata, model_type)
        
        return jsonify({
            "success": True,
            "model_type": model_type,
            "filename": filename,
            "metadata": metadata,
            "results": detailed_results,
            "artifact_path": latest["pkl_path"]
        })
        
    except Exception as e:
        logger.exception("Error getting model results")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def extract_detailed_results(artifact_obj: Any, metadata: Dict, model_type: str) -> Dict:
    """Extract comprehensive results for frontend display."""
    
    results = {
        "model_type": model_type,
        "created_at": metadata.get("created_at"),
        "available": True
    }
    
    try:
        # Get results dict
        if isinstance(artifact_obj, dict):
            pipeline_results = artifact_obj.get("results") or artifact_obj.get("pipeline") or artifact_obj
        else:
            pipeline_results = {}
        
        if model_type == "Sales, Demand & Financial Forecasting Model":
            results.update({
                "target_column": metadata.get("target_column"),
                "datetime_column": metadata.get("datetime_column"),
                "exogenous_features": metadata.get("exogenous_features", []),
                "train_period": {
                    "start": metadata.get("train_start"),
                    "end": metadata.get("train_end")
                }
            })
            
            if isinstance(pipeline_results, dict):
                # Model comparison
                models_info = pipeline_results.get("models", {})
                results["models_compared"] = []
                for name, info in models_info.items():
                    if isinstance(info, dict) and info.get("metrics"):
                        results["models_compared"].append({
                            "name": name,
                            "metrics": info["metrics"],
                            "selected": name == pipeline_results.get("best_model", {}).get("name")
                        })
                
                # Best model details
                best_model = pipeline_results.get("best_model", {})
                results["best_model"] = {
                    "name": best_model.get("name"),
                    "metrics": best_model.get("metrics", {}),
                    "forecast": best_model.get("future_forecast", [])[:20],  # First 20 forecast points
                    "forecast_full_length": len(best_model.get("future_forecast", []))
                }
        
        elif model_type == "Marketing ROI & Attribution Model":
            results.update({
                "target_column": metadata.get("target_column"),
                "spend_columns": metadata.get("spend_columns_detected", []),
                "feature_names": metadata.get("feature_names", []),
                "adstock_settings": {
                    "applied": metadata.get("adstock_applied", False),
                    "decay": metadata.get("adstock_decay"),
                    "saturation_alpha": metadata.get("saturation_alpha")
                }
            })
            
            if isinstance(pipeline_results, dict):
                # Model comparison
                test_metrics = pipeline_results.get("test_metrics", {})
                results["models_compared"] = []
                for name, metrics in test_metrics.items():
                    results["models_compared"].append({
                        "name": name,
                        "metrics": metrics,
                        "selected": name == pipeline_results.get("best_model", {}).get("name")
                    })
                
                # Attribution results
                best_model = pipeline_results.get("best_model", {})
                attribution = best_model.get("attribution", {})
                
                results["attribution"] = []
                for channel, data in attribution.items():
                    if isinstance(data, dict):
                        results["attribution"].append({
                            "channel": channel,
                            "total_spend": data.get("total_spend"),
                            "contribution": data.get("approx_contribution"),
                            "roi": data.get("approx_roi"),
                            "coefficient": data.get("coef")
                        })
                
                results["attribution"].sort(key=lambda x: x.get("roi") or 0, reverse=True)
        
        elif model_type == "Inventory & Replenishment Optimization Model":
            results.update({
                "solver_used": metadata.get("solver_used") or pipeline_results.get("solver_used"),
                "solver_status": pipeline_results.get("solver_status"),
                "skus": metadata.get("skus", [])
            })
            
            if isinstance(pipeline_results, dict):
                # EOQ/POQ results
                eoq = pipeline_results.get("EOQ", {})
                poq = pipeline_results.get("POQ", {})
                
                results["sku_analysis"] = []
                for sku in metadata.get("skus", [])[:50]:  # First 50 SKUs
                    results["sku_analysis"].append({
                        "sku": sku,
                        "eoq": eoq.get(sku),
                        "poq": poq.get(sku),
                        "demand_rate": pipeline_results.get("demand_rate_per_period", {}).get(sku),
                        "ordering_cost": pipeline_results.get("ordering_costs", {}).get(sku),
                        "holding_cost": pipeline_results.get("holding_costs", {}).get(sku)
                    })
                
                # Schedule summary
                schedule = pipeline_results.get("schedule", {})
                results["schedule_summary"] = {
                    "total_skus_with_orders": len([s for s, periods in schedule.items() if periods]),
                    "total_orders": sum(len(periods) for periods in schedule.values()),
                    "total_quantity": sum(sum(periods.values()) for periods in schedule.values())
                }
        
        elif model_type == "Logistics & Supplier Risk Model":
            results.update({
                "suppliers": metadata.get("suppliers", []),
                "solver_used": metadata.get("solver_used") or pipeline_results.get("solver_used"),
                "risk_weight": metadata.get("risk_weight")
            })
            
            if isinstance(pipeline_results, dict):
                # Risk scores
                combined_risk = pipeline_results.get("combined_risk", {})
                stats = pipeline_results.get("stats", {})
                
                results["supplier_analysis"] = []
                for supplier in metadata.get("suppliers", []):
                    supp_stats = stats.get(supplier, {})
                    results["supplier_analysis"].append({
                        "supplier": supplier,
                        "risk_score": combined_risk.get(supplier),
                        "total_supply": supp_stats.get("total_supply"),
                        "avg_lead_time": supp_stats.get("avg_lead_time"),
                        "on_time_rate": supp_stats.get("on_time_rate"),
                        "defect_rate": supp_stats.get("defect_rate"),
                        "unit_cost": pipeline_results.get("supplier_unit_cost", {}).get(supplier)
                    })
                
                results["supplier_analysis"].sort(key=lambda x: x.get("risk_score") or 0, reverse=True)
                
                # Allocation summary
                allocation = pipeline_results.get("allocation", {})
                results["allocation_summary"] = {
                    "suppliers_used": len([s for s, alloc in allocation.items() if alloc]),
                    "total_allocated": sum(sum(rows.values()) for rows in allocation.values())
                }
        
        elif model_type == "Customer Segmentation & Modeling":
            results.update({
                "customers_count": pipeline_results.get("customers_count", 0) if isinstance(pipeline_results, dict) else 0,
                "segmentation_methods": metadata.get("segmentation_methods", [])
            })
            
            if isinstance(pipeline_results, dict):
                segmentation = pipeline_results.get("segmentation", {})
                
                # Segment distribution
                results["segment_distribution"] = {}
                for method, labels_dict in segmentation.items():
                    if "labels" in method and isinstance(labels_dict, dict):
                        method_name = method.replace("_labels", "")
                        counts = {}
                        for customer, segment in labels_dict.items():
                            if segment is not None:
                                counts[int(segment)] = counts.get(int(segment), 0) + 1
                        results["segment_distribution"][method_name] = counts
                
                # Predictive model results
                predictive = pipeline_results.get("predictive", {})
                results["predictive"] = {
                    "trained": predictive.get("trained", False),
                    "metrics": predictive.get("metrics", {}),
                    "high_risk_customers_count": len([v for v in predictive.get("churn_probability_per_customer", {}).values() if v > 0.7]) if predictive.get("churn_probability_per_customer") else 0
                }
                
                # Sample customer summary
                results["customer_sample"] = pipeline_results.get("customer_summary_sample", [])[:10]
    
    except Exception as e:
        logger.exception(f"Error extracting detailed results: {e}")
        results["extraction_error"] = str(e)
    
    return results


@models_bp.route("/api/models/run", methods=["POST"])
def run_model():
    """
    Trigger a model run on a file.
    This re-runs the pipeline and creates new artifacts.
    """
    try:
        data = request.json
        filename = data.get("filename")
        model_type = data.get("model_type")
        
        if not filename or not model_type:
            return jsonify({
                "success": False,
                "error": "Missing filename or model_type"
            }), 400
        
        upload_folder = get_upload_folder()
        file_path = os.path.join(upload_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "File not found"
            }), 404
        
        # Import pipeline functions
        from services.gemini_analyzer import analyze_file_with_gemini, trigger_models_for_file
        
        # Analyze file
        gemini_response = analyze_file_with_gemini(file_path)
        
        if not gemini_response.get("success"):
            return jsonify({
                "success": False,
                "error": "Failed to analyze file"
            }), 500
        
        # Override model_type to force specific model
        gemini_response["analysis"]["model_type"] = model_type
        
        # Trigger model
        models_dir = get_models_directory()
        result = trigger_models_for_file(file_path, gemini_response, models_dir=models_dir)
        
        if result.get("success"):
            # Extract results for the requested model
            runners = result.get("runners", {})
            runner_result = runners.get(model_type, {})
            
            if runner_result.get("success"):
                return jsonify({
                    "success": True,
                    "message": "Model executed successfully",
                    "model_type": model_type,
                    "filename": filename
                })
            else:
                return jsonify({
                    "success": False,
                    "error": runner_result.get("error", "Model execution failed")
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": "Failed to trigger model"
            }), 500
        
    except Exception as e:
        logger.exception("Error running model")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@models_bp.route("/api/models/<model_id>/files", methods=["GET"])
def get_files_for_model(model_id):
    """
    Get all files that have been processed with this model type.
    """
    try:
        model_names = {
            "sales-forecasting": "Sales, Demand & Financial Forecasting Model",
            "marketing-roi": "Marketing ROI & Attribution Model",
            "inventory-optimization": "Inventory & Replenishment Optimization Model",
            "supplier-risk": "Logistics & Supplier Risk Model",
            "customer-segmentation": "Customer Segmentation & Modeling",
        }
        
        model_type = model_names.get(model_id)
        if not model_type:
            return jsonify({
                "success": False,
                "error": "Invalid model ID"
            }), 400
        
        models_dir = get_models_directory()
        
        # Find all artifacts of this model type
        files_with_models = {}
        
        for fname in os.listdir(models_dir):
            if not fname.endswith(".meta.json"):
                continue
            
            meta_path = os.path.join(models_dir, fname)
            metadata = load_json_metadata(meta_path)
            
            detected_model = metadata.get("model") or metadata.get("model_name")
            if detected_model and detected_model.lower() == model_type.lower():
                input_file = metadata.get("input_file")
                created_at = metadata.get("created_at")
                
                if input_file:
                    if input_file not in files_with_models:
                        files_with_models[input_file] = {
                            "filename": input_file,
                            "last_run": created_at,
                            "has_results": True,
                            "run_count": 0
                        }
                    files_with_models[input_file]["run_count"] += 1
                    
                    # Keep latest run time
                    if created_at and (not files_with_models[input_file]["last_run"] or created_at > files_with_models[input_file]["last_run"]):
                        files_with_models[input_file]["last_run"] = created_at
        
        return jsonify({
            "success": True,
            "model_type": model_type,
            "files": list(files_with_models.values())
        })
        
    except Exception as e:
        logger.exception("Error getting files for model")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500