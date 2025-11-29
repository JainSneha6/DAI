import google.generativeai as genai
import os
from typing import Optional, List, Dict, Union
import csv

# Use env var instead of hardcoded key
GEMINI_API_KEY = "AIzaSyDHqtWn2Ye71A_aCc8udlNyjEpZyf15TBw"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_RECOMMENDATIONS = {
    "Sales, Demand & Financial Forecasting Model": {
        "models": ["Prophet", "ARIMA", "Exponential Smoothing"],
        "description": "Time series forecasting for sales and demand prediction"
    },
    "Pricing & Revenue Optimization Model": {
        "models": ["Linear Regression", "Bayesian Hierarchical Models", "Gradient Boosting"],
        "description": "Optimize pricing and revenue based on market conditions"
    },
    "Marketing ROI & Attribution Model": {
        "models": ["Markov Chain Attribution", "Shapley Value Attribution", "Multi-Touch Attribution Models", "Uplift Modeling"],
        "description": "Measure marketing effectiveness and attribute ROI across channels"
    },
    "Customer Segmentation & Modeling": {
        "models": ["K-Means", "DBSCAN", "Gaussian Mixture Models", "Hierarchical Clustering"],
        "description": "Segment customers based on behavior and characteristics"
    },
    "Customer Value & Retention Model": {
        "models": ["Survival Analysis", "Random Forest", "XGBoost", "Logistic Regression", "LTV Prediction Models"],
        "description": "Predict customer churn probability and lifetime value"
    },
    "Sentiment & Intent NLP Model": {
        "models": ["BERT", "RoBERTa", "DistilBERT", "LSTM Networks", "TextCNN"],
        "description": "Analyze customer sentiment and intent from text data"
    },
    "Inventory & Replenishment Optimization Model": {
        "models": ["Economic Order Quantity (EOQ)", "Multi-Echelon Inventory Optimization", "Reinforcement Learning", "Stochastic Optimization"],
        "description": "Optimize inventory levels and stock management"
    },
    "Logistics & Supplier Risk Model": {
        "models": ["Supply Chain Risk Scoring Models", "Bayesian Networks", "Monte Carlo Simulation", "Decision Trees"],
        "description": "Evaluate logistics performance and supplier risk"
    },  
}


def analyze_columns_with_gemini(columns: List[str], model_type: Optional[str] = None) -> Dict:
    """
    Analyze CSV column names for a given model_type (optional). If model_type is provided,
    the prompt is tailored toward that model; otherwise the assistant recommends the best
    model among the known categories.

    Returns JSON-like dict with:
      - model_type: chosen/used model category
      - target_column
      - explanation
      - key_features
      - suggested_models: list of concrete model algorithms (from MODEL_RECOMMENDATIONS)
    """
    try:
        if not columns:
            raise ValueError("columns list is required and cannot be empty")
        columns_text = ", ".join(columns)

        # Validate model_type or build list of available models
        if model_type and model_type not in MODEL_RECOMMENDATIONS:
            raise ValueError(f"Unknown model_type '{model_type}'. Must be one of: {list(MODEL_RECOMMENDATIONS.keys())}")

        if model_type:
            category_prompt_header = f"Target analysis for model category: {model_type}.\nUse the following candidate algorithms: {MODEL_RECOMMENDATIONS[model_type]['models']}"
        else:
            category_prompt_header = "You can recommend one of these model categories:\n" + "\n".join(f"- {k}" for k in MODEL_RECOMMENDATIONS.keys())

        prompt = f"""
        {category_prompt_header}

        Analyze these CSV column names and determine:
        1) The best model category (if not pre-specified) OR confirm the specified model category.
        2) The most suitable target column to predict/optimize for this scenario.
        3) A brief explanation of why this model/category is recommended for the given columns.
        4) Top key features from these column names that should be used in the model.
        5) Concrete model choices from the candidate algorithms and brief rationale for each (if multiple).

        Column Names: {columns_text}

        Respond in this exact JSON format:
        {{
            "model_type": "string",
            "target_column": "string",
            "explanation": "string",
            "key_features": ["column1", "column2", ...],
            "suggested_models": ["Algorithm1", "Algorithm2", ...]
        }}
        """

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)

        import json
        import re

        # Extract JSON from response
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = json.loads(response_text)

        # If Gemini returns a model_type not in mapping, keep it but don't fail
        recommended_models = MODEL_RECOMMENDATIONS.get(analysis.get('model_type'))
        if not recommended_models and model_type:
            # If the user forced model_type and Gemini returns different, use user-provided
            recommended_models = MODEL_RECOMMENDATIONS.get(model_type)
            
        print("Gemini Analysis Result:", analysis)

        return {
            'success': True,
            'analysis': analysis,
            'model_recommendations': recommended_models or MODEL_RECOMMENDATIONS['Sales, Demand & Financial Forecasting Model']
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def extract_columns_from_csv_file(file_path: str) -> List[str]:
    """Return the CSV header columns for a file path. If not CSV or empty return []"""
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            # Use Sniffer to guess delimiter and header
            sample = csvfile.read(2048)
            csvfile.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            reader = csv.reader(csvfile, dialect)
            if has_header:
                header = next(reader, [])
                return [col.strip() for col in header if col is not None and str(col).strip()]
            else:
                # fallback: treat first row as header
                header = next(reader, [])
                return [col.strip() for col in header if col is not None and str(col).strip()]
    except Exception:
        return []

def analyze_file_with_gemini(file_path: str, model_type: Optional[str] = None) -> Dict:
    """
    Read a CSV file, extract header columns, and call analyze_columns_with_gemini.
    """
    try:
        columns = extract_columns_from_csv_file(file_path)
        if not columns:
            return {
                "success": False,
                "error": "Unable to extract CSV columns from file"
            }
        return analyze_columns_with_gemini(columns, model_type)
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_available_models() -> dict:
    """Get all available model recommendations"""
    return MODEL_RECOMMENDATIONS