import google.generativeai as genai
import os

GEMINI_API_KEY = "AIzaSyDHqtWn2Ye71A_aCc8udlNyjEpZyf15TBw"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_RECOMMENDATIONS = {
    "Sales & Demand Forecasting": {
        "models": ["Prophet", "ARIMA", "Exponential Smoothing"],
        "description": "Time series forecasting for sales and demand prediction"
    },
    "Pricing & Revenue Optimization": {
        "models": ["Linear Regression", "Bayesian Hierarchical Models", "Gradient Boosting"],
        "description": "Optimize pricing and revenue based on market conditions"
    },
    "Customer Segmentation": {
        "models": ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"],
        "description": "Segment customers based on behavior and characteristics"
    },
    "Churn Prediction": {
        "models": ["Logistic Regression", "Random Forest", "XGBoost"],
        "description": "Predict customer churn probability"
    },
    "Inventory Optimization": {
        "models": ["Prophet", "ARIMA", "Seasonal Decomposition"],
        "description": "Optimize inventory levels and stock management"
    },
    "Profitability Analysis": {
        "models": ["Multiple Linear Regression", "Random Forest Regression"],
        "description": "Analyze and predict profit margins and profitability"
    }
}

def analyze_columns_with_gemini(columns: list) -> dict:
    """
    Analyze CSV columns using Gemini API to recommend model type and target column
    """
    try:
        columns_text = ", ".join(columns)
        
        prompt = f"""
        Analyze these CSV column names and determine the best predictive model type and target column for analysis.
        
        Column Names: {columns_text}
        
        Based on these columns, provide:
        1. The most appropriate model type from this list:
           - Sales, Demand & Financial Forecasting Model
           - Pricing & Revenue Optimization
           - Customer Segmentation
           - Churn Prediction
           - Inventory Optimization
           - Profitability Analysis
        
        2. The most suitable target column to predict/optimize
        3. A brief explanation of why this model is recommended
        4. Key features that should be used for the model
        
        Respond in this exact JSON format:
        {{
            "model_type": "string",
            "target_column": "string",
            "explanation": "string",
            "key_features": ["column1", "column2", ...]
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
        
        print(analysis)
        
        return {
            'success': True,
            'analysis': analysis,
            'model_recommendations': MODEL_RECOMMENDATIONS.get(
                analysis.get('model_type'),
                MODEL_RECOMMENDATIONS['Sales & Demand Forecasting']
            )
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_available_models() -> dict:
    """Get all available model recommendations"""
    return MODEL_RECOMMENDATIONS