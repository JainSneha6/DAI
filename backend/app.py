from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import pandas as pd
import json
import numpy as np

# Initialize ChromaDB client
client = chromadb.Client()

insight_collection = client.create_collection(name="business_insights")
analysis_collection = client.create_collection(name="business_analysis")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Generative AI integration
api_key = "AIzaSyDHqtWn2Ye71A_aCc8udlNyjEpZyf15TBw"  # Replace with actual API key
llm = ChatGoogleGenerativeAI(api_key=api_key, model='gemini-pro')

# Global variable to store summary of uploaded data
last_uploaded_data_summary = {}

# Function to analyze CSV data
def analyze_csv(data):
    """Analyzes the uploaded CSV data and returns summary statistics."""
    summary = {
        'columns': data.columns.tolist(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'sample_rows': data.head().to_dict(orient='records')
    }
    return summary

# Route to upload business data
@app.route('/api/upload-business-data', methods=['POST'])
def upload_business_data():
    global last_uploaded_data_summary
    try:
        business_name = request.form.get("businessName")
        industry = request.form.get("industry")

        # Store uploaded files in a dictionary
        uploaded_files = {
            'salesData': request.files.get('salesData'),
            'customerData': request.files.get('customerData'),
            'inventoryData': request.files.get('inventoryData'),
            'marketingCampaignsData': request.files.get('marketingCampaignsData')
        }

        file_summaries = {}
        for file_key, file in uploaded_files.items():
            if file:
                file_extension = file.filename.split('.')[-1]
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)  # Save the file

                # Process based on file type
                if file_extension == "csv":
                    data = pd.read_csv(file_path)
                    summary = analyze_csv(data)
                elif file_extension == "xlsx":
                    data = pd.read_excel(file_path)
                    summary = analyze_csv(data)
                elif file_extension == "json":
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                    summary = {"message": f"JSON data for {file_key} uploaded"}
                else:
                    return jsonify({"error": f"Unsupported file format for {file_key}"}), 400

                # Store summary for each file
                file_summaries[file_key] = summary

                # Add data to ChromaDB
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict(orient='records')

                for i, row in enumerate(data):
                    insight_id = f"{business_name}_{file_key}_{i}"
                    insight_text = json.dumps(row)
                    insight_collection.add(ids=[insight_id], documents=[insight_text])

        last_uploaded_data_summary = file_summaries
        return jsonify({"message": "Files uploaded and data added successfully!", "summary": file_summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Route to search for insights
@app.route('/search-insight', methods=['POST'])
def search_insight():
    try:
        query = request.json.get("query")
        # Increase search results to capture more relevant context
        results = insight_collection.search(query_texts=[query], n_results=10)

        return jsonify({"results": results["documents"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/consultant-query', methods=['POST'])
def consultant_query():
    try:
        query = request.json.get("query")
        search_results = insight_collection.query(query_texts=[query], n_results=20)

        documents = search_results["documents"][0]

        parsed_documents = []
        raw_data = []
        for doc in documents:
            try:
                parsed_doc = json.loads(doc)
                raw_data.append(parsed_doc)
                formatted_doc = json.dumps(parsed_doc, indent=2)
                parsed_documents.append(formatted_doc)
            except json.JSONDecodeError:
                parsed_documents.append(doc)

        data_df = pd.DataFrame(raw_data)

        # Construct context
        context = "\n\n".join(parsed_documents)

        prompt_template = """
        You are a highly knowledgeable virtual business consultant with access to comprehensive data about the company. 
        Your task is to always provide insightful, accurate, and helpful answers based on the data provided. The data is always sufficient to derive an answer. 
        You must:
        1. Analyze the context thoroughly.
        2. Use the provided data to support your answer.
        3. If the data is incomplete or missing, make intelligent assumptions based on common business knowledge or trends.
        4. Always give a well-reasoned answer, even if the data appears insufficient. 
        5. Answer the question as if you had enough data, ensuring that it addresses all aspects of the question.
        6. Be concise but informative. Provide actionable insights or strategies where applicable.

        Context: {context}
        Question: {query}
        """


        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain
        answer = chain.run({"context": context, "query": query})

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# Helper function to classify columns by data type
def classify_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number, np.datetime64]).columns.tolist()
    return numeric_columns, categorical_columns

@app.route('/api/analyze-data', methods=['POST'])
def analyze_data():
    global last_analysis_summary  # Keep track of the last analysis summary
    try:
        data_type = request.json.get('dataType')  # Get the type of data (sales, customer, etc.)
        
        # Use the appropriate file based on data type
        file_mapping = {
            'salesData': 'uploads/SalesData.csv',
            'customerData': 'uploads/CustomerData.csv',
            'inventoryData': 'uploads/InventoryData.csv',
            'marketingCampaignsData': 'uploads/MarketingData.csv'
        }
        
        data_file = file_mapping.get(data_type)
        
        if not data_file or not os.path.exists(data_file):
            return jsonify({"error": "File not found or invalid data type"}), 400
        
        # Read CSV into DataFrame
        df = pd.read_csv(data_file)
        
        # Classify columns into numeric and categorical
        numeric_columns, categorical_columns = classify_columns(df)
        
        # Create a response dict to hold analysis results
        analysis_result = {
            "categorical": {},
            "numerical": {}
        }

        # Analyze categorical columns
        for col in categorical_columns:
            analysis_result["categorical"][col] = {
                'unique_values': int(df[col].nunique()),  
                'most_common': df[col].value_counts().idxmax(),  
                'frequency': df[col].value_counts().to_dict()  
            }

        # Analyze numeric columns
        for col in numeric_columns:
            analysis_result["numerical"][col] = {
                'sum': float(df[col].sum()),  
                'mean': float(df[col].mean()),  
                'max': float(df[col].max()),  
                'min': float(df[col].min())  
            }

        # Store the analysis summary in ChromaDB
        analysis_id = f"analysis_{data_type}"
        analysis_text = json.dumps(analysis_result)
        analysis_collection.add(ids=[analysis_id], documents=[analysis_text])

        last_analysis_summary = analysis_result
        return jsonify(analysis_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query-analysis', methods=['POST'])
def query_analysis():
    """Endpoint to query analysis results from ChromaDB."""
    try:
        query = request.json.get("query")
        
        # Search the analysis results collection
        results = analysis_collection.query(query_texts=[query], n_results=20)
        print(results)

        documents = results["documents"][0]

        parsed_documents = []
        raw_data = []
        for doc in documents:
            try:
                parsed_doc = json.loads(doc)
                raw_data.append(parsed_doc)
                formatted_doc = json.dumps(parsed_doc, indent=2)
                parsed_documents.append(formatted_doc)
            except json.JSONDecodeError:
                parsed_documents.append(doc)

        data_df = pd.DataFrame(raw_data)

        # Construct context
        context = "\n\n".join(parsed_documents)

        prompt_template = """
        You are a highly knowledgeable virtual business consultant with access to comprehensive data about the company. 
        Your task is to always provide insightful, accurate, and helpful answers based on the data provided. The data is always sufficient to derive an answer. 
        You must:
        1. Analyze the context thoroughly.
        2. Be concise but informative. Provide actionable insights or strategies where applicable.
        3. Just compare the frequency or see the most common value of the thing present in the query and answer queries accordingly.
        4. Please check the information you have thouroughly because u have everything. 
        5. Do not give I dont have context as it can be irritating. If you feel that u don't have any answer then just give the answer which has most frequency
        6. For marketing data analyse the frequency for success and failure of the campaigns.
        Context: {context}
        Question: {query}
        """


        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain
        answer = chain.run({"context": context, "query": query})

        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)