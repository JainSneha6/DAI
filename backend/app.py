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

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection in ChromaDB for storing your data
collection = client.create_collection(name="business_insights")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = "AIzaSyC6iqFmmBrHeAzOu4VSgO7SYCkNtmwCZM8"  # Replace with your actual API key
llm = ChatGoogleGenerativeAI(api_key=api_key, model='gemini-pro')

# Global variable to store the last uploaded data's summary
last_uploaded_data_summary = {}

def analyze_csv(data):
    """Analyzes the uploaded CSV data."""
    summary = {}

    # Basic info
    summary['columns'] = data.columns.tolist()
    summary['data_types'] = data.dtypes.astype(str).to_dict()
    summary['sample_rows'] = data.head().to_dict(orient='records')
    
    return summary

# Route to upload business data
@app.route('/api/upload-business-data', methods=['POST'])
def upload_business_data():
    global last_uploaded_data_summary  # Access the global summary variable
    try:
        # Get the form data
        business_name = request.form.get("businessName")
        industry = request.form.get("industry")

        # Store all uploaded files in a dictionary
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

                # Store summary for each file type
                file_summaries[file_key] = summary

                # Add data to ChromaDB
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict(orient='records')

                for i, row in enumerate(data):
                    insight_id = f"{business_name}_{file_key}_{i}"  # Unique ID for each entry
                    insight_text = json.dumps(row)  # Convert the row to JSON string format

                    # Add to ChromaDB
                    collection.add(ids=[insight_id], documents=[insight_text])

        # Store a summary of all uploaded data
        last_uploaded_data_summary = file_summaries

        return jsonify({"message": "Files uploaded and data added successfully!", "summary": file_summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Route to search for insights
@app.route('/search-insight', methods=['POST'])
def search_insight():
    try:
        # Get search query from request
        query = request.json.get("query")
        
        # Perform search in ChromaDB
        results = collection.search(query_texts=[query], n_results=5)
        
        # Return search results
        return jsonify({"results": results["documents"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Route to delete an insight
@app.route('/delete-insight', methods=['DELETE'])
def delete_insight():
    try:
        # Get the insight id to delete
        insight_id = request.json.get("id")
        
        # Delete from ChromaDB
        collection.delete(ids=[insight_id])
        
        return jsonify({"message": "Insight deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def summarize_data(data):
    """Summarizes the key insights from the uploaded data."""
    if isinstance(data, pd.DataFrame):
        summary = {
            "total_orders": len(data),
            "columns": data.columns.tolist(),
            "column_summary": {}
        }
        
        # Generate summaries for each column based on its type
        for column in data.columns:
            column_data = data[column]
            if pd.api.types.is_numeric_dtype(column_data):
                summary["column_summary"][column] = {
                    "mean": column_data.mean(),
                    "median": column_data.median(),
                    "std_dev": column_data.std(),
                    "min": column_data.min(),
                    "max": column_data.max()
                }
            elif pd.api.types.is_categorical_dtype(column_data) or pd.api.types.is_object_dtype(column_data):
                summary["column_summary"][column] = {
                    "unique_values": column_data.unique().tolist(),
                    "count": column_data.value_counts().to_dict()
                }
            elif pd.api.types.is_datetime64_any_dtype(column_data):
                summary["column_summary"][column] = {
                    "min_date": column_data.min(),
                    "max_date": column_data.max()
                }
        
        return summary
    return {}

# Route to handle consultant query
@app.route('/consultant-query', methods=['POST'])
def consultant_query():
    try:
        # Get the query from request
        query = request.json.get("query")

        # Retrieve relevant data from ChromaDB
        search_results = collection.query(query_texts=[query], n_results=100)

        documents = search_results["documents"][0]  # Access the list of document strings

        parsed_documents = []
        raw_data = []
        for doc in documents:
            try:
                parsed_doc = json.loads(doc)  # Convert the stringified JSON back into a dictionary
                raw_data.append(parsed_doc)
                formatted_doc = json.dumps(parsed_doc, indent=2)
                parsed_documents.append(formatted_doc)
            except json.JSONDecodeError:
                parsed_documents.append(doc)

        # Convert raw_data to DataFrame to generate insights
        data_df = pd.DataFrame(raw_data)
        

        # Construct context
        context = "\n\n".join(parsed_documents)

        # Add last uploaded data summary if available
        

        # Log context for debugging
        print("Constructed Context:", context)

        # Initialize the LLMChain with the LLM
        prompt_template = """This is the business information of the company. Analyse this information.
        Then use your own intelligence to answer the question properly.
        If proper information cannot be derived from the context, still make assumptions accordingly and answer the query.
        Tackle any query which is being asked even if information may not be present in the context. Make use of your existing knowledge to answer that.
        Give the final answer in one single paragraph without any asterisks.
        {context}
        Question: {query}"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain with the retrieved context and user query
        answer = chain.run({"context": context, "query": query})

        # Return the answer from the LLM
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)