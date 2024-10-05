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

api_key = "AIzaSyB-Q1rafHEDK-De2w8AozD7wH0pSv7Sffg"  # Replace with your actual API key
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

# Route to handle file upload and add data to ChromaDB
@app.route('/api/upload-business-data', methods=['POST'])
def upload_business_data():
    global last_uploaded_data_summary  # Access the global summary variable
    try:
        # Get the form data
        business_name = request.form.get("businessName")
        industry = request.form.get("industry")
        description = request.form.get("description")
        contact_email = request.form.get("contactEmail")
        
        # Get the uploaded file
        file = request.files.get('file')

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save the file to the uploads directory
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the file based on its extension
        file_extension = file.filename.split('.')[-1]
        if file_extension == "csv":
            data = pd.read_csv(file_path)
            last_uploaded_data_summary = analyze_csv(data)  # Analyze the CSV
        elif file_extension == "xlsx":
            data = pd.read_excel(file_path)
            last_uploaded_data_summary = analyze_csv(data)  # Analyze the Excel
        elif file_extension == "json":
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            last_uploaded_data_summary = {"message": "JSON files are not analyzed in detail."}
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Iterate over the data and add each entry to ChromaDB
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries

        for i, row in enumerate(data):
            insight_id = f"{business_name}_{i}"  # Create unique ID for each entry
            insight_text = json.dumps(row)  # Convert the row to JSON string format

            # Add to ChromaDB
            collection.add(ids=[insight_id], embeddings=None, documents=[insight_text])

        return jsonify({"message": "File uploaded and data added successfully!", "summary": last_uploaded_data_summary}), 200

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

# Route to fine-tune or prompt the LLM with data from ChromaDB
@app.route('/consultant-query', methods=['POST'])
def consultant_query():
    try:
        # Get the query from request
        query = request.json.get("query")
        
        # Create a LangChain PromptTemplate
        prompt_template = """Use the following information to answer the query:
        {context}
        Question: {query}"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        
        # Retrieve context from ChromaDB
        search_results = collection.query(query_texts=[query], n_results=5)
        context = "\n".join(search_results["documents"])
        
        # Add the analysis summary to the context for more informed responses
        context += "\n\n" + json.dumps(last_uploaded_data_summary)

        # Initialize the LLMChain with the LLM
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
