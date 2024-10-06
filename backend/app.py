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
import uuid

# Initialize ChromaDB client
client = chromadb.Client()

insight_collection = client.create_collection(name="business_insights")
analysis_collection = client.create_collection(name="business_analysis")
feedback_collection = client.create_collection(name="user_feedback")

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
# Route to upload business data (only saving files, no processing)
@app.route('/api/upload-business-data', methods=['POST'])
def upload_business_data():
    try:
        # Store uploaded files in a dictionary
        uploaded_files = {
            'salesData': request.files.get('salesData'),
            'customerData': request.files.get('customerData'),
            'inventoryData': request.files.get('inventoryData'),
            'marketingCampaignsData': request.files.get('marketingCampaignsData')
        }

        file_paths = {}
        for file_key, file in uploaded_files.items():
            if file:
                file_extension = file.filename.split('.')[-1]
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)

                # Save the file to the specified directory
                file.save(file_path)

                # Store the saved file path for future reference
                file_paths[file_key] = file_path

        # Respond with success message and file paths
        return jsonify({"message": "Files uploaded successfully!", "file_paths": file_paths}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Route to process saved files and store data in ChromaDB when the virtual consultant page is accessed
@app.route('/process-and-store-data', methods=['GET'])
def process_and_store_data():
    try:
        # Map file types to their saved file paths
        file_mapping = {
            'salesData': 'uploads/SalesData.csv',
            'customerData': 'uploads/CustomerData.csv',
            'inventoryData': 'uploads/InventoryData.csv',
            'marketingCampaignsData': 'uploads/MarketingData.csv'
        }

        file_summaries = {}

        # Process each saved file
        for file_key, file_path in file_mapping.items():
            if os.path.exists(file_path):
                file_extension = file_path.split('.')[-1]

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

                # Add data to ChromaDB (only if it's a DataFrame)
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict(orient='records')

                for i, row in enumerate(data):
                    insight_id = f"{file_key}_{i}"
                    insight_text = json.dumps(row)
                    insight_collection.add(ids=[insight_id], documents=[insight_text])

        return jsonify({"message": "Data processed and added to ChromaDB!", "summary": file_summaries}), 200

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
        7. For marketing data give the information you have for that marketing campaign for success and failure of the campaigns.

        ### Examples:

        **Example 1:**
        Context: The company recently launched a digital marketing campaign for a new product targeting millennials. The campaign resulted in a 15% increase in website traffic but a low conversion rate of 2%.
        Question: What can be improved in the marketing campaign?
        Answer: To improve the marketing campaign, consider refining the target audience by conducting a deeper analysis of millennial preferences. Utilize A/B testing for different ad creatives and messages to identify what resonates most. Additionally, implementing a retargeting strategy could help convert the increased traffic into sales by reminding visitors of the product. 

        **Example 2:**
        Context: Sales for product X have been declining over the past two quarters despite an increase in advertising spend. Customer feedback indicates dissatisfaction with customer service.
        Question: What should the company do to address this issue?
        Answer: The company should first investigate the specific issues within customer service that are causing dissatisfaction. Implementing training programs for customer service representatives can enhance their performance. Additionally, evaluating the advertising strategy to ensure it aligns with customer needs can improve overall sales. It might be beneficial to reduce advertising spend temporarily until the customer service issues are resolved.

        **Example 3:**
        Context: The latest quarterly report shows a steady increase in customer retention rates but a decline in new customer acquisition. 
        Question: What strategies can be employed to attract new customers?
        Answer: To attract new customers, consider leveraging referral programs to encourage existing customers to recommend the product to their networks. Additionally, enhance online presence through search engine optimization (SEO) and content marketing to increase visibility. Partnering with influencers who resonate with the target demographic can also drive new customer acquisition.

        **Example 4:**
        Context: A recent market survey indicated that 60% of customers prefer sustainable products. The company's current product line has limited eco-friendly options.
        Question: How can the company align its product offerings with customer preferences?
        Answer: To align product offerings with customer preferences, the company should explore developing a line of eco-friendly products. Conducting a feasibility study on sourcing sustainable materials can provide insights into potential costs and production processes. Additionally, marketing these new products as part of a broader commitment to sustainability can enhance brand loyalty among environmentally conscious consumers.

        **Example 5:**
        Context: The company is facing increasing competition in its primary market. Market share has dropped by 10% over the last year.
        Question: What steps should the company take to regain its competitive edge?
        Answer: To regain its competitive edge, the company should conduct a comprehensive competitor analysis to identify strengths and weaknesses compared to rivals. Focus on innovation in product features or customer service can differentiate the brand. Additionally, consider strategic partnerships or collaborations to expand reach and resources, along with revisiting pricing strategies to ensure competitiveness.

        **Example 6:**
        Context: The customer satisfaction survey results show that 40% of customers are dissatisfied with the delivery times.
        Question: What improvements can be made to enhance delivery performance?
        Answer: To enhance delivery performance, the company should analyze the current logistics and supply chain processes to identify bottlenecks. Partnering with additional carriers or implementing a more robust inventory management system can reduce delays. Communicating realistic delivery times to customers and improving order tracking can also help manage expectations and enhance satisfaction.

        **Example 7:**
        Context: The company is planning to enter a new geographic market where it has no previous presence. Initial research shows a strong demand for its products.
        Question: What should the company consider before launching in this new market?
        Answer: Before launching in a new market, the company should conduct thorough market research to understand local preferences, cultural nuances, and regulatory requirements. Developing a localized marketing strategy that resonates with the target audience is crucial. Additionally, assessing distribution channels and building relationships with local partners can facilitate smoother entry and enhance chances of success.

        **Example 8:**
        Context: The company's product has received mixed reviews online, leading to a decrease in sales. Many reviews mention the high price point as a concern.
        Question: What strategies can the company use to address this feedback?
        Answer: The company should analyze the reviews to identify common concerns and address them in future iterations of the product. Conducting a pricing analysis against competitors can help determine if adjustments are necessary. Additionally, consider implementing promotional strategies, such as limited-time discounts or bundling products, to incentivize purchases and improve perceptions of value.

        Just add some supporting facts and figures from the data you have of the company.
        Context: {context}
        Question: {query}
        """


        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain
        answer = chain.run({"context": context, "query": query})
        
        answer=answer.replace('**','')

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# Function to classify columns into numeric and categorical
def classify_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number, np.datetime64]).columns.tolist()
    return numeric_columns, categorical_columns

def analyze_numerical_by_categorical(df):
    """Analyze numerical columns grouped by categorical values, ignoring numeric columns that contain 'ID' or 'Date'."""
    # Classify columns into numeric and categorical
    numeric_columns, categorical_columns = classify_columns(df)
    
    # Filter out numeric columns containing 'ID' or 'Date'
    numeric_columns = [col for col in numeric_columns if 'ID' not in col and 'Date' not in col]
    
    analysis_result = {}
    
    # For each categorical column, group by and calculate statistics for numeric columns
    for cat_col in categorical_columns:
        # Check if the categorical column contains 'ID' or 'Date'
        if 'ID' in cat_col or 'Date' in cat_col:
            continue  # Skip this column

        # Group by the categorical column and aggregate statistics
        grouped = df.groupby(cat_col)[numeric_columns].agg(['min', 'max', 'mean', 'sum'])
        
        # Flatten the MultiIndex columns for easier JSON serialization
        grouped.columns = ['_'.join(map(str, col)).strip() for col in grouped.columns.values]
        
        # Convert the DataFrame to a dictionary
        analysis_result[cat_col] = grouped.reset_index().to_dict(orient='records')

    return analysis_result


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
            "numerical": {},
            "grouped_analysis": {}  # New section to hold grouped analysis
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

        analysis_id = f"analysis_{data_type}"
        analysis_text = json.dumps(analysis_result)
        analysis_collection.add(ids=[analysis_id], documents=[analysis_text])

        # Perform analysis on numerical columns grouped by categorical values
        grouped_analysis_result = analyze_numerical_by_categorical(df)
        
        analysis_result["grouped_analysis"] = grouped_analysis_result  # Add grouped analysis to the result
        # Store the analysis summary in ChromaDB
        
       
        last_analysis_summary = analysis_result
    
        return jsonify(analysis_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/grouped-analysis', methods=['POST'])  # Change to POST
def get_grouped_analysis():
    try:
        data_type = request.json.get('dataType') 
        print(data_type)  # Get the type of data (sales, customer, etc.)
        
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
        
        # Create a response dict to hold analysis results
        analysis_result = {
            "grouped_analysis": {}  # New section to hold grouped analysis
        }

        # Perform analysis on numerical columns grouped by categorical values
        grouped_analysis_result = analyze_numerical_by_categorical(df)
        
        analysis_result["grouped_analysis"] = grouped_analysis_result  # Add grouped analysis to the result
        
        return jsonify(analysis_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query-analysis', methods=['POST'])
def query_analysis():
    """Endpoint to query analysis results from ChromaDB."""
    try:
        query = request.json.get("query")
        
        # Search the analysis results collection
        results = analysis_collection.query(query_texts=[query], n_results=1)
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
        6. For marketing data give the information you have for that marketing campaign for success and failure of the campaigns.
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
    
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        rating = data.get('rating')  # Expecting a numeric rating (1-5)
        comments = data.get('comments', "")  # Default to empty string if comments not provided

        # Validate the rating to ensure it's a valid integer
        if rating is None or not isinstance(rating, (int, float)):
            return jsonify({"error": "Invalid rating. Rating must be a number."}), 400

        # Ensure comments are stored as a string
        if not isinstance(comments, str):
            return jsonify({"error": "Comments must be a string."}), 400

        # Create a unique ID for the feedback
        feedback_id = str(uuid.uuid4())  # Generate a unique ID

        # Create feedback entry
        feedback_entry = {
            "rating": int(rating),  # Convert to int if needed
            "comments": comments
        }
        

        feedback_text = json.dumps(feedback_entry)
        
        # Insert feedback into ChromaDB
        feedback_collection.add(ids=[feedback_id], documents=[feedback_text])  # Ensure this is compatible, you might not need json.dumps here

        return jsonify({"message": "Feedback submitted successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    
    
@app.route('/feedback/<feedback_id>', methods=['GET'])
def get_feedback(feedback_id):
    try:
        # Assuming you can query by ID directly, this will depend on your ChromaDB setup
        results = feedback_collection.query(query_texts=[feedback_id], n_results=1)

        feedback_list = []
        for doc in results['documents'][0]:
            feedback_list.append(doc)  # Assuming doc already contains the necessary fields

        return jsonify({"feedback": feedback_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    



# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)