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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

client = chromadb.Client()

insight_collection = client.create_collection(name="business_insights")
analysis_collection = client.create_collection(name="business_analysis")
feedback_collection = client.create_collection(name="user_feedback")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = "AIzaSyDHqtWn2Ye71A_aCc8udlNyjEpZyf15TBw"  
llm = ChatGoogleGenerativeAI(api_key=api_key, model='gemini-pro')

analyzer = SentimentIntensityAnalyzer()

last_uploaded_data_summary = {}

def analyze_csv(data):
    """Analyzes the uploaded CSV data and returns summary statistics."""
    summary = {
        'columns': data.columns.tolist(),
        'data_types': data.dtypes.astype(str).to_dict(),
        'sample_rows': data.head().to_dict(orient='records')
    }
    return summary

def analyze_sentiment(text):
    """Analyze the sentiment of the given text using VADER."""
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores

@app.route('/api/upload-business-data', methods=['POST'])
def upload_business_data():
    try:
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

                file.save(file_path)

                file_paths[file_key] = file_path

        return jsonify({"message": "Files uploaded successfully!", "file_paths": file_paths}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

def calcVaderSentiment():
    customer_data_path = f'uploads/CustomerData.csv'
    
    try:
        df = pd.read_csv(customer_data_path)
    except Exception as e:
        return
    
    if 'Product Preferences' in df.columns:
        column_name = 'Product Preferences'
    elif 'Product Feedback' in df.columns:
        column_name = 'Product Feedback'
    else:
        return 
    
    print(column_name)
    
    df['Sentiment'] = df[column_name].apply(lambda x: analyze_sentiment(str(x)))

    sentiment_results = df[['Sentiment', column_name]].to_dict(orient='records')

    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for sentiment in df['Sentiment']:
        compound_score = sentiment['compound']
        if compound_score >= 0.05:
            positive_count += 1
        elif compound_score <= -0.05:
            negative_count += 1
        else:
            neutral_count += 1

    
    return jsonify({
        "answer": f'Positive Feedback: {positive_count}, Neutral Feedback: {neutral_count}, negative Feedback: {negative_count}'
    }), 200

        

@app.route('/process-and-store-data', methods=['GET'])
def process_and_store_data():
    try:
        file_mapping = {
            'salesData': 'uploads/SalesData.csv',
            'customerData': 'uploads/CustomerData.csv',
            'inventoryData': 'uploads/InventoryData.csv',
            'marketingCampaignsData': 'uploads/MarketingData.csv'
        }

        file_summaries = {}

        for file_key, file_path in file_mapping.items():
            if os.path.exists(file_path):
                file_extension = file_path.split('.')[-1]

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

                file_summaries[file_key] = summary

                if isinstance(data, pd.DataFrame):
                    data = data.to_dict(orient='records')

                for i, row in enumerate(data):
                    insight_id = f"{file_key}_{i}"
                    insight_text = json.dumps(row)
                    insight_collection.add(ids=[insight_id], documents=[insight_text])

        return jsonify({"message": "Data processed and added to ChromaDB!", "summary": file_summaries}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/search-insight', methods=['POST'])
def search_insight():
    try:
        query = request.json.get("query")
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
        
        if "feedback" in query.lower():
            print()
            return calcVaderSentiment()

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
        8. If asked about feedbacks then give the sales_data_sentiment_counts that you have.

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

        answer = chain.run({"context": context, "query": query})
        
        answer=answer.replace('**','')

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
def classify_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number, np.datetime64]).columns.tolist()
    return numeric_columns, categorical_columns

def analyze_numerical_by_categorical(df):
    """Analyze numerical columns grouped by categorical values, ignoring numeric columns that contain 'ID' or 'Date'."""
    
    numeric_columns, categorical_columns = classify_columns(df)
    
    numeric_columns = [col for col in numeric_columns if 'ID' not in col and 'Date' not in col]
    
    analysis_result = {}
    
    for cat_col in categorical_columns:
        if 'ID' in cat_col or 'Date' in cat_col:
            continue  

        grouped = df.groupby(cat_col)[numeric_columns].agg(['min', 'max', 'mean', 'sum'])
  
        grouped.columns = ['_'.join(map(str, col)).strip() for col in grouped.columns.values]
        
        analysis_result[cat_col] = grouped.reset_index().to_dict(orient='records')

    return analysis_result


@app.route('/api/analyze-data', methods=['POST'])
def analyze_data():
    global last_analysis_summary  
    try:
        data_type = request.json.get('dataType') 
        
        file_mapping = {
            'salesData': 'uploads/SalesData.csv',
            'customerData': 'uploads/CustomerData.csv',
            'inventoryData': 'uploads/InventoryData.csv',
            'marketingCampaignsData': 'uploads/MarketingData.csv'
        }
        
        data_file = file_mapping.get(data_type)
        
        if not data_file or not os.path.exists(data_file):
            return jsonify({"error": "File not found or invalid data type"}), 400
        
        df = pd.read_csv(data_file)
        
        numeric_columns, categorical_columns = classify_columns(df)
        
        analysis_result = {
            "categorical": {},
            "numerical": {},
            "grouped_analysis": {}  
        }

        for col in categorical_columns:
            analysis_result["categorical"][col] = {
                'unique_values': int(df[col].nunique()),  
                'most_common': df[col].value_counts().idxmax(),  
                'frequency': df[col].value_counts().to_dict()  
            }

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

        grouped_analysis_result = analyze_numerical_by_categorical(df)
        
        analysis_result["grouped_analysis"] = grouped_analysis_result  
       
        last_analysis_summary = analysis_result
    
        return jsonify(analysis_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/grouped-analysis', methods=['POST'])  
def get_grouped_analysis():
    try:
        data_type = request.json.get('dataType') 
        print(data_type)  
        
        file_mapping = {
            'salesData': 'uploads/SalesData.csv',
            'customerData': 'uploads/CustomerData.csv',
            'inventoryData': 'uploads/InventoryData.csv',
            'marketingCampaignsData': 'uploads/MarketingData.csv'
        }
        
        data_file = file_mapping.get(data_type)
        
        if not data_file or not os.path.exists(data_file):
            return jsonify({"error": "File not found or invalid data type"}), 400
        
        df = pd.read_csv(data_file)
        
        analysis_result = {
            "grouped_analysis": {}  
        }

        grouped_analysis_result = analyze_numerical_by_categorical(df)
        
        analysis_result["grouped_analysis"] = grouped_analysis_result 
        
        return jsonify(analysis_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query-analysis', methods=['POST'])
def query_analysis():
    """Endpoint to query analysis results from ChromaDB."""
    try:
        query = request.json.get("query")
        
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
        7. If somebody asks about the highest or lowest of something then just check the frequency of that thing and whoever has the highest or lowest frequest should be your answer.
        Context: {context}
        Question: {query}
        """


        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
        chain = LLMChain(llm=llm, prompt=prompt)

        answer = chain.run({"context": context, "query": query})

        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        rating = data.get('rating')  
        comments = data.get('comments', "")  

        if rating is None or not isinstance(rating, (int, float)):
            return jsonify({"error": "Invalid rating. Rating must be a number."}), 400

        if not isinstance(comments, str):
            return jsonify({"error": "Comments must be a string."}), 400

        feedback_id = str(uuid.uuid4())  

        feedback_entry = {
            "rating": int(rating),  
            "comments": comments
        }

        feedback_text = json.dumps(feedback_entry)
        
        feedback_collection.add(ids=[feedback_id], documents=[feedback_text])  

        return jsonify({"message": "Feedback submitted successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    
    
@app.route('/feedback/<feedback_id>', methods=['GET'])
def get_feedback(feedback_id):
    try:
        results = feedback_collection.query(query_texts=[feedback_id], n_results=1)

        feedback_list = []
        for doc in results['documents'][0]:
            feedback_list.append(doc)  

        return jsonify({"feedback": feedback_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)