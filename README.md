# DecisivAI
**An AI-Powered Enterprise Decision-Making System**
---

## Team
- **Team Name**: EspressoOps
- **Members**:  
  - Sneha Jain  
  - Siddhartha Chakrabarty
 
<img width="968" height="545" alt="image" src="https://github.com/user-attachments/assets/40330bc5-ec67-4031-b63a-621f9e5d80ca" />
---

## Hackathon Theme / Challenge Addressed  
**Open Innovation**  

---

## Demo Video  

---

## Pitch Deck  

---
## How to run it

---

## Problem Statement
<img width="975" height="547" alt="image" src="https://github.com/user-attachments/assets/2ecf22e8-8e1c-40a6-88fc-d61dc87341ba" />

Modern organizations operate in highly dynamic environments where decisions depend on sensitive data spanning finance, marketing, operations, customers, and strategy. While advanced analytics and machine learning models exist, they are often siloed, difficult to access, and unable to respond quickly to ad-hoc *what-if* business questions.

### Core Challenges
- Decision-makers lack a unified system to simulate complex business scenarios in real time  
- Critical insights are delayed due to manual analysis and disconnected tools  
- Strategic decisions rely on static reports instead of predictive, scenario-based intelligence  
- Sensitive enterprise data and proprietary models are exposed to security, privacy, and compliance risks  

### Need for Encryption
DecisivAI operates on **highly confidential data**, including financial forecasts, customer behavior, pricing strategies, and competitive intelligence. Without strong encryption:
- Business-critical insights could be intercepted or leaked  
- Proprietary ML models and decision logic could be compromised  
- Regulatory compliance (GDPR, SOC 2, ISO 27001) would be at risk
  
**DecisivAI** addresses this gap by combining AI-driven scenario simulation with **enterprise-grade encryption**, enabling confident, secure, and intelligent decision-making.

---

## Our Solution
<img width="972" height="550" alt="image" src="https://github.com/user-attachments/assets/32b9a335-3bbf-4a54-9ba1-9ff006259007" />

**DecisivAI** is an AI-powered decision intelligence platform that turns *what-if* business questions into clear, actionable outcomes.

### What it does
- Understands scenarios in natural language  
- Extracts key variables, risks, and constraints  
- Runs secure simulations using ML models retrieved from **Cyborg DB**  
- Forecasts impact across revenue, demand, cost, and risk  
- Produces optimized, explainable recommendations  

### Why it matters
- Accelerates data-driven decision-making  
- Reduces uncertainty in strategic planning  
- Aligns marketing, finance, and operations  
- Empowers leaders with AI-backed confidence  

DecisivAI functions as a **virtual consultant**, combining encrypted intelligence from **Cyborg DB** with advanced analytics to guide better decisions.

---

## 1. Secure Data Ingestion 
<img width="971" height="545" alt="image" src="https://github.com/user-attachments/assets/35c81132-105c-4803-9513-c1a4dfad77ad" />

| Layer | Purpose | Key Capabilities |
|------|--------|------------------|
| **Data Intake Layer** | Ingest enterprise data from multiple sources | • CSV / Excel (sales, inventory, etc.)<br>• Database dumps (ERP, CRM, supply chain)<br>• PDFs and business documents<br>• Unstructured text (emails, notes) |
| **Data Preprocessing** | Prepare data for AI processing | • Normalize structured & unstructured data<br>• Parse documents (python-docx, PyPDF2)<br>• Remove PII when required<br>• Clean noisy text using NLP techniques |
| **AI Auto-Classification** | Automatically categorize enterprise data | • Uses **Gemini 2.5 Flash** for classification<br>• Sales, Marketing, Finance<br>• Inventory & Supply Chain<br>• Customer & Operational data |
| **Embedding Generation** | Convert data into vector representations | • Generates embeddings using Hugging Face sentence transformers<br>• Ensures AI-ready semantic representations |
| **Secure Vector Storage** | Protect and store embeddings | • Stores vectors in **Cyborg DB**<br>• Encrypted vector storage and retrieval<br>• Secure access for downstream ML models |

This architecture ensures **secure ingestion, intelligent classification, and encrypted storage** of enterprise data for trusted AI-driven decision intelligence.

---

## 2. ML Engine Hub
<img width="971" height="533" alt="image" src="https://github.com/user-attachments/assets/97bbaacb-8d5c-4003-8590-3ef6eb88b15e" />

| Name | ML Model | Data Required | Use Cases |
|-----|---------|---------------|-----------|
| **Sales, Demand & Financial Forecasting Model** | Prophet / ARIMA | Sales, Inventory, Financial data | Inventory planning, promotions, revenue projections |
| **Pricing & Revenue Optimization Model** | Regression, Bayesian Hierarchical Models | Sales data | Dynamic pricing, promotion planning, margin optimization |
| **Marketing ROI & Attribution Model** | ElasticNet, Bayesian MMM | Marketing data | Budget allocation, campaign optimization |
| **Customer Segmentation & Modeling** | K-Means, GMM | Customer data | Personalization, funnel optimization |
| **Customer Value & Retention Model** | BG/NBD, LightGBM | Customer data | Retention campaigns, VIP customer handling |
| **Sentiment & Intent NLP Model** | BERT, VADER | Sales, Marketing data | Sentiment analysis, product feedback loops |
| **Inventory & Replenishment Optimization Model** | EOQ, Stochastic Optimization | Inventory data | Automated inventory and supply planning |
| **Logistics & Supplier Risk Model** | GBM, Classification Models | Operational data | Routing decisions, supplier risk assessment |

The ML Engine Hub allows DecisivAI to dynamically select and orchestrate specialized models based on the scenario, ensuring accurate, explainable, and domain-specific decision intelligence.

---

## 3. Virtual Consultant
<img width="973" height="547" alt="image" src="https://github.com/user-attachments/assets/3bff7b30-fe49-47bc-bfaa-1e5aa0f45811" />

| Component | Description |
|---------|-------------|
| **User Interaction** | Users ask questions in natural language to explore business scenarios and insights |
| **Query Embedding** | User queries are converted into vector embeddings for semantic understanding |
| **Secure Retrieval** | Relevant knowledge and ML outputs are securely retrieved from **Cyborg DB** |
| **Intelligence Fusion** | Retrieved knowledge is combined with outputs from the ML Engine Hub |
| **LLM Reasoning** | The LLM analyzes combined context and generates the final response |
| **Predictive Insights** | Delivers forward-looking insights powered by ML forecasting and optimization models |
| **Secure Storage** | Embeddings of ML model outputs are encrypted and stored in **Cyborg DB** |
| **Auto Dashboards** | Automatically generates dashboards for each data domain and ML model |
| **Interactive Analytics** | Users can query and explore insights directly through dynamic dashboards |

This architecture enables a **secure, conversational, and predictive virtual consultant** for enterprise decision-making.

---

## 4. Simulation Engine
<img width="970" height="547" alt="image" src="https://github.com/user-attachments/assets/8ea48a5f-14bc-48b6-9553-30c8484efad2" />

| Step | Description |
|-----|-------------|
| **Scenario Input** | User provides a business scenario in natural language (e.g., “Simulate if our Q4 marketing budget drops by 15%”) |
| **Scenario Parsing** | Virtual Consultant extracts scenario type, key variables, and values (e.g., marketing → budget → –15%) |
| **Model Retrieval** | Relevant ML models are securely retrieved from **Cyborg DB** using encrypted vector search |
| **Scenario Application** | Scenario parameters are applied across selected ML models |
| **Model Recalculation** | Marketing ROI & Attribution Model recalculates expected sales and ROI by channel<br>• Sales, Demand & Financial Forecasting Model recalculates revenue trends and demand impact |
| **Insight Aggregation** | Outputs from multiple models are combined into a unified simulation result |
| **Decision Recommendation** | System generates a clear, actionable recommendation based on simulated outcomes |

The Simulation Engine enables secure, multi-model *what-if* analysis and delivers optimized, data-driven recommendations for confident decision-making.

---
## How CyborgDB Is Being Used
<img width="972" height="550" alt="image" src="https://github.com/user-attachments/assets/0a4d8cf5-66e7-4ffe-a876-6d8a2a6d4b0f" />

| Step | Component | Role of CyborgDB |
|-----|----------|------------------|
| **01** | Secure Data Ingestion | Stores all enterprise data as **encrypted embeddings** with no plaintext storage, ensuring full data privacy |
| **02** | ML Engine Hub | Stores ML predictions, forecasts, risk scores, and outputs as **encrypted vectors** |
| **03** | Virtual Consultant | Performs **secure vector search** for RAG, enabling question answering without exposing raw data |
| **04** | Simulation Engine | Securely saves simulation inputs and results, enabling a **self-improving decision engine** |

CyborgDB acts as the **encrypted intelligence backbone** of DecisivAI, ensuring privacy, security, and trust across every stage of decision-making.

---

## Architecture Diagram

