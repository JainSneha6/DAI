# DecisivAI: Business Decision-Making Platform

DecisivAI is an AI-powered platform designed to help businesses make informed decisions through data analysis, scenario simulation, and interactive insights. This platform integrates advanced machine learning models, natural language processing, and real-time business data visualization to empower enterprises with actionable insights.

## Introduction 

![image](https://github.com/user-attachments/assets/06f6caad-8db7-416c-b7f7-3fd088110201)

As businesses navigate increasingly complex environments, leveraging AI to make informed decisions is becoming essential. Studies show that:

- **97%** of companies believe AI will help their business.
- **46%** have already begun using AI in some capacity.
- **64%** of companies recognize that AI can improve customer relationships.

These statistics highlight the critical role AI will play in decision-making in the coming years. Traditional methods for business decision-making are likely to fall short in this new landscape, as the sheer volume of data and the need for timely insights outpace human analysis capabilities.

### Key Points:

- **AI & Large Language Models (LLMs)**: LLMs, such as the models used in DecisivAI, have the potential to process vast quantities of company data, providing valuable insights that drive business strategies and operations.
  
- **RAG (Retrieval-Augmented Generation)**: This model refines the decision-making process by combining external knowledge with AI's ability to retrieve relevant, real-time data. It enables the system to generate accurate insights, tailored to specific business contexts.

By utilizing these cutting-edge technologies, **DecisivAI** empowers businesses to stay ahead of the competition, make informed decisions faster, and foster stronger customer relationships.


## Overview

This project leverages advanced AI models, including RAG (Retrieval-Augmented Generation), Few-Shot Learning, and BERT, to provide users with a virtual consultant capable of answering queries, simulating scenarios, and providing insights based on real-world data. The system also incorporates an InsightBot for generating in-depth data analysis and an interactive dashboard for visualizing key metrics.

## Key Features

### 1. **Virtual Consultant**

| Feature             | Description                                                                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Query & Answer       | Users can ask the virtual consultant business-related questions, and the system will return relevant insights.                               |
| Feedback Integration | Users can provide feedback to improve the quality of the answers over time, allowing the system to fine-tune and enhance its responses.       |

```mermaid
flowchart LR
    A[User Inputs Query] --> B[Virtual Consultant Answers]
    B --> C[User Provides Feedback]
    C --> D[Model Improves Accuracy]
```

### 2. **InsightBot**

| Feature      | Description                                                                                                        |
|--------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Categorical Data**| Unique values, most common values, and frequency of specific fields.                                                       |
| **Numerical Data**  | Provides minimum, maximum, sum, and mean values for numerical metrics.                                                     |
| **Grouped Analysis**   |Provides all the necessary analysis of numerical data for each categorical variable                                  |
| **Visualization**   | Integrates with ReCharts.js to display bar charts for easy data visualization.                                             |


```mermaid
flowchart LR
    A[Categorical Data] --- B[Unique & Most Common Values]
    A --- C[Frequency Analysis]
    D[Numerical Data] --- E[Min & Max Values]
    D --- F[Sum & Mean Values]
    G[Grouped Analysis] --- H[Analysis per Categorical Variable]
    I[Visualization] --- J[Bar Charts with ReCharts.js]
```

### 3. **Scenario Simulation**

| Scenario Simulation | Description                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| Real-World Scenarios | Simulates real-world business scenarios and fine-tunes the AI models to provide better answers based on evolving trends.   |

```mermaid
flowchart LR
    A[Real-World Scenarios] --- B[Simulate Business Scenarios]
    B --- C[Analyze Trends]
    C --- D[Fine-Tune AI Models]
    D --- E[Provide Better Answers]
```

### 4. **AI-Powered Models**

| Model              | Description                                                                                                                |
|--------------------|----------------------------------------------------------------------------------------------------------------------------|
| **VectorDB**        | Stores business-specific data and industry trends for rapid retrieval.                                                     |
| **RAG**             | Enhances the answers provided by the Virtual Consultant by leveraging external documents or resources.                      |
| **Few-Shot Learning**| Allows the model to generalize answers with minimal additional data training.                                              |
| **BERT**            | A natural language processing model that improves the system's ability to understand user queries and generate insights.    |
| **Sentiment Analysis** | Uses VADER to analyze user feedback and fine-tunes the models based on sentiment (positive, neutral, or negative).       |


```mermaid
flowchart LR
    A[VectorDB] --- B[Store Business Data & Trends]
    C[RAG] --- D[Enhance Answers with External Resources]
    E[Few-Shot Learning] --- F[Generalize Answers with Minimal Data]
    G[BERT] --- H[Improve Query Understanding & Insights]
    I[Sentiment Analysis] --- J[Fine-Tune Models Based on User Feedback]

```

### 5. **Interactive Dashboard**

| Feature              | Description                                                                                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------|
| Visualization         | Displays bar charts to help users grasp key trends and statistics in a user-friendly format using **ReCharts.js**.          |

```mermaid
flowchart LR
    A[Data Input] --- B[Process Data]
    B --- C[Generate Bar Charts]
    C --- D[Display in User-Friendly Format]
    D --- E[Users Grasp Key Trends & Statistics]
```

### 6. **Sentiment Analysis for Feedback**

| Feature              | Description                                                                                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------|
| Sentiment Analysis         | Uses the VADER sentiment analysis model to gauge the emotional tone of user feedback (positive, neutral, or negative).          |
| VectorDB Integration       | Passes the analyzed sentiment data into the VectorDB to fine-tune and improve the AI models based on user responses.            |

```mermaid
flowchart LR
    A[User Feedback] --- B[VADER Sentiment Analysis]
    B --- C[Sentiment Classification]
    C --- D[Store Sentiment Data in VectorDB]
    D --- E[Model Fine-Tunes Based on Feedback Sentiment]

```

## Solution Overview
![image](https://github.com/user-attachments/assets/f3a47973-8c7a-4fd8-8e4a-6b4b7c6ee563)
![image](https://github.com/user-attachments/assets/d1969bc4-1e08-4fd9-b7c9-b2312e34ccc6)



## Project Architecture
![image](https://github.com/user-attachments/assets/9832b6bf-8ce5-4e39-a9a6-537318e42d83)



## How It Works

| **Step**            | **Description**                                                                                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Input**           | Users provide data related to the business, including sales, customer, inventory, and marketing data. This data is stored in the VectorDB.                              |
| **Model Interaction**| The AI model, fine-tuned through feedback and scenario simulations, retrieves relevant data from the VectorDB and applies RAG, Few-Shot Learning, and BERT to generate precise answers and insights. |
| **Data Categorization** | The InsightBot processes the input data, splitting it into categorical and numerical values.                                                                         |
| **Grouped Analysis** | The categorized data is used to calculate the minimum, maximum, average, and sum of each numerical column for individual categorical data. The calculated data is then stored in the VectorDB. |
| **Insights Generation** | The system presents the output in a tabular format and interactive bar charts on the dashboard.                                                                        |
| **Feedback Loop**   | Users can provide feedback to enhance the model's performance, leading to better results in the future.                                                                  |

## Technology Stack

![image](https://github.com/user-attachments/assets/1353361e-5251-4906-a367-cb571e305e51)






## Technology Comparison of DecisivAI with other alternatives

![image](https://github.com/user-attachments/assets/fa5f3cfe-5f16-400b-8ac8-f93c04d1cc18)



## Key Benefits of Our AI-Powered Decision-Making Platform

![image](https://github.com/user-attachments/assets/7c80b97c-c27b-4740-96e8-3d9177d60c48)






