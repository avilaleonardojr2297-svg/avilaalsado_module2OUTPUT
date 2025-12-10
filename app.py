import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import time
from dotenv import load_dotenv  
import os  

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
try:
    st.set_page_config(layout="wide", page_title="Avalanche Product Team Dashboard")
except st.errors.StreamlitAPIException as e:
    if "must be called as the first Streamlit command" in str(e):
        pass
    else:
        raise e

# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not API_KEY:
    st.error("⚠️ API Key not found! Please check your .env file.")
    st.stop()

# OpenAI Configuration
MODEL_NAME = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
API_URL = "https://api.openai.com/v1/chat/completions"

# --- Data Loading and Mocks (Step 1 & 2) ---

@st.cache_data
def load_data():
    """
    Simulates connecting to Snowflake and loading the 'reviews_with_sentiment' table.
    """
    # Create mock data mimicking the structure of the combined table
    data = {
        'product_id': np.random.choice(['PROD-101', 'PROD-202', 'PROD-303', 'PROD-404'], 500),
        'region': np.random.choice(['North America', 'Europe', 'Asia', 'South America'], 500),
        'delivery_status': np.random.choice(['On Time', 'Delayed', 'Early'], 500, p=[0.8, 0.15, 0.05]),
        'sentiment_score': np.round(np.random.normal(0.2, 0.6, 500), 2),
        'review_text': [
            "The product was great but shipping was slow.",
            "Absolutely perfect, arrived early and works well.",
            "Neutral feelings, nothing special.",
            "Completely broken on arrival, terrible service.",
        ] * 125,
        'timestamp': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.arange(500), unit='D')
    }
    df = pd.DataFrame(data)

    # Clean up sentiment scores to be within [-1, 1] and add labels
    df['sentiment_score'] = df['sentiment_score'].clip(-1.0, 1.0)
    df['sentiment_label'] = pd.cut(
        df['sentiment_score'],
        bins=[-1.1, -0.3, 0.3, 1.1],
        labels=['Negative', 'Neutral', 'Positive'],
        right=True
    )
    return df

# Load data
df = load_data()

# --- LLM Helper Function (OpenAI API) ---

def call_openai_chat(messages, system_instruction, max_retries=5):
    """
    Calls OpenAI's Chat Completions API.
    """
    for i in range(max_retries):
        try:
            # Prepare messages for OpenAI format
            openai_messages = [
                {"role": "system", "content": system_instruction}
            ]
            openai_messages.extend(messages)
            
            payload = {
                "model": MODEL_NAME,
                "messages": openai_messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                API_URL,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {API_KEY}'
                },
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            text = result['choices'][0]['message']['content']
            return text
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and i < max_retries - 1:
                wait_time = 2 ** i
                time.sleep(wait_time)
            else:
                return f"Error: API Request failed after {i+1} attempts: {e}\n\nResponse: {response.text if response else 'No response'}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {e}"
    
    return "Error: Maximum retries exceeded for API call."


# --- Streamlit App Layout ---

st.title("❄️ Avalanche Product Performance Dashboard")
st.markdown("Dashboard for exploring customer sentiment and shipping trends. This uses a simulated dataset, with the Chatbot powered by OpenAI's API.")

# 1. Sidebar Filters (Step 2)
st.sidebar.header("Filter Data")

# Filter options
all_regions = ['All'] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", all_regions)

all_products = ['All'] + sorted(df['product_id'].unique().tolist())
selected_product = st.sidebar.selectbox("Select Product ID", all_products)

# Apply filters
filtered_df = df.copy()
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]
if selected_product != 'All':
    filtered_df = filtered_df[filtered_df['product_id'] == selected_product]

st.sidebar.markdown(f"---")
st.sidebar.metric("Reviews Displayed", f"{len(filtered_df):,}")

# --- Main Content Tabs ---
tab_viz, tab_issues, tab_chatbot = st.tabs([
    "Sentiment & Trends",
    "Delivery Issues Table",
    "Cortex LLM Assistant"
])

# --- Tab 1: Sentiment & Trends (Step 3) ---
with tab_viz:
    st.header("Overall Performance Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sentiment by Region")
        sentiment_by_region = filtered_df.groupby('region')['sentiment_score'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn((sentiment_by_region + 1) / 2)
        sentiment_by_region.plot(kind='bar', ax=ax, color=colors)
        
        ax.set_title(f'Average Sentiment Score ({selected_product})')
        ax.set_ylabel('Average Sentiment Score (-1.0 to 1.0)')
        ax.set_xlabel('Region')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("Which regions have the most negative feedback?")
        st.info("Regions with bars extending below the 0-line have average negative sentiment.")

    with col2:
        st.subheader("Delivery Status Distribution")
        delivery_counts = filtered_df['delivery_status'].value_counts()
        fig_del, ax_del = plt.subplots(figsize=(10, 6))
        delivery_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax_del, colors=['#4CAF50', '#FF9800', '#2196F3'])
        ax_del.set_title('Delivery Status Distribution')
        ax_del.set_ylabel('')
        st.pyplot(fig_del)
        
    st.subheader("Data Preview")
    st.dataframe(filtered_df[['timestamp', 'product_id', 'region', 'sentiment_label', 'delivery_status', 'review_text']], use_container_width=True)


# --- Tab 2: Delivery Issues Table (Step 4) ---
with tab_issues:
    st.header("Highlighting Critical Delivery Issues")
    
    issues_df = filtered_df[
        (filtered_df['sentiment_label'] == 'Negative') & 
        (filtered_df['delivery_status'] != 'On Time')
    ].copy()
    
    st.subheader(f"Negative Reviews with Delivery Issues ({len(issues_df)} records)")
    
    if issues_df.empty:
        st.success("No records match the filter criteria (Negative Sentiment & Delivery Issues). Great job!")
    else:
        display_cols = ['product_id', 'region', 'delivery_status', 'sentiment_label', 'review_text']
        
        st.markdown("The table below shows customer reviews that are **Negative** AND where the **Delivery Status** was not 'On Time'.")
        st.dataframe(issues_df[display_cols], use_container_width=True)
        
        agg_df = issues_df.groupby(['product_id', 'region']).size().reset_index(name='Issue Count')
        st.subheader("Issue Count by Product and Region")
        st.dataframe(agg_df.sort_values(by='Issue Count', ascending=False), use_container_width=True)


# --- Tab 3: OpenAI Assistant (Step 5: Chatbot) ---
with tab_chatbot:
    st.header("Cortex LLM Assistant")
    st.markdown("Ask the assistant to summarize or analyze the data using the **Cortex-powered** backend (simulated with OpenAI's API).")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the customer reviews..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner('Assistant is thinking...'):
            # Prepare data context (limit to 50 rows to avoid token limits)
            data_summary = filtered_df[['review_text', 'sentiment_label', 'product_id', 'region', 'delivery_status']].head(50).to_string(index=False)
            
            system_instruction = (
                "You are an Avalanche Product Data Analyst. "
                "Your goal is to answer the user's question concisely based **only** on the data provided below. "
                "The data is a table of customer reviews and their sentiment scores. "
                "Do not mention the data limit, simply state your findings.\n\n"
                "Current Filtered Data Context:\n"
                f"{data_summary}"
            )
            
            # Get response from OpenAI
            response_text = call_openai_chat(
                st.session_state.messages,
                system_instruction
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- End of Streamlit App ---
