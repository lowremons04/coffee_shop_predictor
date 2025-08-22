# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Project NextCup - Customer Purchase Predictor",
    page_icon="☕",
    layout="wide"
)

# --- Load the Trained Pipeline ---
# The pipeline includes the preprocessor, SMOTE (for training), and the final model.
# We use st.cache_resource to load it only once.
@st.cache_resource
def load_pipeline():
    """Load the complete prediction pipeline."""
    pipeline = joblib.load('coffee_purchase_predictor.joblib')
    return pipeline

pipeline = load_pipeline()

# Extract feature names from the preprocessor step of the pipeline
# This makes the app adaptable if you retrain the model with different features
try:
    # Accessing transformers in a scikit-learn pipeline
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Let's get the original feature names for the UI
    original_numerical_features = pipeline.named_steps['preprocessor'].transformers_[0][2]
    original_categorical_features = pipeline.named_steps['preprocessor'].transformers_[1][2]
except Exception as e:
    st.error(f"Could not extract feature names from the pipeline. Error: {e}")
    # Fallback to a hardcoded list if the above fails
    original_numerical_features = ['total_visits', 'total_spent', 'total_items', 'days_since_last_visit', 'customer_lifetime_days', 'avg_spent_per_visit', 'avg_items_per_visit', 'count_americano', 'count_americano_with_milk', 'count_cappuccino', 'count_cocoa', 'count_cortado', 'count_espresso', 'count_hot_chocolate', 'count_latte']
    original_categorical_features = ['most_frequent_store', 'favorite_weekday', 'favorite_time_of_day']


# --- App Layout ---
st.title("☕ Project NextCup")
st.subheader("Predicting a Customer's Next Purchase Category")
st.markdown("""
This app uses a machine learning model (XGBoost) to predict what a customer is likely to buy on their next visit.
Please enter the customer's historical data in the sidebar to get a prediction.
""")

st.sidebar.header("Customer Profile Input")

# --- Input Widgets in the Sidebar ---
def user_input_features():
    """Create sidebar widgets and return a DataFrame of the inputs."""
    inputs = {}
    
    st.sidebar.subheader("Visit & Spending Habits")
    inputs['total_visits'] = st.sidebar.slider("Total Number of Visits", 1, 100, 10)
    inputs['total_spent'] = st.sidebar.slider("Total Money Spent (in R)", 10, 5000, 500)
    inputs['days_since_last_visit'] = st.sidebar.slider("Days Since Last Visit", 0, 365, 30)

    st.sidebar.subheader("Behavioral Traits")
    inputs['most_frequent_store'] = st.sidebar.selectbox("Most Frequented Store Location", ['Hell\'s Kitchen', 'Lower Manhattan', 'Astoria'])
    inputs['favorite_weekday'] = st.sidebar.selectbox("Most Common Visit Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    inputs['favorite_time_of_day'] = st.sidebar.selectbox("Most Common Visit Time", ['Morning', 'Afternoon', 'Night'])
    
    st.sidebar.subheader("Historical Purchase Counts")
    # Dynamically create sliders for the product categories found during training
    for feature in original_numerical_features:
        if 'count_' in feature:
            # Clean up the name for the UI
            product_name = feature.replace('count_', '').replace('_', ' ').title()
            inputs[feature] = st.sidebar.slider(f"Count of '{product_name}'", 0, 50, 5)

    # Add any remaining numerical features that weren't in the purchase counts
    # This makes the app robust to feature changes
    for feature in original_numerical_features:
        if feature not in inputs:
             inputs[feature] = 0 # Default to 0 if not a purchase count

    data = pd.DataFrame([inputs])
    return data

input_df = user_input_features()

# --- Display User Input ---
st.write("---")
st.header("👤 Customer Input Profile")
st.dataframe(input_df, use_container_width=True)

# --- Prediction and Output ---
if st.button("Predict Next Purchase", type="primary"):
    
    # The pipeline handles all preprocessing and prediction in one step
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)
    
    st.write("---")
    st.header("📈 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Category:")
        st.success(f"**{prediction[0]}**")
        
    with col2:
        st.subheader("Prediction Confidence:")
        # Create a DataFrame for the probabilities
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=pipeline.classes_,
            index=["Probability"]
        ).T.sort_values("Probability", ascending=False)
        
        st.dataframe(proba_df.style.format("{:.2%}"))

st.markdown("---")
st.write("Developed by Low Jia Yuan, Abigail Chong Yung Ping, and Nathaniel Woo Shih Yan.")
