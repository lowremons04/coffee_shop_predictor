# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Purchase Predictor",
    page_icon="☕",
    layout="wide"
)

# --- Caching the Model and Encoder ---
# Use caching to load the model and encoder only once
@st.cache_resource
def load_model_and_encoder():
    """Load the trained pipeline and label encoder from disk."""
    try:
        pipeline = joblib.load('coffee_purchase_predictor.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error("Model or encoder files not found. Make sure 'coffee_purchase_predictor.joblib' and 'label_encoder.joblib' are in the same directory.")
        return None, None

pipeline, label_encoder = load_model_and_encoder()

# --- Application Title and Description ---
st.title("☕ Next Coffee Purchase Predictor")
st.markdown("""
This app predicts a customer's next coffee purchase based on their historical data. 
Enter the customer's details in the sidebar to get a prediction.
This demonstrates a practical application of a machine learning model in a business context.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Customer Features")

# Helper function to create inputs
def user_input_features():
    total_visits = st.sidebar.slider('Total Visits', 1, 100, 10)
    total_spent = st.sidebar.slider('Total Spent (R)', 50.0, 5000.0, 500.0, step=10.0)
    days_since_last_visit = st.sidebar.slider('Days Since Last Visit', 0, 365, 30)

    # Derived feature (calculated for convenience, but the model needs it)
    avg_spent_per_visit = total_spent / total_visits if total_visits > 0 else 0
    st.sidebar.metric("Average Spent per Visit (R)", f"{avg_spent_per_visit:.2f}")

    favorite_weekday = st.sidebar.selectbox('Favorite Weekday', 
                                            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    favorite_time_of_day = st.sidebar.selectbox('Favorite Time of Day', 
                                                ['Morning', 'Afternoon', 'Evening'])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Previous Coffee Counts")
    
    # Get the coffee types from the label encoder
    coffee_types = label_encoder.classes_
    
    coffee_counts = {}
    for coffee in coffee_types:
        # Create a user-friendly name for the input field
        input_label = f'Count of {coffee}'
        # Create the column name the model expects (e.g., 'count_cappuccino')
        col_name = f'count_{coffee.lower().replace(" ", "_")}'
        coffee_counts[col_name] = st.sidebar.number_input(input_label, min_value=0, max_value=50, value=2)

    data = {
        'total_visits': total_visits,
        'total_spent': total_spent,
        'days_since_last_visit': days_since_last_visit,
        'avg_spent_per_visit': avg_spent_per_visit,
        'favorite_weekday': favorite_weekday,
        'favorite_time_of_day': favorite_time_of_day,
        **coffee_counts # Unpack the dictionary of coffee counts
    }
    
    # The order of columns in this DataFrame MUST match the order used during training
    feature_df = pd.DataFrame(data, index=[0])
    return feature_df

input_df = user_input_features()

# --- Display Input Data ---
st.subheader("Customer Data Input")
st.write("The following features will be used for the prediction:")
st.dataframe(input_df, hide_index=True)

# --- Prediction Logic ---
if st.button('Predict Next Purchase', type="primary"):
    if pipeline is not None and label_encoder is not None:
        try:
            # The pipeline handles all preprocessing (scaling, one-hot encoding)
            prediction_encoded = pipeline.predict(input_df)
            
            # Inverse transform the numeric prediction to get the coffee name
            prediction_label = label_encoder.inverse_transform(prediction_encoded)
            
            st.success(f"🎉 The model predicts the customer will buy a **{prediction_label[0]}** next!")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model is not loaded. Cannot make a prediction.")
