# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Shop Business Intelligence",
    page_icon="☕",
    layout="wide"
)

# --- Caching the Model and Encoder ---
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
st.title("☕ Coffee Shop Business Intelligence Dashboard")
st.markdown("""
Welcome to your business dashboard. This tool provides three key functions:
1.  **Single Customer Prediction:** Predict the next coffee purchase for an individual customer.
2.  **Batch Forecasting:** Upload a CSV of customer data to predict next purchases for many customers at once.
3.  **Inventory & Insights:** Get a demand forecast for the upcoming week and view historical favorites based on your uploaded data.
""")

# --- Create Tabs for different functionalities ---
tab1, tab2, tab3 = st.tabs(["👤 Single Customer Prediction", "📈 Batch Forecasting", "📊 Inventory & Insights"])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab1:
    st.header("Predict Next Purchase for a Single Customer")
    
    with st.form("single_customer_form"):
        # Create columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Purchase History")
            total_visits = st.slider('Total Visits', 1, 100, 10)
            total_spent = st.slider('Total Spent (R)', 50.0, 5000.0, 500.0, step=10.0)
            days_since_last_visit = st.slider('Days Since Last Visit', 0, 365, 30)
            avg_spent_per_visit = total_spent / total_visits if total_visits > 0 else 0
            st.metric("Average Spent per Visit (R)", f"{avg_spent_per_visit:.2f}")

        with col2:
            st.subheader("Customer Habits")
            favorite_weekday = st.selectbox('Favorite Weekday', 
                                            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key='single_weekday')
            favorite_time_of_day = st.selectbox('Favorite Time of Day', 
                                                ['Morning', 'Afternoon', 'Evening'], key='single_time')

        st.subheader("Previous Coffee Counts")
        coffee_types = label_encoder.classes_ if label_encoder else []
        coffee_cols = st.columns(len(coffee_types))
        coffee_counts = {}

        for i, coffee in enumerate(coffee_types):
            with coffee_cols[i]:
                input_label = coffee
                col_name = f'count_{coffee.lower().replace(" ", "_")}'
                coffee_counts[col_name] = st.number_input(input_label, min_value=0, max_value=50, value=2, key=f'single_{col_name}')

        # Submit button for the form
        submitted = st.form_submit_button("Predict Next Purchase")
        
        if submitted:
            if pipeline is not None and label_encoder is not None:
                data = {
                    'total_visits': total_visits, 'total_spent': total_spent,
                    'days_since_last_visit': days_since_last_visit, 'avg_spent_per_visit': avg_spent_per_visit,
                    'favorite_weekday': favorite_weekday, 'favorite_time_of_day': favorite_time_of_day,
                    **coffee_counts
                }
                input_df = pd.DataFrame(data, index=[0])
                
                prediction_encoded = pipeline.predict(input_df)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)
                
                st.success(f"🎉 The model predicts the customer will buy a **{prediction_label[0]}** next!")
                st.balloons()
            else:
                st.warning("Model is not loaded. Cannot make a prediction.")


# ==============================================================================
# TAB 2 & 3: BATCH FORECASTING AND INSIGHTS
# ==============================================================================
# We define a placeholder for the results to share between tabs
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

with tab2:
    st.header("Batch Forecasting with CSV File")
    st.write("Upload a CSV file with customer data to predict their next purchases.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("✅ **CSV Uploaded Successfully!** Here's a preview:")
            st.dataframe(batch_df.head())

            # Basic validation to check for required columns
            required_cols = ['total_visits', 'total_spent', 'days_since_last_visit', 'avg_spent_per_visit', 'favorite_weekday', 'favorite_time_of_day']
            if all(col in batch_df.columns for col in required_cols):
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Predicting for all customers..."):
                        predictions_encoded = pipeline.predict(batch_df)
                        predictions_labels = label_encoder.inverse_transform(predictions_encoded)
                        
                        results_df = batch_df.copy()
                        results_df['predicted_next_purchase'] = predictions_labels
                        st.session_state.batch_results = results_df # Save to session state
                        
                        st.success("✅ **Batch Prediction Complete!**")
                        st.write("Results:")
                        st.dataframe(results_df)
                        
                        # Provide a download button for the results
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name='predicted_purchases.csv',
                            mime='text/csv',
                        )
            else:
                st.error(f"❌ **CSV Error:** Your file is missing one or more required columns. Please ensure it contains at least: {required_cols}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab3:
    st.header("Inventory Demand Forecast & Customer Insights")
    st.write("This tab uses the results from the 'Batch Forecasting' tab to generate insights.")
    
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- Feature 2: Stock Count Prediction ---
            st.subheader("Predicted Demand for Next Week")
            st.write("Based on the predicted next purchase for each customer in your uploaded file.")
            
            demand_forecast = results_df['predicted_next_purchase'].value_counts().reset_index()
            demand_forecast.columns = ['Coffee Type', 'Predicted Number of Sales']
            
            st.bar_chart(demand_forecast, x='Coffee Type', y='Predicted Number of Sales', color="#FF8C00")
            st.write("This chart estimates the number of units you might sell for each coffee type, assuming each customer in the batch visits once next week. Use this to guide your inventory stocking.")

        with col2:
            # --- Feature 3: Top Favorite Drinks (Historical) ---
            st.subheader("Historical Customer Favorites")
            st.write("Based on the total purchase counts from your uploaded file.")

            count_cols = [col for col in results_df.columns if col.startswith('count_')]
            if count_cols:
                historical_favorites = results_df[count_cols].sum().sort_values(ascending=False)
                historical_favorites.index = [idx.replace('count_', '').replace('_', ' ').title() for idx in historical_favorites.index]
                historical_favorites = historical_favorites.reset_index()
                historical_favorites.columns = ['Coffee Type', 'Total Historical Purchases']

                st.bar_chart(historical_favorites, x='Coffee Type', y='Total Historical Purchases', color="#008080")
                st.write("This chart shows which drinks have been the most popular historically among the customers in your uploaded file.")
            else:
                st.warning("Could not find historical purchase count columns (e.g., 'count_latte') in the uploaded file.")
    else:
        st.info("ℹ️ Please upload a CSV and run a batch prediction in the **'Batch Forecasting'** tab to see insights here.")
