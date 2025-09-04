# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

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

# --- INITIALIZE SESSION STATE ---
# This is the corrected placement. Initialize state variables at the top.
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# --- Application Title and Description ---
st.title("☕ Coffee Shop Business Intelligence Dashboard")
st.markdown("""
Welcome to your business dashboard. This tool provides three key functions:
1.  **Single Customer Prediction:** Predict the next coffee purchase for an individual customer.
2.  **Batch Forecasting:** Upload a CSV of customer data to predict next purchases for many customers at once.
3.  **Inventory & Insights:** Get a demand forecast for the upcoming week/month and view historical favorites based on your uploaded data.
""")

# --- Create Tabs for different functionalities ---
tab1, tab2, tab3 = st.tabs(["👤 Single Customer Prediction", "📈 Batch Forecasting", "📊 Inventory & Insights"])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab1:
    # This tab's code is correct.
    st.header("Predict Next Purchase for a Single Customer")
    
    with st.form("single_customer_form"):
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
        if pipeline is not None and label_encoder is not None:
            coffee_types = label_encoder.classes_
            num_columns = min(len(coffee_types), 5)
            coffee_cols_rows = [st.columns(num_columns) for _ in range((len(coffee_types) + num_columns - 1) // num_columns)]
            coffee_counts = {}
            all_cols = [col for row in coffee_cols_rows for col in row]
            for i, coffee in enumerate(coffee_types):
                with all_cols[i]:
                    input_label = coffee
                    col_name = f'count_{coffee.lower().replace(" ", "_")}'
                    coffee_counts[col_name] = st.number_input(input_label, min_value=0, max_value=50, value=2, key=f'single_{col_name}')
        else:
            st.warning("Model not loaded. Cannot display coffee types.")
            coffee_counts = {}

        submitted = st.form_submit_button("Predict Next Purchase")
        if submitted:
            if pipeline is not None and label_encoder is not None:
                required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
                data = {
                    'total_visits': total_visits, 'total_spent': total_spent,
                    'days_since_last_visit': days_since_last_visit, 'avg_spent_per_visit': avg_spent_per_visit,
                    'favorite_weekday': favorite_weekday, 'favorite_time_of_day': favorite_time_of_day,
                    **coffee_counts
                }
                input_df = pd.DataFrame(data, index=[0])
                for col in required_model_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df_aligned = input_df[required_model_cols]
                prediction_encoded = pipeline.predict(input_df_aligned)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)
                st.success(f"🎉 The model predicts the customer will buy a **{prediction_label[0]}** next!")
                st.balloons()
            else:
                st.warning("Model is not loaded. Cannot make a prediction.")


# ==============================================================================
# TAB 2: BATCH FORECASTING
# ==============================================================================
with tab2:
    # This tab's code is correct.
    st.header("Batch Forecasting with CSV File")
    st.write("Upload a CSV file with customer data to predict their next purchases.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df_original = pd.read_csv(uploaded_file)
            st.write("✅ **CSV Uploaded Successfully!** Here's a preview:")
            st.dataframe(batch_df_original.head())
            required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
            core_cols = ['total_visits', 'total_spent', 'days_since_last_visit', 'avg_spent_per_visit', 'favorite_weekday', 'favorite_time_of_day']
            
            if all(col in batch_df_original.columns for col in core_cols):
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Preparing data and predicting..."):
                        batch_df_processed = batch_df_original.copy()
                        for col in required_model_cols:
                            if col not in batch_df_processed.columns:
                                batch_df_processed[col] = 0
                        batch_df_aligned = batch_df_processed[required_model_cols]
                        predictions_encoded = pipeline.predict(batch_df_aligned)
                        predictions_labels = label_encoder.inverse_transform(predictions_encoded)
                        results_df = batch_df_original.copy()
                        results_df['predicted_next_purchase'] = predictions_labels
                        st.session_state.batch_results = results_df # This line now safely updates the initialized state
                        st.success("✅ **Batch Prediction Complete!**")
                        st.dataframe(results_df)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name='predicted_purchases.csv',
                            mime='text/csv',
                        )
            else:
                st.error("❌ **CSV Error:** Your file is missing one or more essential columns.")
                st.write("**Essential columns required:**")
                st.json(core_cols)
                st.write("**Columns found in your file:**")
                st.json(batch_df_original.columns.tolist())
                st.info("Please check your CSV file. The very first line must be the header, and it must contain all the essential column names.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ==============================================================================
# TAB 3: INVENTORY & INSIGHTS
# ==============================================================================
with tab3:
    st.header("Inventory Demand Forecast & Customer Insights")
    st.write("This tab uses the results from the 'Batch Forecasting' tab to generate insights.")
    
    # This check will now work safely because st.session_state.batch_results was initialized to None
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predicted Demand Forecast")
            forecast_period = st.radio(
                "Select Forecast Period",
                ("Next Week", "Next Month"),
                horizontal=True
            )
            multiplier = 4 if forecast_period == "Next Month" else 1
            all_coffee_types = label_encoder.classes_
            predicted_counts = results_df['predicted_next_purchase'].value_counts()
            demand_forecast = pd.Series(0, index=all_coffee_types)
            demand_forecast.update(predicted_counts)
            demand_forecast = (demand_forecast * multiplier).astype(int)
            demand_forecast_df = demand_forecast.reset_index()
            demand_forecast_df.columns = ['Coffee Type', 'Predicted Number of Sales']
            st.write(f"Based on the predicted next purchase for each customer in your file, extrapolated for the {forecast_period.lower()}.")
            st.bar_chart(demand_forecast_df.set_index('Coffee Type'), color="#FF8C00")

        with col2:
            st.subheader("Historical Customer Favorites")
            st.write("Based on the total purchase counts from your uploaded file.")

            count_cols = [col for col in results_df.columns if col.startswith('count_')]
            if count_cols:
                historical_favorites = results_df[count_cols].sum()
                historical_favorites.index = [idx.replace('count_', '').replace('_', ' ').title() for idx in historical_favorites.index]
                
                favorites_df = historical_favorites.reset_index()
                favorites_df.columns = ['Coffee Type', 'Total Historical Purchases']
                
                chart = alt.Chart(favorites_df).mark_bar(color='#008080').encode(
                    x=alt.X('Coffee Type', type='nominal', sort='-y'),
                    y=alt.Y('Total Historical Purchases', type='quantitative'),
                    tooltip=['Coffee Type', 'Total Historical Purchases']
                )
                
                st.altair_chart(chart, use_container_width=True)

                st.write("This chart shows which drinks have been the most popular historically among the customers in your uploaded file.")
            else:
                st.warning("Could not find historical purchase count columns (e.g., 'count_latte') in the uploaded file.")
    else:
        st.info("ℹ️ Please upload a CSV and run a batch prediction in the **'Batch Forecasting'** tab to see insights here.")
