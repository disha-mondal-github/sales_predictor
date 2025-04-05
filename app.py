import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('sales.pkl')

st.set_page_config(page_title="📈 Sales Predictor", layout="centered")
st.title("📈 Advertising Sales Prediction App")
st.write("Enter ad budgets below to predict **Sales** using a Linear Regression model.")

# Input fields for new data
tv = st.number_input("📺 TV Budget (in thousands)", min_value=0.0, value=17.2, step=0.1)
radio = st.number_input("📻 Radio Budget (in thousands)", min_value=0.0, value=45.9, step=0.1)
newspaper = st.number_input("📰 Newspaper Budget (in thousands)", min_value=0.0, value=69.3, step=0.1)

if st.button("Predict Sales"):
    input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Sales: **{prediction:.2f}** (in thousands)")
