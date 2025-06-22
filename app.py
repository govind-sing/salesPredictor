# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page title
st.title('Marketing Spend to Sales Predictor')

# Load model and scaler
model = joblib.load('outputs/linear_regression_model.pkl')
scaler = joblib.load('outputs/standard_scaler.pkl')

# Create input sliders
st.header('Enter Advertising Budgets ($)')
tv = st.slider('TV Budget', min_value=0.0, max_value=300.0, value=150.0, step=0.1)
radio = st.slider('Radio Budget', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
newspaper = st.slider('Newspaper Budget', min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Prepare input for prediction as a DataFrame
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]

# Display prediction
st.header('Predicted Sales')
st.write(f'Estimated Sales: ${prediction:.2f} thousand')

# Visualize input budgets
st.header('Budget Allocation')
fig, ax = plt.subplots()
budgets = [tv, radio, newspaper]
labels = ['TV', 'Radio', 'Newspaper']
ax.bar(labels, budgets, color=['blue', 'green', 'red'])
ax.set_ylabel('Budget ($)')
ax.set_title('Advertising Budget Allocation')
st.pyplot(fig)

# Save input and prediction to Excel
results = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper],
    'Predicted Sales': [prediction]
})
results.to_excel('outputs/user_predictions.xlsx', index=False)

# Provide download button for Excel
with open('outputs/user_predictions.xlsx', 'rb') as f:
    st.download_button('Download Prediction', f, file_name='user_predictions.xlsx')