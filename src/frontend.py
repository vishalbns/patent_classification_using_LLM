import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://localhost:8000/predict"

# Streamlit webpage title
st.title("Patent Classification Prediction")

# Input form for user text
user_input = st.text_area("Enter the patent description:")

# Button to trigger the prediction
if st.button("Get Prediction"):
    if user_input.strip():
        # Send the user input to the FastAPI endpoint
        response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            # Extract the prediction result from the API response
            prediction = response.json().get("prediction")
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error: Unable to get prediction from the API.")
    else:
        st.warning("Please enter some text to predict.")
