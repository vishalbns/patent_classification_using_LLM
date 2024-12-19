'''
This Streamlit application provides a user-friendly interface for patent classification. 
Users can input a patent description, which is sent to a FastAPI server for processing. 
The FastAPI backend predicts the patent category using a fine-tuned classification model. 
If the input is valid, the app displays the predicted class; otherwise, it shows an appropriate error or warning message.
'''

import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://34.58.248.101:9000/predict"

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
