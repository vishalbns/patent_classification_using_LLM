"""
This FastAPI application serves as the backend for a patent classification system. 
It uses a fine-tuned model based on Llama-3.2-1B and PEFT (Parameter-Efficient Fine-Tuning) for sequence classification. 
The app processes incoming requests with patent descriptions, tokenizes the input text, and predicts the patent category using the loaded model. 
The predicted category is then returned as a response to the client, with class labels mapped to human-readable names. 
The application is designed to handle HTTP POST requests at the "/predict" endpoint.
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, PeftConfig
from pydantic import BaseModel
import torch

# Load the base model (Llama-3.2-1B)
base_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("../final_model")

# Load the PEFT adapter model configuration
peft_config = PeftConfig.from_pretrained("../final_model")
print(peft_config)

# Apply the PEFT model on top of the base model
model = get_peft_model(base_model, peft_config).to("cpu")

app = FastAPI()

# Allow CORS from all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your Streamlit frontend domain here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

# Define the class names corresponding to the numeric labels

class_names = [
    'Human Necessities', 'Performing Operations; Transporting', 'Chemistry; Metallurgy', 'Textiles; Paper', 
    'Fixed Constructions', 'Mechanical Engineering; Lightning; Heating; Weapons; Blasting', 'Physics', 
    'Electricity', 'General tagging of new or cross-sectional technology'
]

@app.post("/predict")
def predict(request: TextRequest):
    # Tokenize the input text
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        # Get the model outputs
        outputs = model(**inputs)
    
    # Get the predicted class index
    prediction_idx = outputs.logits.argmax(-1).item()

    # Get the predicted class label name
    prediction_label = class_names[prediction_idx]

    return {"prediction": prediction_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
