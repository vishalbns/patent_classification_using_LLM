from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, PeftConfig
from pydantic import BaseModel
import torch

# Load the base model (Llama-3.2-1B)
base_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./final_model")

# Load the PEFT adapter model configuration
peft_config = PeftConfig.from_pretrained("./final_model")
print(peft_config)

# Apply the PEFT model on top of the base model
model = get_peft_model(base_model, peft_config).to("cpu")

app = FastAPI()

# Create a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

# Define the class names corresponding to the numeric labels
class_names = [
    "A61 (Human Necessities)", "A01 (Agriculture)", "G06 (Computing; Calculation; Counting)",
    "H01 (Basic Electric Elements)", "B01 (Physical or Chemical Processes)", "C07 (Organic Chemistry)",
    "G08 (Signaling)", "F16 (Engineering Elements)", "H04 (Electric Communication Techniques)"
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
