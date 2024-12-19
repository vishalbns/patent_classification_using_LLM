"""
This script fine-tunes a sequence classification model using a dataset of patent descriptions. 
It employs the Llama-3.2-1B model with PEFT (Parameter-Efficient Fine-Tuning) using LoRA (Low-Rank Adaptation) to 
reduce the number of trainable parameters from 100% to 0.07%, improving computational efficiency. 
The dataset is preprocessed and tokenized for training and evaluation. 
LoRA configuration is applied to specific model layers, enabling lightweight fine-tuning. 
Metrics like accuracy and F1-score are computed to evaluate model performance. 
The script uses the Hugging Face Trainer API for training and evaluation, logs progress to TensorBoard, 
and saves the fine-tuned model for deployment.
"""


from datasets import load_dataset
import time
import evaluate
import pandas as pd
import numpy as np

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

'''

DATA PREP

'''

ds = load_dataset("ccdv/patent-classification", "patent")

# Print the unique labels in the dataset
label_names = ds["train"].features["label"].names
print(label_names)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
# Apply tokenization to the entire dataset
encoded_dataset = ds.map(tokenize_function, batched=True)

# Check the structure of the dataset
print(encoded_dataset)

# Split the dataset into train and eval sets
train_dataset = encoded_dataset["train"].select(range(500))
eval_dataset = encoded_dataset["validation"].select(range(50))


'''

MODEL PREP

'''

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "meta-llama/Llama-3.2-1B"
config = AutoConfig.from_pretrained(model_id)

# Change the architecture to LlamaForSequenceClassification
config.architectures = ["LlamaForSequenceClassification"]

# Add a fallback for rope_scaling attribute, if it exists
if hasattr(config, "rope_scaling") and "type" not in config.rope_scaling:
    config.rope_scaling["type"] = "default"

num_labels = len(ds["train"].features["label"].names)

# Set num_labels in the config
config.num_labels = num_labels

# Load and configure the model
label_names = ds["train"].features["label"].names

# Update model configuration with label mappings
config = AutoConfig.from_pretrained(model_id)
config.label2id = {label: idx for idx, label in enumerate(label_names)}
config.id2label = {idx: label for idx, label in enumerate(label_names)}

''' QUANTIZATION

# Use BitsAndBytesConfig for 8-bit quantization
from bitsandbytes import nn as bnb
bnb_config = bnb.BitsAndBytesConfig(
    load_in_8bit=True,  # Set this to True for 8-bit quantization
    quantization_method="nf4",  # You can use 'int4' or 'nf4' for quantization
)
'''
# Load the sequence classification model
original_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    config=config,  # Pass the config with num_labels and updated architecture
    torch_dtype=torch.float32,  # Use float16 precision for memory efficiency. does not work with cpu.
    #quantization_config=bnb_config,
).to(device)

# Ensure model is on the correct device
original_model = original_model.to(device)

# Set the padding token ID if not already set
original_model.config.pad_token_id = tokenizer.pad_token_id

'''

PRINT NUMBER OF PARAMETERS TO TRAIN ORIGINAL MODEL

'''

# Function to print trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))


'''

LoRA

'''

from peft import get_peft_model, LoraConfig, TaskType
# Set up LoRA configuration
lora_config = LoraConfig(
    r=8,            # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor for LoRA layers
    target_modules=["q_proj", "v_proj"],  # LoRA applies to these layers
    lora_dropout=0.1,  # Dropout in LoRA layers
    bias="none",  # Bias term handling
)

# Apply LoRA to the original model and move it to the appropriate device
peft_model = get_peft_model(original_model, lora_config)
peft_model.to(device)  # Ensure it's moved to the correct device
peft_model.config.pad_token_id = tokenizer.pad_token_id

'''

PRINT NUMBER OF PARAMETERS TO TRAIN LORA

'''

print(print_number_of_trainable_model_parameters(peft_model))

'''

# Force model to use CPU
device = torch.device('cpu')

# Load model and move to the correct device
original_model = original_model.to(device)
peft_model = peft_model.to(device)

# Check which device it's on
print(f"Model is on {peft_model.device}")
'''


'''

SET MODEL TRAIN PARAMETERS

'''

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

#Compute metrics function is not being called by trainer because the peft_model is used not original model. 
#It is a work in progress issue.
def compute_metrics(eval_pred):
    print("Inside compute_metrics function.")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)  # Convert logits to predicted labels
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy
    f1 = f1_score(labels, predictions, average="weighted")  # Calculate weighted F1 score
    print(f"Predictions: {predictions}")
    print(f"Labels: {labels}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return {"accuracy": accuracy, "f1": f1}

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=1,              # Number of training epochs
    per_device_train_batch_size=1,   # Batch size for training
    per_device_eval_batch_size=1,    # Batch size for evaluation
    learning_rate=1e-3,
    logging_dir="./logs",            # Directory for storing logs
    logging_steps=25,                # Frequency of logging
    report_to="tensorboard",         # Log to TensorBoard
    eval_strategy="steps",           # Evaluate every 10 steps
    eval_steps=25,
    do_train=True,
    do_eval=True,
    save_strategy="epoch",           # Save only at the end of each epoch
    save_total_limit=25,              # Keep only the last 2 checkpoints
)

trainer = Trainer(
    model=peft_model,                    # The pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=eval_dataset,           # Evaluation dataset
    tokenizer=tokenizer,                 # Tokenizer
    compute_metrics=compute_metrics,     # Custom metrics
)


'''

TRAIN MODEL

'''

trainer.train()
print(f"Trainer: {trainer}")
'''

EVALUATE MODEL

'''

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

'''

SAVE MODEL

'''

# Save the final model
trainer.save_model("../final_model")