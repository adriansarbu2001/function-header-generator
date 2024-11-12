import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define the model directory where the model and tokenizer are saved
model_dir = "./t5_summarization_model"

# # Load the tokenizer and model from the directory
# tokenizer = T5Tokenizer.from_pretrained(model_dir)
# model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Load the pre-trained T5-small model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define an input text
input_text = "summarize: the total number of people in a group."

# Tokenize and prepare the input for the model
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate a summary
summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=8, early_stopping=True)

# Decode the generated summary and print it
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
