import os
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Define the path to the combined JSONL file
file_path = '../combined_data.jsonl'  # replace with your file path

# Prepare a list to collect the data
data = []

# Read the JSONL file
with open(file_path, 'r') as file:
    for line in file:
        entry = json.loads(line)
        question = entry.get('question')
        answer = entry.get('answer')
        
        if question and answer:
            context = entry.get('context', '')  # Assuming 'context' field is in the JSONL
            # Append to the data list
            data.append(f"{context} [QUESTION] {question} [ANSWER] {answer}")

# Convert list to DataFrame
df = pd.DataFrame(data, columns=['text'])

# Initialize Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))  # Resize the model's token embeddings

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Prepare Dataset
train_data = Dataset.from_pandas(df)
tokenized_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator,
)

# Fine-Tuning the Model
trainer.train()

# Save the Fine-Tuned Model
trainer.save_model('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

