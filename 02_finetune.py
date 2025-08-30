# # train_lora.py
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# import torch

MODEL = "Qwen/Qwen3-0.6B" #"meta-llama/Llama-3.2-3B" #"mistralai/Mistral-7B-Instruct-v0.2"  # or your code model
DATA = "wiki_training_synthetic_training_data.jsonl"                     # your JSONL
VAL  = "wiki_eval_synthetic_training_data.jsonl"

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    IntervalStrategy
)
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.model_selection import train_test_split

class JSONLDataset(Dataset):
    """Custom Dataset class for JSONL data"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512, task_type="classification"):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # Load JSONL file
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task_type == "classification":
            # For classification tasks
            text = item['text']  # Adjust field name as needed
            label = item['label']  # Adjust field name as needed
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        elif self.task_type == "causal_lm":
            # For causal language modeling (text generation)
            text = item['text']  # Your full text
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()  # For causal LM, labels = input_ids
            }

def load_and_prepare_data(jsonl_file, tokenizer, task_type="classification"):
    """Load JSONL data and prepare for training"""
    
    # Read JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(data)
    
    if task_type == "classification":
        # Split into train/validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label'].tolist()
        )
        
        # Tokenize the data
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        
        # Create Hugging Face datasets
        train_dataset = HFDataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        val_dataset = HFDataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        # Apply tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, val_dataset
    
    elif task_type == "causal_lm":
        print(df)
        # For causal language modeling
        texts = df #df['text'].tolist()
        train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
        
        def tokenize_function(examples):
            # For causal LM, we tokenize and set labels = input_ids
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        train_dataset = HFDataset.from_dict({'text': train_texts})
        val_dataset = HFDataset.from_dict({'text': val_texts})
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, val_dataset

def fine_tune_model(
    model_name=MODEL,
    jsonl_file="wiki_synthetic_training_data.jsonl",
    task_type="classification",
    num_labels=2,
    output_dir="./outputs",
    num_epochs=3,
    learning_rate=2e-5,
    batch_size=16
):
    """
    Fine-tune a Hugging Face model with JSONL data
    
    Args:
        model_name: Name of the base model from Hugging Face Hub
        jsonl_file: Path to your JSONL dataset file
        task_type: "classification" or "causal_lm"
        num_labels: Number of classes for classification (ignored for causal_lm)
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Training batch size
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist (especially for GPT models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    elif task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError("task_type must be 'classification' or 'causal_lm'")
    
    # Prepare datasets
    train_dataset, val_dataset = load_and_prepare_data(jsonl_file, tokenizer, task_type)
    
    # Set up data collator
    if task_type == "classification":
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard logging
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    return trainer, model, tokenizer

# Example usage for different tasks:

def example_classification():
    """Example for text classification"""
    # Example JSONL format for classification:
    # {"text": "This movie is great!", "label": 1}
    # {"text": "This movie is terrible.", "label": 0}
    
    trainer, model, tokenizer = fine_tune_model(
        model_name="distilbert-base-uncased",
        jsonl_file="sentiment_data.jsonl",
        task_type="classification",
        num_labels=2,  # Binary classification
        output_dir="./sentiment-model",
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=16
    )
    
    return trainer, model, tokenizer

def example_text_generation():
    """Example for text generation/causal language modeling"""
    # Example JSONL format for text generation:
    # {"text": "Question: What is AI? Answer: Artificial Intelligence is..."}
    # {"text": "Question: How does ML work? Answer: Machine Learning works by..."}
    
    trainer, model, tokenizer = fine_tune_model(
        model_name=MODEL,
        jsonl_file="wiki_synthetic_training_data.jsonl",
        task_type="causal_lm",
        output_dir="./qa-model",
        num_epochs=3,
        learning_rate=5e-5,
        batch_size=8
    )
    
    return trainer, model, tokenizer

# Function to test the fine-tuned model
def test_model(model_path, text_input, task_type):
    """Test the fine-tuned model"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Tokenize input
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
        
        return predicted_class.item(), predictions
    
    elif task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Generate text
        inputs = tokenizer(text_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    # Example usage - uncomment the task you want to run
    
    # For classification task
    # trainer, model, tokenizer = example_classification()
    
    # For text generation task
    # trainer, model, tokenizer = example_text_generation()
    
    # Test the model after training
    result = test_model("./qa-model", "What token is?", task_type="causal_lm")
    print(f"Prediction: {result}")
    
    #pass