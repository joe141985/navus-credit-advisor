#!/usr/bin/env python3
"""
NAVUS Mistral 7B Fine-tuning Script
Fine-tunes Mistral 7B Instruct on Canadian credit card advisory data
"""

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

class NAVUSTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        # Initialize Weights & Biases (optional)
        wandb.init(
            project="navus-credit-advisor", 
            name="mistral-7b-finetune",
            config={
                "model": model_name,
                "dataset": "canadian-credit-cards",
                "task": "credit-card-advisory"
            }
        )
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with 4-bit quantization"""
        print("🔄 Loading model and tokenizer...")
        
        # 4-bit quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        print("✅ Model and tokenizer loaded")
    
    def setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        print("🔄 Setting up LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter for LoRA scaling
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ LoRA setup complete")
    
    def load_and_prepare_dataset(self, data_path="navus_chat_format.jsonl"):
        """Load and prepare the training dataset"""
        print("🔄 Loading dataset...")
        
        # Load the JSONL file
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        def format_chat_template(example):
            """Format the chat template for Mistral"""
            messages = example["messages"]
            
            # Mistral chat format
            formatted_text = ""
            for message in messages:
                if message["role"] == "user":
                    formatted_text += f"[INST] {message['content']} [/INST]"
                elif message["role"] == "assistant":
                    formatted_text += f" {message['content']}"
            
            # Tokenize
            tokenized = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Apply formatting
        self.dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
        
        # Split into train/validation (90/10 split)
        dataset_split = self.dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = dataset_split["train"]
        self.eval_dataset = dataset_split["test"]
        
        print(f"✅ Dataset prepared - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
        
        return self.train_dataset, self.eval_dataset
    
    def train(self, output_dir="./navus_mistral_finetuned"):
        """Fine-tune the model"""
        print("🚀 Starting training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb",
            evaluation_strategy="steps",
            eval_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training complete! Model saved to {output_dir}")
        
        return trainer

def main():
    """Main training function"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. This script requires GPU for training.")
        return
    
    print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    trainer = NAVUSTrainer()
    
    # Setup model and LoRA
    trainer.setup_model_and_tokenizer()
    trainer.setup_lora()
    
    # Load dataset
    trainer.load_and_prepare_dataset()
    
    # Train the model
    trained_model = trainer.train()
    
    print("🎉 NAVUS Credit Advisor training complete!")
    print("📁 Model saved to: ./navus_mistral_finetuned")

if __name__ == "__main__":
    main()
