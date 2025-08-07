#!/usr/bin/env python3
"""
NAVUS Mistral 7B Fine-tuning Script - OPTIMIZED
Fine-tunes Mistral 7B Instruct on Canadian credit card advisory data
🚀 Optimized for cost, speed, and accuracy
"""

import os
import json
import torch
import gc
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAVUSTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        # Initialize Weights & Biases (optional - can be disabled)
        try:
            wandb.init(
                project="navus-credit-advisor", 
                name="mistral-7b-optimized",
                config={
                    "model": model_name,
                    "dataset": "canadian-credit-cards",
                    "task": "credit-card-advisory",
                    "optimization": "memory-efficient"
                }
            )
            logger.info("✅ W&B initialized")
        except:
            logger.warning("⚠️  W&B not available, continuing without logging")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with optimized 4-bit quantization"""
        logger.info("🔄 Loading model and tokenizer...")
        
        # Optimized 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer with proper settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=True  # Use fast tokenizer for speed
        )
        
        # Set proper special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False  # Disable cache to save memory during training
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("✅ Model and tokenizer loaded with optimizations")
    
    def setup_lora(self):
        """Setup optimized LoRA configuration"""
        logger.info("🔄 Setting up LoRA...")
        
        # Optimized LoRA config for financial domain
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Increased rank for better performance on domain-specific tasks
            lora_alpha=64,  # Higher alpha for stronger adaptation
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",     # MLP layers
            ],
            modules_to_save=None,  # Don't save additional modules to save space
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"📊 Trainable params: {trainable_params:,} || "
                   f"All params: {all_param:,} || "
                   f"Trainable%: {100 * trainable_params / all_param:.2f}%")
        
        logger.info("✅ LoRA setup complete")
    
    def load_and_prepare_dataset(self, data_path="navus_chat_format.jsonl"):
        """Load and prepare dataset with optimized formatting"""
        logger.info("🔄 Loading and preparing dataset...")
        
        # Load dataset
        dataset = load_dataset("json", data_files=data_path, split="train")
        logger.info(f"📁 Loaded {len(dataset)} examples")
        
        def format_chat_template(example):
            """Optimized chat template formatting for Mistral"""
            messages = example["messages"]
            
            # Build conversation with proper Mistral format
            conversation = ""
            for message in messages:
                if message["role"] == "user":
                    conversation += f"[INST] {message['content']} [/INST]"
                elif message["role"] == "assistant":
                    conversation += f" {message['content']}{self.tokenizer.eos_token}"
            
            return {"text": conversation}
        
        # Apply formatting
        formatted_dataset = dataset.map(
            format_chat_template, 
            remove_columns=dataset.column_names,
            desc="Formatting conversations"
        )
        
        def tokenize_function(examples):
            """Tokenize with proper settings"""
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=768,  # Increased for longer conversations
                padding=False,  # Don't pad during tokenization
                return_overflowing_tokens=False,
            )
            
            # Labels are the same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Filter out examples that are too short (less than 10 tokens)
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) >= 10,
            desc="Filtering short examples"
        )
        
        # Split dataset (85/15 for better eval)
        dataset_split = tokenized_dataset.train_test_split(
            test_size=0.15, 
            seed=42,
            shuffle=True
        )
        
        self.train_dataset = dataset_split["train"]
        self.eval_dataset = dataset_split["test"]
        
        logger.info(f"✅ Dataset prepared - Train: {len(self.train_dataset)}, "
                   f"Eval: {len(self.eval_dataset)}")
        
        return self.train_dataset, self.eval_dataset
    
    def train(self, output_dir="./navus_mistral_finetuned"):
        """Fine-tune with optimized training arguments"""
        logger.info("🚀 Starting optimized training...")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training schedule
            num_train_epochs=2,  # Reduced epochs to save cost/time
            max_steps=-1,
            
            # Batch sizes (optimized for memory)
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Increased for effective batch size
            
            # Optimizer settings
            optim="paged_adamw_32bit",
            learning_rate=1e-4,  # Slightly lower LR for stability
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,  # Longer warmup for stability
            
            # Memory optimizations
            dataloader_pin_memory=False,
            gradient_checkpointing=True,  # Save memory at cost of speed
            bf16=True,
            fp16=False,
            
            # Regularization
            max_grad_norm=0.3,
            
            # Logging and saving
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,  # Keep only 2 checkpoints to save space
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Misc
            remove_unused_columns=False,
            group_by_length=True,  # Group similar length sequences
            report_to=["wandb"] if wandb.run else [],
            run_name="navus-mistral-optimized",
            seed=42,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Start training
        logger.info("🔥 Beginning training...")
        train_result = trainer.train()
        
        # Save the model
        logger.info("💾 Saving model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        with open(f"{output_dir}/training_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"✅ Training complete! Model saved to {output_dir}")
        logger.info(f"📊 Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        
        return trainer

def main():
    """Main training function with error handling"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This script requires GPU for training.")
        return False
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🎯 Using GPU: {gpu_name}")
    logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 16:
        logger.warning("⚠️  GPU has less than 16GB memory. Training may be slow.")
    
    try:
        # Initialize trainer
        trainer = NAVUSTrainer()
        
        # Setup model with optimizations
        trainer.setup_model_and_tokenizer()
        trainer.setup_lora()
        
        # Load and prepare dataset
        trainer.load_and_prepare_dataset()
        
        # Train the model
        trained_model = trainer.train()
        
        logger.info("🎉 NAVUS Credit Advisor training completed successfully!")
        logger.info("📁 Model files saved to: ./navus_mistral_finetuned")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)