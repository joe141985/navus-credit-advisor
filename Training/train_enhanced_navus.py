"""
Enhanced NAVUS Training Script
Optimized for MacBook training with advanced financial decision-making capabilities
"""

import json
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNAVUSTrainer:
    """Enhanced training class for NAVUS financial advisor model"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", output_dir="./enhanced_navus_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"  # MacBook optimization
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
    def load_training_data(self):
        """Load and combine all training datasets"""
        datasets = []
        
        # Load original NAVUS data
        try:
            with open('navus_alpaca_format.json', 'r') as f:
                original_data = json.load(f)
                datasets.extend(original_data)
                logger.info(f"Loaded {len(original_data)} original examples")
        except FileNotFoundError:
            logger.warning("Original NAVUS data not found")
        
        # Load enhanced debt payoff data
        try:
            with open('enhanced_debt_payoff_dataset.json', 'r') as f:
                enhanced_data = json.load(f)
                datasets.extend(enhanced_data)
                logger.info(f"Loaded {len(enhanced_data)} enhanced examples")
        except FileNotFoundError:
            logger.warning("Enhanced debt payoff data not found")
        
        # Generate additional synthetic data
        synthetic_data = self.generate_synthetic_training_data()
        datasets.extend(synthetic_data)
        logger.info(f"Generated {len(synthetic_data)} synthetic examples")
        
        logger.info(f"Total training examples: {len(datasets)}")
        return datasets
    
    def generate_synthetic_training_data(self):
        """Generate synthetic training data for various financial scenarios"""
        synthetic_examples = []
        
        # Card comparison scenarios
        card_comparisons = [
            {
                "instruction": "Compare RBC Cashback vs BMO CashBack for everyday spending",
                "input": "monthly_spend: $1500, categories: groceries_gas_dining",
                "output": "**Card Comparison Analysis:**\n\n**RBC Cashback Preferred Mastercard:**\n- Annual fee: $0\n- Cashback: 1% unlimited\n- Monthly rewards: $15\n- Annual rewards: $180\n\n**BMO CashBack Mastercard:**\n- Annual fee: $0\n- Cashback: 3% groceries (up to $500/month), 1% other\n- Monthly rewards: ~$22.50\n- Annual rewards: ~$270\n\n📊 **Winner: BMO CashBack (+$90/year)**\n🎯 **Strategy Score: 8.5/10**\n\n**Recommendation:** BMO CashBack for higher grocery rewards, especially if you spend $400+ monthly on groceries."
            },
            {
                "instruction": "Should I get a premium card with $150 annual fee?",
                "input": "monthly_spend: $3000, travel_frequency: moderate, current_card: basic",
                "output": "**Premium Card Analysis:**\n\n**Break-even Analysis:**\n- Annual fee: $150\n- Required additional rewards: $150+\n- Monthly spend needed: $3000\n\n**Premium Benefits (BMO World Elite):**\n- 2x points vs 1x basic card\n- Additional annual value: $360\n- Travel insurance: $200+ value\n- Airport lounge: $300+ value\n\n📊 **Total Value: $860 vs $150 cost**\n🎯 **Strategy Score: 9.2/10**\n\n**Recommendation:** YES! Premium card justified with your spending level. Net benefit: $710+ annually."
            }
        ]
        
        # Budget planning scenarios
        budget_scenarios = [
            {
                "instruction": "Help me create a budget that includes debt payoff and emergency fund",
                "input": "monthly_income: $4500, fixed_expenses: $2800, debt_payment: $400, wants_emergency_fund: yes",
                "output": "**Comprehensive Budget Plan:**\n\n**Income:** $4,500\n**Fixed Expenses:** $2,800 (62%)\n**Available for Goals:** $1,700\n\n**Recommended Allocation:**\n💳 **Debt Payment:** $600 (35% of available)\n💰 **Emergency Fund:** $300 (18% of available)\n🎯 **Discretionary:** $800 (47% of available)\n\n**📊 Goal Timeline:**\n- Emergency Fund ($3,000): 10 months\n- Debt payoff acceleration: 33% faster\n- Total strategy score: 9.1/10\n\n**💡 Optimization Tips:**\n- Automate transfers on payday\n- Use high-yield savings for emergency fund\n- Apply raises/bonuses to debt first"
            }
        ]
        
        # Credit building scenarios
        credit_scenarios = [
            {
                "instruction": "I'm new to Canada. How do I build credit from scratch?",
                "input": "newcomer_to_canada: yes, income: $55000, no_credit_history: yes",
                "output": "**Canadian Credit Building Strategy:**\n\n**Phase 1: Foundation (Month 1-2)**\n✅ Open bank account with Big 5 bank\n✅ Apply for RBC Secured Visa ($0 fee)\n✅ Deposit $500-1000 security\n✅ Use card for small purchases ($100/month)\n\n**Phase 2: Establishment (Month 3-8)**\n✅ Keep utilization under 10%\n✅ Pay in full every month\n✅ Add telecom/utility bills to credit file\n✅ Monitor credit score monthly\n\n**Phase 3: Growth (Month 9-12)**\n✅ Apply for unsecured card (RBC Cashback)\n✅ Increase credit limits\n✅ Consider adding authorized user\n\n**📊 Expected Timeline:**\n```\nMonth 3:  Score appears (~650)\nMonth 6:  Score ~680-700\nMonth 12: Score ~720-750\n```\n\n🎯 **Success Rate: 95% with this strategy**"
            }
        ]
        
        synthetic_examples.extend(card_comparisons)
        synthetic_examples.extend(budget_scenarios) 
        synthetic_examples.extend(credit_scenarios)
        
        return synthetic_examples
    
    def prepare_dataset(self, training_data):
        """Prepare dataset for training"""
        # Format data for causal language modeling
        formatted_data = []
        
        for example in training_data:
            # Create conversation format
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            if input_text:
                conversation = f"Human: {instruction}\nContext: {input_text}\nNAVUS: {output}"
            else:
                conversation = f"Human: {instruction}\nNAVUS: {output}"
            
            formatted_data.append({"text": conversation})
        
        # Convert to dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return tokenized_dataset
    
    def train_model(self, dataset, epochs=3, batch_size=4, learning_rate=5e-5):
        """Train the enhanced NAVUS model"""
        
        # MacBook-optimized training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,  # Smaller batch for MacBook
            gradient_accumulation_steps=4,  # Simulate larger batch
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=False,  # Disable for MPS
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",  # Disable for faster training
            save_total_limit=2,
            prediction_loss_only=True,
            dataloader_pin_memory=False,  # Better for MPS
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")
    
    def test_model(self):
        """Test the trained model with sample queries"""
        test_queries = [
            "I have $3000 debt at 22% interest. Should I get a balance transfer card?",
            "What's the best no-fee credit card for building credit?",
            "Help me create a debt payoff plan for $8000 with $400/month budget",
            "Compare RBC vs TD credit cards for travel rewards"
        ]
        
        logger.info("Testing trained model...")
        self.model.eval()
        
        for query in test_queries:
            prompt = f"Human: {query}\nNAVUS:"
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + 200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n🤖 Query: {query}")
            print(f"🎯 Response: {response[len(prompt):]}\n" + "="*80)

def main():
    """Main training pipeline"""
    print("🚀 Starting Enhanced NAVUS Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = EnhancedNAVUSTrainer()
    
    # Load training data
    training_data = trainer.load_training_data()
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(training_data)
    
    # Train model (optimized for MacBook)
    trainer.train_model(dataset, epochs=2, batch_size=2)  # Conservative settings
    
    # Test model
    trainer.test_model()
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()