#!/usr/bin/env python3
"""
NAVUS Credit Advisor Inference
Load and run the fine-tuned model for credit card recommendations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

class NAVUSAdvisor:
    def __init__(self, model_path="./navus_mistral_finetuned"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("🔄 Loading NAVUS Credit Advisor...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load fine-tuned LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        print("✅ NAVUS Credit Advisor loaded!")
    
    def generate_response(self, user_question, max_length=300):
        """Generate response to user question"""
        
        # Format for Mistral chat template
        prompt = f"[INST] {user_question} [/INST]"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = full_response.split("[/INST]")[-1].strip()
        
        return response
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("💬 NAVUS Credit Card Advisor - Chat Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using NAVUS!")
                break
            
            if user_input:
                print("🤔 Thinking...")
                response = self.generate_response(user_input)
                print(f"NAVUS: {response}")
                print("-" * 50)

def main():
    """Main inference function"""
    
    # Initialize advisor
    advisor = NAVUSAdvisor()
    
    # Test with sample questions
    test_questions = [
        "What's the best no-fee travel card in Canada?",
        "I'm a student looking for my first credit card",
        "Best cashback card for groceries?"
    ]
    
    print("🧪 Testing with sample questions...")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = advisor.generate_response(question)
        print(f"A: {response}")
    
    # Start interactive chat
    advisor.chat_loop()

if __name__ == "__main__":
    main()
