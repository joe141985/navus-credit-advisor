#!/usr/bin/env python3
"""
NAVUS LLM Training Data Generator
Converts credit card dataset into chat/QA format for fine-tuning
"""

import pandas as pd
import json
import random
import re
from typing import List, Dict, Any

class NAVUSTrainingDataGenerator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.training_data = []
        
        # Card category mappings for better responses
        self.category_descriptions = {
            'travel': 'travel rewards and benefits',
            'cashback': 'cash back rewards',
            'premium': 'premium benefits and high rewards',
            'student': 'students and young adults',
            'secured': 'credit building and guaranteed approval',
            'business': 'business expenses and commercial use',
            'basic': 'everyday spending with basic rewards',
            'rewards': 'points and rewards programs',
            'no_fee': 'no annual fee cards',
            'retail': 'retail and store-specific rewards'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and format text for better readability"""
        if pd.isna(text) or text == '':
            return 'Not specified'
        
        text = str(text)
        # Replace pipe separators with commas
        text = text.replace(' | ', ', ')
        text = text.replace('|', ', ')
        # Clean up spacing
        text = ' '.join(text.split())
        return text
    
    def format_currency(self, amount) -> str:
        """Format monetary amounts"""
        if pd.isna(amount) or amount == '' or amount == 0:
            return '$0'
        try:
            return f"${float(amount):,.0f}"
        except:
            return str(amount)
    
    def format_rate(self, rate) -> str:
        """Format interest rates"""
        if pd.isna(rate) or rate == '':
            return 'Not specified'
        try:
            rate_num = float(rate)
            return f"{rate_num}%"
        except:
            return str(rate)
    
    def get_card_summary(self, card: Dict) -> str:
        """Generate a comprehensive card summary"""
        name = card['name']
        issuer = card['issuer']
        category = card.get('category', 'general')
        annual_fee = self.format_currency(card.get('annual_fee', 0))
        rewards = self.clean_text(card.get('rewards_type', ''))
        features = self.clean_text(card.get('features', ''))
        
        summary = f"The {name} is issued by {issuer}. "
        
        if category in self.category_descriptions:
            summary += f"It's designed for {self.category_descriptions[category]}. "
        
        summary += f"Annual fee: {annual_fee}. "
        
        if rewards != 'Not specified':
            summary += f"Rewards: {rewards}. "
        
        if features != 'Not specified':
            summary += f"Key features: {features}."
        
        return summary
    
    def generate_card_recommendation_qa(self, card: Dict) -> List[Dict]:
        """Generate Q&A pairs for card recommendations"""
        qa_pairs = []
        name = card['name']
        category = card.get('category', 'general')
        annual_fee = float(card.get('annual_fee', 0)) if pd.notna(card.get('annual_fee')) and card.get('annual_fee') != '' else 0
        rewards = self.clean_text(card.get('rewards_type', ''))
        
        # Question types based on card attributes
        questions = []
        
        # Fee-based questions
        if annual_fee == 0:
            questions.extend([
                f"What's a good no-fee {category} credit card?",
                f"Can you recommend a free {category} card?",
                f"I don't want to pay an annual fee. What {category} card should I get?"
            ])
        
        # Category-based questions
        if category == 'travel':
            questions.extend([
                "What's the best travel credit card in Canada?",
                "I travel frequently. Which card should I get?",
                "Recommend a good travel rewards card"
            ])
        elif category == 'cashback':
            questions.extend([
                "What's the best cash back card?",
                "I want to earn cash back on purchases. Which card?",
                "Recommend a good cashback credit card"
            ])
        elif category == 'student':
            questions.extend([
                "What's the best credit card for students?",
                "I'm a student looking for my first credit card",
                "Good starter credit card for college students?"
            ])
        elif category == 'secured':
            questions.extend([
                "I need to build credit. What card should I get?",
                "Best secured credit card in Canada?",
                "Credit building card recommendations?"
            ])
        elif category == 'premium':
            questions.extend([
                "What's the best premium credit card?",
                "I want a high-end credit card with great benefits",
                "Luxury credit card recommendations?"
            ])
        
        # Generate responses
        for question in questions[:3]:  # Limit to 3 per card
            response = f"I'd recommend the {name}. {self.get_card_summary(card)}"
            
            qa_pairs.append({
                "instruction": question,
                "input": "",
                "output": response
            })
        
        return qa_pairs
    
    def generate_specific_card_qa(self, card: Dict) -> List[Dict]:
        """Generate Q&A pairs about specific card details"""
        qa_pairs = []
        name = card['name']
        
        # Annual fee questions
        annual_fee = self.format_currency(card.get('annual_fee', 0))
        qa_pairs.append({
            "instruction": f"What's the annual fee for the {name}?",
            "input": "",
            "output": f"The annual fee for the {name} is {annual_fee}."
        })
        
        # Rewards questions
        rewards = self.clean_text(card.get('rewards_type', ''))
        if rewards != 'Not specified':
            qa_pairs.append({
                "instruction": f"What rewards does the {name} offer?",
                "input": "",
                "output": f"The {name} offers {rewards} as its rewards program."
            })
        
        # Features questions
        features = self.clean_text(card.get('features', ''))
        if features != 'Not specified':
            qa_pairs.append({
                "instruction": f"What are the key features of the {name}?",
                "input": "",
                "output": f"The key features of the {name} include: {features}."
            })
        
        # Interest rate questions
        purchase_rate = self.format_rate(card.get('purchase_rate', ''))
        if purchase_rate != 'Not specified':
            qa_pairs.append({
                "instruction": f"What's the interest rate on the {name}?",
                "input": "",
                "output": f"The purchase interest rate for the {name} is {purchase_rate}."
            })
        
        # Welcome bonus questions
        welcome_bonus = card.get('welcome_bonus_amount', '')
        if pd.notna(welcome_bonus) and welcome_bonus != '' and float(welcome_bonus) > 0:
            bonus_formatted = self.format_currency(welcome_bonus)
            qa_pairs.append({
                "instruction": f"Does the {name} have a welcome bonus?",
                "input": "",
                "output": f"Yes, the {name} offers a welcome bonus worth {bonus_formatted}."
            })
        
        return qa_pairs
    
    def generate_comparison_qa(self) -> List[Dict]:
        """Generate comparison questions between cards"""
        qa_pairs = []
        
        # Get cards by category for comparisons
        categories = self.df['category'].value_counts()
        
        for category in ['travel', 'cashback', 'premium', 'student']:
            category_cards = self.df[self.df['category'] == category].head(3)
            
            if len(category_cards) >= 2:
                card_names = category_cards['name'].tolist()
                comparison = []
                
                for _, card in category_cards.iterrows():
                    name = card['name']
                    fee = self.format_currency(card.get('annual_fee', 0))
                    rewards = self.clean_text(card.get('rewards_type', ''))
                    
                    comparison.append(f"• {name}: {fee} annual fee, {rewards}")
                
                qa_pairs.append({
                    "instruction": f"Compare the best {category} credit cards in Canada",
                    "input": "",
                    "output": f"Here are the top {category} credit cards:\n" + "\n".join(comparison)
                })
        
        return qa_pairs
    
    def generate_general_advice_qa(self) -> List[Dict]:
        """Generate general credit card advice"""
        return [
            {
                "instruction": "How do I choose the right credit card?",
                "input": "",
                "output": "When choosing a credit card, consider: 1) Your spending habits and categories where you spend most, 2) Annual fees vs. rewards value, 3) Your credit score and income requirements, 4) Welcome bonuses and their requirements, 5) Additional benefits like insurance and purchase protection. Match the card's strengths to your financial needs and lifestyle."
            },
            {
                "instruction": "What's the difference between cash back and points rewards?",
                "input": "",
                "output": "Cash back gives you a percentage of your purchases back as statement credits or direct deposits - it's simple and flexible. Points/miles rewards accumulate in loyalty programs and can be redeemed for travel, merchandise, or sometimes cash. Points often offer higher value for travel redemptions but require more effort to maximize. Choose cash back for simplicity, points for potentially higher value if you'll use them strategically."
            },
            {
                "instruction": "Should I pay an annual fee for a credit card?",
                "input": "",
                "output": "Pay an annual fee only if the card's benefits and rewards exceed the fee cost. Calculate: (annual rewards earned + value of benefits used) - annual fee. If the result is positive and significant, the fee is worthwhile. Premium cards with high fees often provide airport lounge access, travel credits, and higher reward rates that can justify the cost for frequent travelers or high spenders."
            },
            {
                "instruction": "How can I improve my credit score with credit cards?",
                "input": "",
                "output": "To improve your credit score: 1) Pay your full balance on time every month, 2) Keep credit utilization below 30% (ideally under 10%), 3) Don't close old cards (keep credit history length), 4) Only apply for new cards when necessary to avoid hard inquiries, 5) Monitor your credit report for errors. Consider a secured card if you're building credit from scratch."
            }
        ]
    
    def generate_all_training_data(self) -> List[Dict]:
        """Generate complete training dataset"""
        print("🔄 Generating training data...")
        
        # Process each card
        for idx, row in self.df.iterrows():
            card_dict = row.to_dict()
            
            # Generate different types of Q&A
            self.training_data.extend(self.generate_card_recommendation_qa(card_dict))
            self.training_data.extend(self.generate_specific_card_qa(card_dict))
        
        # Add comparison and general advice
        self.training_data.extend(self.generate_comparison_qa())
        self.training_data.extend(self.generate_general_advice_qa())
        
        # Shuffle the data
        random.shuffle(self.training_data)
        
        print(f"✅ Generated {len(self.training_data)} training examples")
        return self.training_data
    
    def save_training_data(self, output_path: str, format_type: str = "alpaca"):
        """Save training data in specified format"""
        if format_type == "alpaca":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == "chat":
            # Convert to chat format for Mistral
            chat_data = []
            for example in self.training_data:
                chat_example = {
                    "messages": [
                        {"role": "user", "content": example["instruction"]},
                        {"role": "assistant", "content": example["output"]}
                    ]
                }
                chat_data.append(chat_example)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in chat_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved training data to {output_path} in {format_type} format")

def main():
    # Paths
    csv_path = "/Users/joebanerjee/NAVUS/Data/master_card_dataset_cleaned.csv"
    output_dir = "/Users/joebanerjee/NAVUS/Training"
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate training data
    generator = NAVUSTrainingDataGenerator(csv_path)
    generator.generate_all_training_data()
    
    # Save in different formats
    generator.save_training_data(f"{output_dir}/navus_alpaca_format.json", "alpaca")
    generator.save_training_data(f"{output_dir}/navus_chat_format.jsonl", "chat")
    
    # Print sample
    print("\n📋 Sample training examples:")
    for i, example in enumerate(generator.training_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {example['instruction']}")
        print(f"A: {example['output'][:200]}...")

if __name__ == "__main__":
    main()