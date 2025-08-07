#!/usr/bin/env python3
"""
NAVUS Model Evaluation Script
Test the fine-tuned credit card advisor with realistic user questions
"""

import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import time
from typing import List, Dict, Tuple
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAVUSEvaluator:
    def __init__(self, model_path="./navus_mistral_finetuned", base_model="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.dataset_df = None
        
        # Load credit card dataset for reference
        self.load_credit_card_data()
    
    def load_credit_card_data(self):
        """Load the credit card dataset for reference answers"""
        try:
            csv_path = "/Users/joebanerjee/NAVUS/Data/master_card_dataset_cleaned.csv"
            if os.path.exists(csv_path):
                self.dataset_df = pd.read_csv(csv_path)
                logger.info(f"✅ Loaded {len(self.dataset_df)} credit cards for reference")
            else:
                logger.warning("⚠️  Credit card dataset not found, continuing without reference data")
        except Exception as e:
            logger.warning(f"⚠️  Could not load credit card dataset: {e}")
    
    def load_model(self):
        """Load the fine-tuned model or fallback to base model"""
        logger.info("🔄 Loading NAVUS model...")
        
        try:
            # Try to load fine-tuned model
            if os.path.exists(self.model_path):
                logger.info("📁 Loading fine-tuned model...")
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    padding_side="right"
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load LoRA weights
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                
                logger.info("✅ Fine-tuned model loaded successfully!")
                return True
                
            else:
                raise FileNotFoundError("Fine-tuned model not found")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not load fine-tuned model: {e}")
            logger.info("🔄 Falling back to base model...")
            
            # Fallback to base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("✅ Base model loaded as fallback")
            return False
    
    def generate_response(self, question: str, max_length: int = 300) -> Tuple[str, float]:
        """Generate response and measure inference time"""
        start_time = time.time()
        
        # Format for Mistral chat template
        prompt = f"[INST] {question} [/INST]"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[-1].strip()
        else:
            response = full_response
        
        inference_time = time.time() - start_time
        return response, inference_time
    
    def evaluate_test_questions(self) -> Dict:
        """Evaluate model on test questions"""
        logger.info("🧪 Starting evaluation with test questions...")
        
        # Test questions covering different scenarios
        test_cases = [
            {
                "category": "travel",
                "question": "Which card gives the best travel rewards in BC?",
                "expected_keywords": ["travel", "rewards", "points", "miles", "BC", "Canada"]
            },
            {
                "category": "cashback",
                "question": "Best no-fee cashback card with low interest?",
                "expected_keywords": ["cashback", "no fee", "$0", "interest", "rate"]
            },
            {
                "category": "student", 
                "question": "I'm a student looking for my first credit card with no income requirement",
                "expected_keywords": ["student", "first", "no income", "building credit"]
            },
            {
                "category": "premium",
                "question": "What's the best premium card with airport lounge access in Canada?",
                "expected_keywords": ["premium", "lounge", "airport", "travel", "benefits"]
            },
            {
                "category": "secured",
                "question": "I need to build credit history. What secured card should I get?",
                "expected_keywords": ["secured", "build credit", "credit history", "deposit"]
            },
            {
                "category": "comparison",
                "question": "Compare RBC Avion vs TD Aeroplan travel cards",
                "expected_keywords": ["RBC", "Avion", "TD", "Aeroplan", "compare", "travel"]
            },
            {
                "category": "specific",
                "question": "What's the annual fee for the American Express Platinum Card?",
                "expected_keywords": ["American Express", "Platinum", "annual fee", "$699"]
            },
            {
                "category": "income",
                "question": "Good credit cards for someone making $45,000 per year?",
                "expected_keywords": ["income", "$45", "eligible", "require"]
            }
        ]
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"📋 Test {i}/{len(test_cases)}: {test_case['category']}")
            logger.info(f"❓ Question: {test_case['question']}")
            
            # Generate response
            response, inference_time = self.generate_response(test_case["question"])
            total_time += inference_time
            
            # Check for expected keywords
            response_lower = response.lower()
            keyword_matches = sum(1 for keyword in test_case["expected_keywords"] 
                                if keyword.lower() in response_lower)
            keyword_score = keyword_matches / len(test_case["expected_keywords"])
            
            logger.info(f"💬 Response: {response[:150]}...")
            logger.info(f"⏱️  Time: {inference_time:.2f}s")
            logger.info(f"🎯 Keyword score: {keyword_score:.2f} ({keyword_matches}/{len(test_case['expected_keywords'])})")
            logger.info("-" * 80)
            
            results.append({
                "category": test_case["category"],
                "question": test_case["question"],
                "response": response,
                "inference_time": inference_time,
                "keyword_score": keyword_score,
                "keyword_matches": keyword_matches,
                "total_keywords": len(test_case["expected_keywords"])
            })
        
        # Calculate overall metrics
        avg_inference_time = total_time / len(test_cases)
        avg_keyword_score = sum(r["keyword_score"] for r in results) / len(results)
        
        evaluation_summary = {
            "total_questions": len(test_cases),
            "avg_inference_time": avg_inference_time,
            "avg_keyword_score": avg_keyword_score,
            "total_evaluation_time": total_time,
            "results": results
        }
        
        logger.info("📊 EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Questions evaluated: {len(test_cases)}")
        logger.info(f"⏱️  Average response time: {avg_inference_time:.2f}s")
        logger.info(f"🎯 Average keyword relevance: {avg_keyword_score:.2f}")
        logger.info(f"⏰ Total evaluation time: {total_time:.1f}s")
        
        return evaluation_summary
    
    def evaluate_dataset_coverage(self) -> Dict:
        """Evaluate how well the model knows the dataset"""
        if self.dataset_df is None:
            logger.warning("⚠️  No dataset loaded, skipping coverage evaluation")
            return {}
        
        logger.info("📊 Evaluating dataset coverage...")
        
        # Test specific card knowledge
        sample_cards = self.dataset_df.sample(min(5, len(self.dataset_df))).to_dict('records')
        coverage_results = []
        
        for card in sample_cards:
            card_name = card['name']
            question = f"Tell me about the {card_name}."
            
            response, inference_time = self.generate_response(question)
            
            # Check if model mentions key card attributes
            card_keywords = [
                card['issuer'].lower() if pd.notna(card['issuer']) else "",
                str(card['annual_fee']) if pd.notna(card['annual_fee']) else "",
                card['category'].lower() if pd.notna(card['category']) else "",
                card['rewards_type'].lower() if pd.notna(card['rewards_type']) else ""
            ]
            card_keywords = [kw for kw in card_keywords if kw and kw != "nan"]
            
            response_lower = response.lower()
            matches = sum(1 for kw in card_keywords if kw in response_lower)
            coverage_score = matches / len(card_keywords) if card_keywords else 0
            
            coverage_results.append({
                "card_name": card_name,
                "question": question,
                "response": response,
                "coverage_score": coverage_score,
                "matches": matches,
                "total_keywords": len(card_keywords)
            })
            
            logger.info(f"🏦 {card_name}: Coverage {coverage_score:.2f}")
        
        avg_coverage = sum(r["coverage_score"] for r in coverage_results) / len(coverage_results)
        
        return {
            "avg_coverage_score": avg_coverage,
            "tested_cards": len(coverage_results),
            "results": coverage_results
        }
    
    def save_evaluation_report(self, evaluation_results: Dict, coverage_results: Dict):
        """Save evaluation results to file"""
        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "evaluation_results": evaluation_results,
            "coverage_results": coverage_results
        }
        
        # Save as JSON
        report_file = f"/Users/joebanerjee/NAVUS/Reports/evaluation_report_{int(time.time())}.json"
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Evaluation report saved to: {report_file}")
        return report_file
    
    def interactive_test(self):
        """Interactive testing mode"""
        logger.info("💬 Interactive testing mode - Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🤔 Ask NAVUS: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input:
                    print("🤖 NAVUS:", end=" ")
                    response, inference_time = self.generate_response(user_input)
                    print(response)
                    print(f"   ⏱️ Response time: {inference_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main evaluation function"""
    
    # Initialize evaluator
    evaluator = NAVUSEvaluator()
    
    # Load model
    is_finetuned = evaluator.load_model()
    
    if is_finetuned:
        print("🎯 Evaluating FINE-TUNED model")
    else:
        print("⚠️ Evaluating BASE model (fine-tuned not available)")
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_test_questions()
    coverage_results = evaluator.evaluate_dataset_coverage()
    
    # Save results
    report_file = evaluator.save_evaluation_report(evaluation_results, coverage_results)
    
    # Print final summary
    print("\n🎉 EVALUATION COMPLETE!")
    print("=" * 50)
    print(f"📊 Overall keyword relevance: {evaluation_results['avg_keyword_score']:.2f}")
    print(f"⏱️  Average response time: {evaluation_results['avg_inference_time']:.2f}s")
    if coverage_results:
        print(f"📋 Dataset coverage: {coverage_results['avg_coverage_score']:.2f}")
    print(f"📁 Report saved: {report_file}")
    
    # Ask if user wants interactive testing
    try:
        choice = input("\n🤔 Start interactive testing? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            evaluator.interactive_test()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()