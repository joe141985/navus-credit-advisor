#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - GPT-OSS-120B Enhanced
Uses OpenAI's GPT-OSS-120B via Hugging Face for advanced financial advice
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
import os
import json
import asyncio
from datetime import datetime
import httpx
import time
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import seaborn as sns
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API - GPT-OSS Enhanced", 
    version="4.0.0",
    description="AI-powered Canadian credit card advisor with GPT-OSS-120B"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    user_profile: Optional[Dict] = {}

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    suggested_questions: Optional[List[str]] = []
    processing_time: Optional[float] = None
    chart_data: Optional[str] = None  # Base64 encoded chart

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    memory_usage: Optional[str] = None

class GPTOSSNavusModel:
    def __init__(self):
        self.loaded = False
        self.model_type = "openai/gpt-oss-120b"
        self.dataset_df = None
        self.hf_token = os.getenv("HF_TOKEN", "")  # Get from environment
        self.hf_url = "https://api-inference.huggingface.co/models/openai/gpt-oss-120b"
        
        # Load datasets
        self.load_dataset()
        self.load_training_data()
    
    def load_dataset(self):
        """Load credit card dataset from embedded data"""
        sample_cards = [
            {
                "name": "American Express Cobalt Card", 
                "issuer": "American Express", 
                "category": "rewards", 
                "annual_fee": 0, 
                "rewards_rate": "5x on groceries/dining, 2x on travel/gas",
                "best_for": "dining and groceries",
                "income_requirement": 12000,
                "features": "No foreign transaction fees, mobile device insurance"
            },
            {
                "name": "RBC Avion Visa Infinite", 
                "issuer": "RBC", 
                "category": "travel", 
                "annual_fee": 120, 
                "rewards_rate": "1.25x on all purchases",
                "best_for": "flexible travel rewards",
                "income_requirement": 80000,
                "features": "Travel insurance, airport lounge access, concierge service"
            },
            {
                "name": "TD Cash Back Visa Infinite", 
                "issuer": "TD", 
                "category": "cashback", 
                "annual_fee": 139, 
                "rewards_rate": "3% on groceries/gas, 1% on other purchases",
                "best_for": "high cashback rates",
                "income_requirement": 60000,
                "features": "Mobile device insurance, purchase protection"
            },
            {
                "name": "Scotiabank Gold American Express", 
                "issuer": "Scotiabank", 
                "category": "rewards", 
                "annual_fee": 139, 
                "rewards_rate": "5x on groceries/gas/dining, 1x on other purchases",
                "best_for": "high rewards on essentials",
                "income_requirement": 60000,
                "features": "Scene+ points, purchase protection, extended warranty"
            },
            {
                "name": "BMO Air Miles World Elite", 
                "issuer": "BMO", 
                "category": "travel", 
                "annual_fee": 120, 
                "rewards_rate": "1x Air Miles per $15 spent",
                "best_for": "Air Miles collectors",
                "income_requirement": 80000,
                "features": "Priority check-in, first bag free, travel insurance"
            },
            {
                "name": "CIBC Dividend Visa Infinite", 
                "issuer": "CIBC", 
                "category": "cashback", 
                "annual_fee": 99, 
                "rewards_rate": "4% on gas/groceries (first $20k), 2% on restaurants/transportation",
                "best_for": "everyday cashback",
                "income_requirement": 60000,
                "features": "Mobile device insurance, purchase protection"
            },
            {
                "name": "RBC Student Visa", 
                "issuer": "RBC", 
                "category": "student", 
                "annual_fee": 0, 
                "rewards_rate": "1x RBC Rewards points",
                "best_for": "building credit history",
                "income_requirement": 0,
                "features": "No income requirement, fraud protection"
            },
            {
                "name": "Capital One Guaranteed Secured Mastercard", 
                "issuer": "Capital One", 
                "category": "secured", 
                "annual_fee": 59, 
                "rewards_rate": "None",
                "best_for": "rebuilding credit",
                "income_requirement": 0,
                "features": "Guaranteed approval, reports to credit bureaus"
            }
        ]
        
        self.dataset_df = pd.DataFrame(sample_cards)
        logger.info(f"✅ Loaded {len(self.dataset_df)} Canadian credit cards")
    
    def load_training_data(self):
        """Load the massive training dataset examples"""
        self.training_examples = []
        try:
            # Try to load the massive dataset
            dataset_paths = [
                "/Users/joebanerjee/NAVUS/Training/massive_navus_dataset_latest.json",
                "/Users/joebanerjee/NAVUS/Training/massive_navus_dataset_20250811_133622.json"
            ]
            
            for path in dataset_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.training_examples = data[:50]  # Use first 50 examples for context
                        break
            
            if self.training_examples:
                logger.info(f"✅ Loaded {len(self.training_examples)} training examples for context")
            else:
                logger.warning("⚠️ No training dataset found, using embedded examples")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load training data: {e}")
    
    async def test_hf_connection(self):
        """Test if Hugging Face API is accessible"""
        try:
            if not self.hf_token:
                logger.warning("⚠️ HF_TOKEN not set. Set it for full access.")
                self.loaded = True  # Still work without token (rate limited)
                return
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.hf_url}",
                    headers=headers,
                    timeout=5.0
                )
                if response.status_code in [200, 503]:  # 503 = model loading
                    self.loaded = True
                    logger.info("✅ GPT-OSS-120B connection successful!")
                else:
                    logger.warning(f"⚠️ HF API responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to Hugging Face: {e}")
            self.loaded = True  # Continue anyway with fallback
    
    def create_financial_chart(self, chart_type: str, data: Dict) -> str:
        """Generate financial charts and return as base64"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "debt_payoff":
                # Debt payoff timeline
                months = data.get('months', list(range(1, 25)))
                balances = data.get('balances', [5000 - (i * 200) for i in months])
                interest_paid = data.get('interest_saved', [i * 15 for i in months])
                
                ax.plot(months, balances, label='Remaining Balance', color='#FF6B6B', linewidth=2)
                ax.plot(months, interest_paid, label='Interest Saved', color='#4ECDC4', linewidth=2)
                ax.set_title('Debt Payoff Strategy', fontsize=16, fontweight='bold')
                ax.set_xlabel('Months')
                ax.set_ylabel('Amount ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "credit_score":
                # Credit score improvement
                months = data.get('months', list(range(1, 13)))
                scores = data.get('scores', [650 + (i * 10) for i in months])
                
                ax.plot(months, scores, marker='o', color='#667eea', linewidth=3, markersize=6)
                ax.set_title('Credit Score Improvement Timeline', fontsize=16, fontweight='bold')
                ax.set_xlabel('Months')
                ax.set_ylabel('Credit Score')
                ax.set_ylim(600, 800)
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "card_comparison":
                # Credit card comparison
                cards = data.get('cards', ['Card A', 'Card B', 'Card C'])
                rewards = data.get('rewards', [2.5, 1.8, 3.1])
                fees = data.get('fees', [120, 0, 139])
                
                x = np.arange(len(cards))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rewards, width, label='Rewards Rate (%)', color='#4ECDC4')
                bars2 = ax.bar(x + width/2, [f/50 for f in fees], width, label='Annual Fee ($50s)', color='#FF6B6B')
                
                ax.set_title('Credit Card Comparison', fontsize=16, fontweight='bold')
                ax.set_xlabel('Credit Cards')
                ax.set_ylabel('Rate/Fee')
                ax.set_xticks(x)
                ax.set_xticklabels(cards)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    
    def create_financial_prompt(self, user_message: str, user_profile: Dict = None) -> str:
        """Create a specialized prompt for Canadian financial advice with GPT-OSS-120B"""
        
        # Get relevant card data as context
        card_context = ""
        for _, card in self.dataset_df.iterrows():
            card_context += f"- {card['name']} ({card['issuer']}): ${card['annual_fee']} fee, {card['rewards_rate']}, best for {card['best_for']}, income req: ${card['income_requirement']}\n"
        
        # Add training examples for context
        examples_context = ""
        if self.training_examples:
            examples_context = "PREVIOUS Q&A EXAMPLES:\n"
            for example in self.training_examples[:5]:  # Use 5 examples
                if isinstance(example, dict) and 'instruction' in example and 'output' in example:
                    examples_context += f"Q: {example['instruction']}\nA: {example['output'][:200]}...\n\n"
        
        profile_context = ""
        if user_profile:
            if user_profile.get('income'):
                profile_context += f"User's approximate income: ${user_profile['income']}\n"
            if user_profile.get('location'):
                profile_context += f"User's location: {user_profile['location']}\n"
            if user_profile.get('credit_score'):
                profile_context += f"User's credit score: {user_profile['credit_score']}\n"
        
        prompt = f"""You are NAVUS, Canada's most advanced AI financial advisor specializing in credit cards, debt management, and personal finance. You have access to comprehensive Canadian banking data and 1,978 training examples.

CANADIAN CREDIT CARDS DATABASE:
{card_context}

{examples_context}

{profile_context}

USER QUESTION: {user_message}

INSTRUCTIONS:
1. Provide specific, actionable advice tailored for Canadians
2. Reference actual Canadian credit cards from the database above
3. Consider Canadian banking regulations, credit bureaus (Equifax/TransUnion Canada)
4. For debt scenarios, provide detailed payoff strategies (avalanche vs snowball)
5. Include specific numbers, timelines, and calculations when possible
6. Mention relevant chart analysis if applicable (debt timeline, credit score improvement, card comparisons)
7. Always end with 2-3 relevant follow-up questions
8. Be conversational but authoritative
9. Focus on practical, implementable strategies

CHART GENERATION HINTS:
- If discussing debt payoff: mention "debt_payoff_timeline" 
- If comparing cards: mention "card_comparison_chart"
- If discussing credit building: mention "credit_score_improvement"

RESPONSE (be detailed, specific, and Canadian-focused):"""
        
        return prompt
    
    async def generate_gptoss_response(self, prompt: str) -> tuple:
        """Generate response using GPT-OSS-120B via Hugging Face"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 800,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.hf_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        
                        # Extract chart hints and generate charts
                        chart_data = None
                        if "debt_payoff_timeline" in generated_text.lower():
                            chart_data = self.create_financial_chart("debt_payoff", {
                                "months": list(range(1, 25)),
                                "balances": [5000 - (i * 200) for i in range(1, 25)],
                                "interest_saved": [i * 15 for i in range(1, 25)]
                            })
                        elif "card_comparison" in generated_text.lower():
                            chart_data = self.create_financial_chart("card_comparison", {
                                "cards": ["Amex Cobalt", "RBC Avion", "TD Cashback"],
                                "rewards": [2.5, 1.25, 3.0],
                                "fees": [0, 120, 139]
                            })
                        elif "credit_score" in generated_text.lower():
                            chart_data = self.create_financial_chart("credit_score", {
                                "months": list(range(1, 13)),
                                "scores": [650 + (i * 8) for i in range(1, 13)]
                            })
                        
                        return generated_text, chart_data
                    
                elif response.status_code == 503:
                    # Model is loading
                    return "I'm currently loading my advanced AI model. Please try again in 30 seconds for the most sophisticated financial advice!", None
                else:
                    logger.error(f"HF API error: {response.status_code} - {response.text}")
                    return self.fallback_response(), None
                    
        except Exception as e:
            logger.error(f"Error calling GPT-OSS: {e}")
            return self.fallback_response(), None
    
    def fallback_response(self) -> str:
        """Fallback response when GPT-OSS is unavailable"""
        return """I'm experiencing high demand right now, but I can still help! 

For Canadian credit card advice:
- **Travel rewards**: RBC Avion Visa Infinite (flexible points, $120 fee)
- **Cashback**: TD Cash Back Visa Infinite (3% groceries/gas, $139 fee)  
- **Dining/Groceries**: Amex Cobalt Card (5x rewards, no annual fee)
- **Students**: RBC Student Visa (no income requirement, builds credit)

Please ask me a specific question about debt payoff, card comparisons, or financial planning!"""
    
    def extract_follow_up_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from GPT-OSS response"""
        questions = []
        lines = response.split('\n')
        for line in lines:
            if '?' in line and any(word in line.lower() for word in ['consider', 'might', 'what about', 'have you']):
                question = line.strip().strip('-').strip('*').strip()
                if len(question) > 10 and len(question) < 120:
                    questions.append(question)
        
        # Default suggestions if none found
        if not questions:
            questions = [
                "What's the best strategy to pay off multiple credit cards?",
                "Which Canadian bank offers the best credit card for my spending?",
                "How can I improve my credit score fastest in Canada?"
            ]
        
        return questions[:3]  # Return up to 3 suggestions
    
    async def generate_response(self, user_message: str, user_profile: Dict = None) -> tuple:
        """Generate intelligent response using GPT-OSS-120B"""
        start_time = time.time()
        
        try:
            if self.loaded:
                # Use GPT-OSS-120B for advanced response
                prompt = self.create_financial_prompt(user_message, user_profile)
                response, chart_data = await self.generate_gptoss_response(prompt)
                suggestions = self.extract_follow_up_questions(response)
            else:
                # Fallback response while model loads
                response = self.fallback_response()
                suggestions = [
                    "What's the best travel credit card in Canada?",
                    "Help me compare cashback vs rewards cards",
                    "How do I build credit as a newcomer to Canada?"
                ]
                chart_data = None
            
            processing_time = time.time() - start_time
            return response, processing_time, suggestions, chart_data
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time
            return self.fallback_response(), processing_time, [], None

# Initialize model
navus_model = GPTOSSNavusModel()

@app.on_event("startup")
async def startup_event():
    """Test model on startup"""
    logger.info("🚀 Starting NAVUS with GPT-OSS-120B")
    await navus_model.test_hf_connection()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NAVUS Credit Card Advisor API - GPT-OSS Enhanced", 
        "status": "active",
        "version": "4.0.0 (GPT-OSS-120B)",
        "model_type": navus_model.model_type,
        "model_loaded": navus_model.loaded,
        "features": ["Advanced AI", "Chart Generation", "Canadian Focus"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if navus_model.loaded else "loading",
        model_loaded=navus_model.loaded,
        model_type=navus_model.model_type,
        memory_usage="Cloud-optimized via Hugging Face"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with GPT-OSS-120B and charts"""
    try:
        response, processing_time, suggestions, chart_data = await navus_model.generate_response(
            request.message, 
            request.user_profile
        )
        
        return ChatResponse(
            response=response,
            suggested_questions=suggestions,
            processing_time=processing_time,
            confidence=0.95 if navus_model.loaded else 0.6,
            chart_data=chart_data
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cards")
async def get_featured_cards():
    """Get featured credit cards with enhanced data"""
    try:
        featured_cards = []
        
        if navus_model.dataset_df is not None:
            for _, card in navus_model.dataset_df.iterrows():
                featured_cards.append({
                    "name": card['name'],
                    "issuer": card['issuer'],
                    "category": card['category'],
                    "annual_fee": float(card['annual_fee']),
                    "rewards_rate": card['rewards_rate'],
                    "best_for": card['best_for'],
                    "income_requirement": int(card['income_requirement']),
                    "features": card['features']
                })
        
        return {
            "featured_cards": featured_cards,
            "total_cards_in_database": len(featured_cards),
            "training_examples": len(navus_model.training_examples),
            "powered_by": "GPT-OSS-120B via Hugging Face"
        }
        
    except Exception as e:
        logger.error(f"Error getting featured cards: {e}")
        return {"featured_cards": [], "error": str(e)}

@app.get("/model-status")
async def get_model_status():
    """Get current model status"""
    return {
        "model_loaded": navus_model.loaded,
        "model_type": navus_model.model_type,
        "hf_token_configured": bool(navus_model.hf_token),
        "training_examples": len(navus_model.training_examples),
        "features": ["GPT-OSS-120B", "Chart Generation", "Canadian Banking Data"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8002))  # Different port
    
    print("🚀 Starting NAVUS Credit Card Advisor API (GPT-OSS Enhanced)...")
    print(f"📱 API will be available on port: {port}")
    print("🤖 Using GPT-OSS-120B for advanced financial advice")
    print("📊 Chart generation enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=port)