#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - OpenAI GPT-4 Enhanced
Uses OpenAI's GPT-4 for advanced financial advice with chart generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Cookie, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import authentication modules
from database import get_db, create_tables, User
from auth import (
    authenticate_user, create_user, create_session, get_user_by_session,
    validate_password_strength, logout_session, AuthError, create_or_get_google_user,
    create_or_get_twitch_user
)
from google_oauth import google_oauth
from twitch_oauth import twitch_oauth

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API - GPT-4 Enhanced", 
    version="5.0.0",
    description="AI-powered Canadian credit card advisor with OpenAI GPT-4"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8081", 
        "http://127.0.0.1:8081",
        "http://192.168.1.76:8081",
        "https://navus.chat",
        "https://web-production-685ca.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

# Authentication models
class UserRegister(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    keep_logged_in: bool = False

class AuthResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    name: Optional[str] = None
    session_token: Optional[str] = None

class PasswordStrengthResponse(BaseModel):
    score: int
    max_score: int
    is_strong: bool
    issues: List[str]

class OpenAIGPT4NavusModel:
    def __init__(self):
        self.loaded = False
        self.model_type = "gpt-3.5-turbo"
        self.dataset_df = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        
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
                            self.training_examples = data[:100]  # Use first 100 examples for context
                        break
            
            if self.training_examples:
                logger.info(f"✅ Loaded {len(self.training_examples)} training examples for context")
            else:
                logger.warning("⚠️ No training dataset found, using embedded examples")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load training data: {e}")
    
    async def test_openai_connection(self):
        """Test if OpenAI API is accessible"""
        try:
            if not self.openai_api_key:
                logger.warning("⚠️ OPENAI_API_KEY not set. Please provide your OpenAI API key.")
                return
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Test with a simple request
            test_payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.openai_url,
                    headers=headers,
                    json=test_payload,
                    timeout=10.0
                )
                if response.status_code == 200:
                    self.loaded = True
                    logger.info("✅ OpenAI GPT-4 connection successful!")
                else:
                    logger.warning(f"⚠️ OpenAI API responded with status {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to OpenAI: {e}")
            self.loaded = False  # Don't continue without API access
    
    def create_financial_chart(self, chart_type: str, data: Dict) -> str:
        """Generate financial charts and return as base64"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == "debt_payoff":
                # Enhanced debt payoff timeline
                months = data.get('months', list(range(1, 37)))
                balances = data.get('balances', [10000 - (i * 280) for i in months])
                interest_paid = data.get('interest_saved', [i * 25 for i in months])
                
                ax.plot(months, balances, label='Remaining Balance', color='#FF6B6B', linewidth=3, marker='o')
                ax.plot(months, interest_paid, label='Cumulative Interest Saved', color='#4ECDC4', linewidth=3, marker='s')
                ax.set_title('Accelerated Debt Payoff Strategy', fontsize=18, fontweight='bold')
                ax.set_xlabel('Months', fontsize=14)
                ax.set_ylabel('Amount ($CAD)', fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "credit_score":
                # Enhanced credit score improvement
                months = data.get('months', list(range(1, 25)))
                scores = data.get('scores', [620 + (i * 12) for i in months])
                
                ax.plot(months, scores, marker='o', color='#667eea', linewidth=4, markersize=8)
                ax.fill_between(months, scores, alpha=0.3, color='#667eea')
                ax.set_title('Credit Score Improvement Journey', fontsize=18, fontweight='bold')
                ax.set_xlabel('Months', fontsize=14)
                ax.set_ylabel('Credit Score', fontsize=14)
                ax.set_ylim(600, 850)
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "card_comparison":
                # Enhanced credit card comparison
                cards = data.get('cards', ['Amex Cobalt', 'RBC Avion', 'TD Cashback', 'Scotia Gold'])
                rewards = data.get('rewards', [5.0, 1.25, 3.0, 5.0])
                fees = data.get('fees', [0, 120, 139, 139])
                
                x = np.arange(len(cards))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rewards, width, label='Max Rewards Rate (%)', color='#4ECDC4')
                bars2 = ax.bar(x + width/2, [f/25 for f in fees], width, label='Annual Fee ($25s)', color='#FF6B6B')
                
                ax.set_title('Canadian Credit Card Comparison', fontsize=18, fontweight='bold')
                ax.set_xlabel('Credit Cards', fontsize=14)
                ax.set_ylabel('Rate/Fee Scale', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(cards, rotation=45)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    
    def create_financial_prompt(self, user_message: str, user_profile: Dict = None) -> List[Dict]:
        """Create a specialized prompt for Canadian financial advice with GPT-4"""
        
        # Get relevant card data as context
        card_context = ""
        for _, card in self.dataset_df.iterrows():
            card_context += f"- {card['name']} ({card['issuer']}): ${card['annual_fee']} fee, {card['rewards_rate']}, best for {card['best_for']}, income req: ${card['income_requirement']}\n"
        
        # Add training examples for context
        examples_context = ""
        if self.training_examples:
            examples_context = "PREVIOUS Q&A EXAMPLES:\n"
            for example in self.training_examples[:8]:  # Use 8 examples
                if isinstance(example, dict) and 'instruction' in example and 'output' in example:
                    examples_context += f"Q: {example['instruction']}\nA: {example['output'][:300]}...\n\n"
        
        profile_context = ""
        if user_profile:
            if user_profile.get('income'):
                profile_context += f"User's approximate income: ${user_profile['income']}\n"
            if user_profile.get('location'):
                profile_context += f"User's location: {user_profile['location']}\n"
            if user_profile.get('credit_score'):
                profile_context += f"User's credit score: {user_profile['credit_score']}\n"
        
        system_message = f"""You are NAVUS, Canada's most advanced AI financial advisor specializing in credit cards, debt management, and personal finance. You have access to comprehensive Canadian banking data and 1,978 training examples.

CANADIAN CREDIT CARDS DATABASE:
{card_context}

{examples_context}

{profile_context}

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

Respond as the expert Canadian financial advisor NAVUS."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    async def generate_gpt4_response(self, messages: List[Dict]) -> tuple:
        """Generate response using OpenAI GPT-4"""
        try:
            if not self.openai_api_key:
                return "OpenAI API key not configured. Please provide your API key.", None
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 1200,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    self.openai_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result['choices'][0]['message']['content']
                    
                    # Extract chart hints and generate charts (improved detection)
                    chart_data = None
                    lower_text = generated_text.lower()
                    user_lower = messages[-1]["content"].lower()  # Get user message from messages
                    
                    logger.info(f"Chart detection - User: '{user_lower[:100]}...'")
                    logger.info(f"Chart detection - Response: '{lower_text[:100]}...'")
                    
                    # Debt payoff scenarios - MUCH MORE SPECIFIC matching
                    debt_keywords = ["pay off", "payoff", "debt strategy", "debt plan", "payment plan"]
                    debt_exclusions = ["student", "credit card for", "first credit", "best card", "which card"]
                    
                    # Only match if debt keywords AND no exclusions AND contains debt amounts or payment terms
                    has_debt_keyword = any(keyword in user_lower for keyword in debt_keywords)
                    has_exclusion = any(exclusion in user_lower for exclusion in debt_exclusions)
                    has_debt_context = any(word in user_lower for word in ["$", "debt", "balance", "owe", "monthly payment", "interest rate", "apr"])
                    
                    debt_match = has_debt_keyword and not has_exclusion and has_debt_context
                    logger.info(f"Debt keywords match: {debt_match} (keyword:{has_debt_keyword}, exclusion:{has_exclusion}, context:{has_debt_context})")
                    
                    if debt_match:
                        # Extract actual user data for contextual charts
                        import re
                        debt_amount = 5000  # default
                        interest_rate = 0.19  # default 19%
                        min_payment = 125  # default
                        
                        # Try to extract debt amount
                        amounts = re.findall(r'\$?([0-9,]+)', user_message)
                        if amounts:
                            try:
                                debt_amount = int(amounts[0].replace(',', ''))
                            except:
                                pass
                        
                        # Try to extract interest rate
                        rates = re.findall(r'(\d+(?:\.\d+)?)%', user_message)
                        if rates:
                            try:
                                interest_rate = float(rates[0]) / 100
                            except:
                                pass
                        
                        # Calculate realistic payments
                        min_payment = max(25, debt_amount * 0.025)  # 2.5% minimum or $25
                        accelerated_payment = min_payment * 2.5
                        
                        # Generate contextual chart data
                        months = list(range(1, min(37, int(debt_amount/min_payment) + 12)))
                        balances = []
                        current_balance = debt_amount
                        
                        for month in months:
                            interest = current_balance * (interest_rate / 12)
                            principal = accelerated_payment - interest
                            current_balance = max(0, current_balance - principal)
                            balances.append(current_balance)
                            if current_balance <= 0:
                                break
                        
                        chart_data = self.create_financial_chart("debt_payoff", {
                            "months": months[:len(balances)],
                            "balances": balances,
                            "interest_saved": [i * (interest_rate * debt_amount / 12) for i in range(len(balances))]
                        })
                    
                    # Card comparison scenarios - Only for explicit comparisons
                    elif ("compare" in user_lower and ("card" in user_lower or "credit" in user_lower)) or \
                         ("vs" in user_lower and "card" in user_lower) or \
                         ("versus" in user_lower and "card" in user_lower):
                        chart_data = self.create_financial_chart("card_comparison", {
                            "cards": ["Amex Cobalt", "RBC Avion", "TD Cashback", "Scotia Gold"],
                            "rewards": [5.0, 1.25, 3.0, 5.0],
                            "fees": [0, 120, 139, 139]
                        })
                    
                    # Credit score scenarios - Only for explicit credit score improvement queries
                    elif ("credit score" in user_lower and ("improve" in user_lower or "build" in user_lower or "increase" in user_lower)) or \
                         ("improve credit" in user_lower) or \
                         ("build credit" in user_lower and "score" in user_lower):
                        # Extract starting score if provided
                        import re
                        starting_score = 620  # default
                        scores_found = re.findall(r'(\d{3})', user_message)
                        if scores_found:
                            for score in scores_found:
                                score_int = int(score)
                                if 300 <= score_int <= 850:  # Valid credit score range
                                    starting_score = score_int
                                    break
                        
                        # Generate realistic improvement timeline
                        target_score = min(starting_score + 150, 780)
                        months = list(range(1, 25))
                        scores = []
                        for month in months:
                            # Realistic improvement curve - faster initially, then slows down
                            improvement = (target_score - starting_score) * (1 - (0.95 ** month))
                            score = min(target_score, starting_score + improvement)
                            scores.append(int(score))
                        
                        chart_data = self.create_financial_chart("credit_score", {
                            "months": months,
                            "scores": scores
                        })
                    
                    return generated_text, chart_data
                    
                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    return self.fallback_response(), None
                    
        except Exception as e:
            logger.error(f"Error calling OpenAI GPT-4: {e}")
            return self.fallback_response(), None
    
    def fallback_response(self) -> str:
        """Fallback response when GPT-4 is unavailable"""
        return """I'm experiencing technical difficulties with my advanced AI model right now, but I can still help! 

For Canadian credit card advice:
- **Travel rewards**: RBC Avion Visa Infinite (flexible points, $120 fee)
- **Cashback**: TD Cash Back Visa Infinite (3% groceries/gas, $139 fee)  
- **Dining/Groceries**: Amex Cobalt Card (5x rewards, no annual fee)
- **Students**: RBC Student Visa (no income requirement, builds credit)

Please ask me a specific question about debt payoff, card comparisons, or financial planning!"""
    
    def extract_follow_up_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from GPT-4 response"""
        questions = []
        lines = response.split('\n')
        for line in lines:
            if '?' in line and any(word in line.lower() for word in ['consider', 'might', 'what about', 'have you', 'would you', 'could you']):
                question = line.strip().strip('-').strip('*').strip()
                if len(question) > 10 and len(question) < 150:
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
        """Generate intelligent response using OpenAI GPT-4"""
        start_time = time.time()
        
        try:
            if self.loaded and self.openai_api_key:
                # Use GPT-4 for advanced response
                messages = self.create_financial_prompt(user_message, user_profile)
                response, chart_data = await self.generate_gpt4_response(messages)
                suggestions = self.extract_follow_up_questions(response)
            else:
                # Fallback response
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
navus_model = OpenAIGPT4NavusModel()

@app.on_event("startup")
async def startup_event():
    """Test model on startup"""
    logger.info("🚀 Starting NAVUS with OpenAI GPT-4")
    await navus_model.test_openai_connection()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NAVUS Credit Card Advisor API - OpenAI GPT-4 Enhanced", 
        "status": "active",
        "version": "5.0.0 (GPT-4 Turbo)",
        "model_type": navus_model.model_type,
        "model_loaded": navus_model.loaded,
        "features": ["OpenAI GPT-4", "Chart Generation", "Canadian Focus", "1,978 Training Examples"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if navus_model.loaded else "needs_api_key",
        model_loaded=navus_model.loaded,
        model_type=navus_model.model_type,
        memory_usage="Cloud-optimized via OpenAI"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with OpenAI GPT-4 and charts"""
    try:
        response, processing_time, suggestions, chart_data = await navus_model.generate_response(
            request.message, 
            request.user_profile
        )
        
        return ChatResponse(
            response=response,
            suggested_questions=suggestions,
            processing_time=processing_time,
            confidence=0.98 if navus_model.loaded else 0.6,
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
            "powered_by": "OpenAI GPT-4 Turbo"
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
        "api_key_configured": bool(navus_model.openai_api_key),
        "training_examples": len(navus_model.training_examples),
        "features": ["OpenAI GPT-4 Turbo", "Chart Generation", "Canadian Banking Data"],
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/auth/register", response_model=AuthResponse)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Validate password strength
        password_check = validate_password_strength(user_data.password)
        if not password_check["is_strong"]:
            return AuthResponse(
                success=False,
                message="Password does not meet strength requirements: " + ", ".join(password_check["issues"])
            )
        
        # Create user
        user = create_user(db, user_data.email, user_data.name, user_data.password)
        
        # Create session
        session_token = create_session(db, user.user_id, is_persistent=False)
        
        logger.info(f"New user registered: {user_data.email}")
        
        return AuthResponse(
            success=True,
            message="Account created successfully",
            user_id=user.user_id,
            name=user.name,
            session_token=session_token
        )
        
    except AuthError as e:
        return AuthResponse(success=False, message=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return AuthResponse(success=False, message="Registration failed. Please try again.")

@app.post("/auth/login", response_model=AuthResponse)
async def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    try:
        # Authenticate user
        user = authenticate_user(db, user_data.email, user_data.password)
        if not user:
            return AuthResponse(success=False, message="Invalid email or password")
        
        # Create session
        session_token = create_session(db, user.user_id, is_persistent=user_data.keep_logged_in)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        logger.info(f"User logged in: {user_data.email}")
        
        return AuthResponse(
            success=True,
            message="Login successful",
            user_id=user.user_id,
            name=user.name,
            session_token=session_token
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return AuthResponse(success=False, message="Login failed. Please try again.")

@app.post("/auth/logout")
async def logout_user(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Logout user"""
    try:
        if not session_token:
            return {"success": False, "message": "No active session"}
        
        success = logout_session(db, session_token)
        if success:
            logger.info("User logged out successfully")
            return {"success": True, "message": "Logged out successfully"}
        else:
            return {"success": False, "message": "Invalid session"}
            
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"success": False, "message": "Logout failed"}

@app.get("/auth/status")
async def auth_status(authorization: str = Header(None), session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Check authentication status"""
    try:
        # Check Authorization header first, then cookie
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
        elif session_token:
            token = session_token
            
        if not token:
            return {"authenticated": False, "user": None}
        
        user = get_user_by_session(db, token)
        if user:
            user_data = {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "profile_picture": user.profile_picture,
                "oauth_provider": user.oauth_provider
            }
            return {"authenticated": True, "user": user_data}
        else:
            return {"authenticated": False, "user": None}
            
    except Exception as e:
        logger.error(f"Auth status check error: {e}")
        return {"authenticated": False, "user": None}

@app.get("/auth/me")
async def get_current_user(session_token: str = Cookie(None), db: Session = Depends(get_db)):
    """Get current user information"""
    try:
        if not session_token:
            return {"authenticated": False, "message": "No active session"}
        
        user = get_user_by_session(db, session_token)
        if not user:
            return {"authenticated": False, "message": "Invalid or expired session"}
        
        return {
            "authenticated": True,
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        return {"authenticated": False, "message": "Authentication check failed"}

@app.post("/auth/validate-password", response_model=PasswordStrengthResponse)
async def validate_password(password: dict):
    """Validate password strength"""
    try:
        password_str = password.get("password", "")
        result = validate_password_strength(password_str)
        
        return PasswordStrengthResponse(
            score=result["score"],
            max_score=result["max_score"],
            is_strong=result["is_strong"],
            issues=result["issues"]
        )
        
    except Exception as e:
        logger.error(f"Password validation error: {e}")
        return PasswordStrengthResponse(
            score=0,
            max_score=5,
            is_strong=False,
            issues=["Password validation failed"]
        )

# Google OAuth endpoints
@app.get("/auth/google")
async def google_login():
    """Initiate Google OAuth login"""
    try:
        if not google_oauth.is_configured():
            return {"error": "Google OAuth is not configured"}
        
        authorization_url = google_oauth.get_authorization_url()
        if not authorization_url:
            return {"error": "Failed to generate Google authorization URL"}
        
        return {"authorization_url": authorization_url}
        
    except Exception as e:
        logger.error(f"Google OAuth initiation error: {e}")
        return {"error": "Failed to initiate Google login"}

@app.get("/auth/google/callback")
async def google_callback(code: str, db: Session = Depends(get_db)):
    """Handle Google OAuth callback"""
    try:
        if not google_oauth.is_configured():
            return {"error": "Google OAuth is not configured"}
        
        # Exchange authorization code for user info
        user_info = google_oauth.exchange_code_for_token(code)
        if not user_info:
            return {"error": "Failed to get user information from Google"}
        
        # Create or get user
        user = create_or_get_google_user(db, user_info)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create session
        session_token = create_session(db, user.user_id, is_persistent=True)  # Google login is persistent by default
        
        logger.info(f"Google OAuth login successful: {user.email}")
        
        # Redirect to frontend with success and set cookie
        frontend_url = "http://localhost:8081" if os.getenv('ENVIRONMENT') == 'development' else "https://navus.chat"
        response = RedirectResponse(url=f"{frontend_url}?auth=success&token={session_token}")
        
        # Set session token as cookie
        is_production = os.getenv('ENVIRONMENT') != 'development'
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=is_production,  # True for HTTPS production, False for localhost development
            samesite="none" if is_production else "lax",  # Use "none" for cross-origin in production
            max_age=86400 * 30  # 30 days
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        frontend_url = "http://localhost:8081" if os.getenv('ENVIRONMENT') == 'development' else "https://navus.chat"
        return RedirectResponse(url=f"{frontend_url}/login?error=oauth_failed")

@app.get("/auth/google/status")
async def google_oauth_status():
    """Check Google OAuth configuration status"""
    return {
        "configured": google_oauth.is_configured(),
        "redirect_uri": google_oauth.redirect_uri if google_oauth.is_configured() else None
    }

# Twitch OAuth endpoints
@app.get("/auth/twitch")
async def twitch_login():
    """Initiate Twitch OAuth login"""
    try:
        if not twitch_oauth.is_configured():
            return {"error": "Twitch OAuth is not configured"}
        
        # Generate secure state parameter
        state = twitch_oauth.generate_state()
        
        # Store state in session (for production, use Redis or database)
        # For now, we'll validate it in the callback
        
        authorization_url = twitch_oauth.get_authorization_url(state)
        if not authorization_url:
            return {"error": "Failed to generate Twitch authorization URL"}
        
        return {
            "authorization_url": authorization_url,
            "state": state  # Frontend should store this to validate callback
        }
        
    except Exception as e:
        logger.error(f"Twitch OAuth initiation error: {e}")
        return {"error": "Failed to initiate Twitch login"}

@app.get("/auth/twitch/callback")
async def twitch_callback(code: str, state: str, db: Session = Depends(get_db)):
    """Handle Twitch OAuth callback"""
    try:
        if not twitch_oauth.is_configured():
            return {"error": "Twitch OAuth is not configured"}
        
        # Exchange authorization code for access token
        access_token = await twitch_oauth.exchange_code_for_token(code)
        if not access_token:
            return {"error": "Failed to exchange code for access token"}
        
        # Get user information from Twitch
        user_info = await twitch_oauth.get_user_info(access_token)
        if not user_info:
            return {"error": "Failed to get user information from Twitch"}
        
        # Create or get user
        user = create_or_get_twitch_user(db, user_info)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create session
        session_token = create_session(db, user.user_id, is_persistent=True)  # Twitch login is persistent by default
        
        logger.info(f"Twitch OAuth login successful: {user_info.get('login')}")
        
        # Redirect to frontend with success and set cookie
        frontend_url = "http://localhost:8081" if os.getenv('ENVIRONMENT') == 'development' else "https://navus.chat"
        response = RedirectResponse(url=f"{frontend_url}?auth=success&token={session_token}")
        
        # Set session token as cookie
        is_production = os.getenv('ENVIRONMENT') != 'development'
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=is_production,  # True for HTTPS production, False for localhost development
            samesite="none" if is_production else "lax",  # Use "none" for cross-origin in production
            max_age=86400 * 30  # 30 days
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Twitch OAuth callback error: {e}")
        frontend_url = "http://localhost:8081" if os.getenv('ENVIRONMENT') == 'development' else "https://navus.chat"
        return RedirectResponse(url=f"{frontend_url}/login?error=oauth_failed")

@app.get("/auth/twitch/status")
async def twitch_oauth_status():
    """Check Twitch OAuth configuration status"""
    return {
        "configured": twitch_oauth.is_configured(),
        "redirect_uri": twitch_oauth.redirect_uri if twitch_oauth.is_configured() else None,
        "client_id": twitch_oauth.client_id if twitch_oauth.is_configured() else None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Initialize database tables
    print("🗄️ Initializing database tables...")
    try:
        create_tables()
        print("✅ Database tables initialized successfully")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        print("🔄 The application will still start, but authentication may not work without a database connection")
    
    port = int(os.environ.get("PORT", 8003))  # Different port
    
    print("🚀 Starting NAVUS Credit Card Advisor API (OpenAI GPT-4 Enhanced)...")
    print(f"📱 API will be available on port: {port}")
    print("🤖 Using OpenAI GPT-4 Turbo for advanced financial advice")
    print("📊 Enhanced chart generation enabled")
    print("🔐 Authentication system enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=port)