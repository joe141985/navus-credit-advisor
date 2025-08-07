#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - DEMO VERSION
Lightweight version with mock LLM for fast deployment and investor demos
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
import os
import json
import random
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API - DEMO", 
    version="2.0.0",
    description="AI-powered Canadian credit card advisor (Demo Version)"
)

# Production CORS - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.vercel.app",
        "https://navus-credit-advisor.vercel.app",
        "http://localhost:3000",
        "https://localhost:3000"
    ],
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
    suggestions: List[str]
    processing_time: float

# Global variables for dataset
cards_df = None

def load_credit_cards():
    """Load credit card dataset"""
    global cards_df
    try:
        # Try different possible paths
        possible_paths = [
            "../Data/master_card_dataset_cleaned.csv",
            "Data/master_card_dataset_cleaned.csv",
            "/opt/render/project/src/Data/master_card_dataset_cleaned.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cards_df = pd.read_csv(path)
                logger.info(f"✅ Loaded {len(cards_df)} credit cards from {path}")
                return cards_df
                
        # If no file found, create mock data
        cards_df = create_mock_dataset()
        logger.info("⚠️ Using mock credit card dataset")
        return cards_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        cards_df = create_mock_dataset()
        return cards_df

def create_mock_dataset():
    """Create mock credit card dataset for demo"""
    mock_cards = [
        {"Card_Name": "RBC Avion Visa Infinite", "Bank": "RBC", "Annual_Fee": 139, "Reward_Program": "RBC Avion", "Key_Benefits": "Travel Rewards | Insurance"},
        {"Card_Name": "TD Aeroplan Visa Infinite", "Bank": "TD", "Annual_Fee": 139, "Reward_Program": "Aeroplan", "Key_Benefits": "Air Canada Benefits | Lounge Access"},
        {"Card_Name": "BMO CashBack Mastercard", "Bank": "BMO", "Annual_Fee": 0, "Reward_Program": "Cash Back", "Key_Benefits": "No Annual Fee | 3% Groceries"},
        {"Card_Name": "CIBC Aventura Visa Infinite", "Bank": "CIBC", "Annual_Fee": 139, "Reward_Program": "Aventura Points", "Key_Benefits": "Flexible Travel Rewards | Travel Insurance"},
        {"Card_Name": "Scotia Momentum Visa Infinite", "Bank": "Scotiabank", "Annual_Fee": 99, "Reward_Program": "Cash Back", "Key_Benefits": "High Cash Back on Gas and Groceries"}
    ]
    return pd.DataFrame(mock_cards)

def get_mock_llm_response(message: str, user_profile: Dict = None, cards_data: pd.DataFrame = None) -> Dict:
    """Generate intelligent mock LLM response based on user message and credit card data"""
    
    message_lower = message.lower()
    
    # Analyze user intent
    if any(word in message_lower for word in ['travel', 'vacation', 'trip', 'miles', 'points']):
        category = "travel"
        relevant_cards = [
            "**RBC Avion Visa Infinite** (RBC): $139 annual fee, RBC Avion points. Travel Rewards | Insurance",
            "**TD Aeroplan Visa Infinite** (TD): $139 annual fee, Aeroplan. Air Canada Benefits | Lounge Access",
            "**CIBC Aventura Visa Infinite** (CIBC): $139 annual fee, Aventura Points. Flexible Travel Rewards | Travel Insurance"
        ]
        suggestions = [
            "Which travel card has no foreign transaction fees?",
            "Best card for airport lounge access?",
            "Compare RBC Avion vs TD Aeroplan cards"
        ]
        
    elif any(word in message_lower for word in ['cashback', 'cash back', 'money back', 'groceries', 'gas']):
        category = "cashback"
        relevant_cards = [
            "**BMO CashBack Mastercard** (BMO): $0 annual fee, Cash Back. No Annual Fee | 3% on Groceries",
            "**Scotia Momentum Visa Infinite** (Scotiabank): $99 annual fee, Cash Back. High Cash Back on Gas and Groceries",
            "**Tangerine Money-Back Credit Card** (Tangerine): $0 annual fee, Cash Back. 2% on 2 Categories"
        ]
        suggestions = [
            "Which card gives highest cashback on groceries?",
            "Best no-fee cashback card?",
            "How do I maximize cashback rewards?"
        ]
        
    elif any(word in message_lower for word in ['student', 'first card', 'credit building', 'no credit']):
        category = "student"
        relevant_cards = [
            "**RBC Student Visa** (RBC): $0 annual fee, RBC Rewards. No Annual Fee | Student Benefits",
            "**TD Student Visa** (TD): $0 annual fee, Cash Back. No Annual Fee | Student Discounts",
            "**BMO SPC CashBack Mastercard** (BMO): $0 annual fee, Cash Back. Student Discounts | No Annual Fee"
        ]
        suggestions = [
            "Best student cards with no income requirement?",
            "How to build credit as a student?",
            "Student cards that graduate to regular cards?"
        ]
        
    else:
        category = "general"
        relevant_cards = [
            "**RBC Avion Visa Infinite** (RBC): $139 annual fee, RBC Avion points. Travel Rewards | Insurance",
            "**BMO CashBack Mastercard** (BMO): $0 annual fee, Cash Back. No Annual Fee | 3% on Groceries",
            "**TD Aeroplan Visa Infinite** (TD): $139 annual fee, Aeroplan. Air Canada Benefits | Lounge Access"
        ]
        suggestions = [
            "What's the best card for my spending habits?",
            "Should I pay an annual fee for better rewards?",
            "How do I choose between points and cashback?"
        ]
    
    # Build personalized response
    response_parts = []
    
    # Add persona-specific intro
    if user_profile and user_profile.get('persona'):
        persona = user_profile['persona']
        if persona == 'frequent_traveler':
            response_parts.append("As a frequent traveler, I'd recommend focusing on travel rewards cards with comprehensive benefits.")
        elif persona == 'student':
            response_parts.append("Perfect for students building credit history:")
        elif persona == 'cashback_focused':
            response_parts.append("For maximizing cashback rewards in Canada:")
        else:
            response_parts.append("Based on your needs, here are the best Canadian options:")
    else:
        response_parts.append(f"For {category} rewards, here are the top Canadian options:")
    
    # Add card recommendations
    response_parts.append("")
    for card in relevant_cards[:3]:  # Limit to top 3
        response_parts.append(f"• {card}")
    
    # Add personalized advice
    if category == "travel":
        response_parts.append("\nBoth offer excellent travel benefits, but RBC Avion provides more redemption flexibility while TD Aeroplan is best for Air Canada flyers.")
    elif category == "cashback":
        response_parts.append("\nThe key is matching your spending patterns. High spenders benefit from premium cashback cards despite annual fees.")
    elif category == "student":
        response_parts.append("\nBoth are designed for students with no credit history. Start with one, use responsibly, and upgrade later as your income grows.")
    else:
        response_parts.append("\nConsider your spending patterns and whether you prefer cashback or travel rewards.")
    
    # Add location note
    if user_profile and user_profile.get('location'):
        location = user_profile['location']
        response_parts.append(f"\nAll recommendations are available in {location} and across Canada.")
    
    return {
        "response": "\n".join(response_parts),
        "suggestions": suggestions,
        "processing_time": round(random.uniform(0.001, 0.008), 3)  # Mock realistic processing time
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting NAVUS Credit Card Advisor API (Demo Version)")
    load_credit_cards()
    logger.info("✅ NAVUS API ready for investor demonstrations!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NAVUS Credit Card Advisor API",
        "version": "2.0.0-demo",
        "cards_loaded": len(cards_df) if cards_df is not None else 0,
        "message": "🤖 NAVUS is ready to help Canadians find their perfect credit card!"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for credit card advice"""
    try:
        start_time = datetime.now()
        
        # Get mock LLM response
        llm_result = get_mock_llm_response(
            request.message,
            request.user_profile,
            cards_df
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"💬 Chat request processed in {processing_time:.3f}s")
        
        return ChatResponse(
            response=llm_result["response"],
            suggestions=llm_result["suggestions"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.get("/cards")
async def get_cards():
    """Get available credit cards"""
    try:
        if cards_df is None:
            load_credit_cards()
        
        return {
            "cards": cards_df.to_dict('records'),
            "total_cards": len(cards_df),
            "message": "Available Canadian credit cards"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cards: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "NAVUS Credit Card Advisor",
        "version": "2.0.0-demo",
        "cards_loaded": len(cards_df) if cards_df is not None else 0,
        "features": [
            "✅ Credit card recommendations",
            "✅ Personalized advice",
            "✅ Canadian market focus",
            "✅ Fast response times",
            "✅ Investor demo ready"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)