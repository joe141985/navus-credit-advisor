#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - LLAMA ENHANCED
Uses local Llama 3.1 8B for advanced financial advice
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API - Llama Enhanced", 
    version="3.0.0",
    description="AI-powered Canadian credit card advisor with Llama 3.1"
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

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    memory_usage: Optional[str] = None

class LlamaNavusModel:
    def __init__(self):
        self.loaded = False
        self.model_type = "llama3.1:8b"
        self.dataset_df = None
        self.ollama_url = "http://localhost:11434"
        
        # Load credit card dataset
        self.load_dataset()
    
    def load_dataset(self):
        """Load credit card dataset from embedded data"""
        sample_cards = [
            {
                "name": "American Express Cobalt Card", 
                "issuer": "American Express", 
                "category": "rewards", 
                "annual_fee": 0, 
                "rewards_rate": "5x on groceries/food",
                "best_for": "dining and groceries"
            },
            {
                "name": "RBC Avion Visa Infinite", 
                "issuer": "RBC", 
                "category": "travel", 
                "annual_fee": 120, 
                "rewards_rate": "1.25x on purchases",
                "best_for": "flexible travel rewards"
            },
            {
                "name": "TD Cash Back Visa Infinite", 
                "issuer": "TD", 
                "category": "cashback", 
                "annual_fee": 139, 
                "rewards_rate": "3% on groceries/gas",
                "best_for": "high cashback rates"
            },
            {
                "name": "RBC Student Visa", 
                "issuer": "RBC", 
                "category": "student", 
                "annual_fee": 0, 
                "rewards_rate": "1x on purchases",
                "best_for": "building credit history"
            },
            {
                "name": "Capital One Secured Mastercard", 
                "issuer": "Capital One", 
                "category": "secured", 
                "annual_fee": 59, 
                "rewards_rate": "None",
                "best_for": "rebuilding credit"
            },
            {
                "name": "Scotiabank Gold American Express", 
                "issuer": "Scotiabank", 
                "category": "rewards", 
                "annual_fee": 139, 
                "rewards_rate": "5x on groceries/gas",
                "best_for": "high rewards on essentials"
            },
            {
                "name": "BMO Air Miles World Elite", 
                "issuer": "BMO", 
                "category": "travel", 
                "annual_fee": 120, 
                "rewards_rate": "1x Air Miles per $15",
                "best_for": "Air Miles collectors"
            },
            {
                "name": "CIBC Dividend Visa Infinite", 
                "issuer": "CIBC", 
                "category": "cashback", 
                "annual_fee": 99, 
                "rewards_rate": "4% on gas/groceries",
                "best_for": "everyday cashback"
            }
        ]
        
        self.dataset_df = pd.DataFrame(sample_cards)
        logger.info(f"✅ Loaded {len(self.dataset_df)} Canadian credit cards")
    
    async def test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    models = response.json()
                    model_names = [model['name'] for model in models.get('models', [])]
                    if 'llama3.1:8b' in model_names:
                        self.loaded = True
                        logger.info("✅ Llama 3.1 8B model is ready!")
                    else:
                        logger.warning("⚠️  Llama 3.1 8B not found. Available models: " + str(model_names))
                else:
                    logger.warning("⚠️  Ollama service not responding")
        except Exception as e:
            logger.warning(f"⚠️  Could not connect to Ollama: {e}")
    
    def create_financial_prompt(self, user_message: str, user_profile: Dict = None) -> str:
        """Create a specialized prompt for Canadian financial advice"""
        
        # Get relevant card data as context
        card_context = ""
        for _, card in self.dataset_df.iterrows():
            card_context += f"- {card['name']} ({card['issuer']}): ${card['annual_fee']} fee, {card['rewards_rate']}, best for {card['best_for']}\n"
        
        profile_context = ""
        if user_profile:
            if user_profile.get('income'):
                profile_context += f"User's approximate income: ${user_profile['income']}\n"
            if user_profile.get('location'):
                profile_context += f"User's location: {user_profile['location']}\n"
        
        prompt = f"""You are NAVUS, an expert Canadian financial advisor specializing in credit cards and debt management. You have deep knowledge of Canadian banking, credit bureaus (Equifax and TransUnion Canada), and financial regulations.

CANADIAN CREDIT CARDS DATABASE:
{card_context}

{profile_context}

USER QUESTION: {user_message}

INSTRUCTIONS:
1. Provide specific, actionable advice for Canadians
2. Reference actual Canadian credit cards from the database above
3. Consider Canadian banking regulations and practices
4. If discussing debt payoff, mention both avalanche and snowball methods
5. Always mention specific card names, issuers, and key benefits
6. Be conversational but professional
7. If asked about complex scenarios, provide step-by-step guidance
8. End with 1-2 relevant follow-up question suggestions

RESPONSE (be helpful, specific, and Canadian-focused):"""
        
        return prompt
    
    async def generate_llama_response(self, prompt: str) -> str:
        """Generate response using local Llama 3.1"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 500,
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'Sorry, I could not generate a response.')
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return self.fallback_response()
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return self.fallback_response()
    
    def fallback_response(self) -> str:
        """Fallback response when Llama is unavailable"""
        return """I apologize, but I'm having trouble accessing my advanced AI model right now. 
        
For immediate help, I can tell you that some top Canadian credit cards include:
- American Express Cobalt (excellent for groceries/dining)
- RBC Avion Visa Infinite (flexible travel rewards)  
- TD Cash Back Visa Infinite (high cashback rates)

Please try again in a moment, or ask me a specific question about Canadian credit cards!"""
    
    def extract_follow_up_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from Llama response"""
        # Simple extraction logic - can be enhanced
        questions = []
        lines = response.split('\n')
        for line in lines:
            if '?' in line and ('ask' in line.lower() or 'consider' in line.lower() or 'might' in line.lower()):
                question = line.strip().strip('-').strip()
                if len(question) > 10 and len(question) < 100:
                    questions.append(question)
        
        # Default suggestions if none found
        if not questions:
            questions = [
                "What's the best credit card for my spending habits?",
                "How can I improve my credit score in Canada?",
                "Should I pay off debt or invest extra money?"
            ]
        
        return questions[:2]  # Return up to 2 suggestions
    
    async def generate_response(self, user_message: str, user_profile: Dict = None) -> tuple:
        """Generate intelligent response using Llama 3.1"""
        start_time = time.time()
        
        try:
            if self.loaded:
                # Use Llama 3.1 for advanced response
                prompt = self.create_financial_prompt(user_message, user_profile)
                response = await self.generate_llama_response(prompt)
                suggestions = self.extract_follow_up_questions(response)
            else:
                # Fallback to basic response while model loads
                response = self.fallback_response()
                suggestions = [
                    "What's the best travel credit card in Canada?",
                    "Help me compare cashback vs rewards cards"
                ]
            
            processing_time = time.time() - start_time
            return response, processing_time, suggestions
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time
            return self.fallback_response(), processing_time, []

# Initialize model
navus_model = LlamaNavusModel()

@app.on_event("startup")
async def startup_event():
    """Test model on startup"""
    logger.info("🚀 Starting NAVUS with Llama 3.1 8B")
    await navus_model.test_ollama_connection()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NAVUS Credit Card Advisor API - Llama Enhanced", 
        "status": "active",
        "version": "3.0.0 (Llama 3.1)",
        "model_type": navus_model.model_type,
        "model_loaded": navus_model.loaded
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if navus_model.loaded else "loading",
        model_loaded=navus_model.loaded,
        model_type=navus_model.model_type,
        memory_usage="Optimized for Apple Silicon"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with Llama 3.1"""
    try:
        response, processing_time, suggestions = await navus_model.generate_response(
            request.message, 
            request.user_profile
        )
        
        return ChatResponse(
            response=response,
            suggested_questions=suggestions,
            processing_time=processing_time,
            confidence=0.85 if navus_model.loaded else 0.5
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cards")
async def get_featured_cards():
    """Get featured credit cards"""
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
                    "best_for": card['best_for']
                })
        
        return {
            "featured_cards": featured_cards,
            "total_cards_in_database": len(featured_cards),
            "powered_by": "Llama 3.1 8B"
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
        "ollama_url": navus_model.ollama_url,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))  # Different port to avoid conflicts
    
    print("🚀 Starting NAVUS Credit Card Advisor API (Llama Enhanced)...")
    print(f"📱 API will be available on port: {port}")
    print("🤖 Using Llama 3.1 8B for advanced financial advice")
    
    uvicorn.run(app, host="0.0.0.0", port=port)