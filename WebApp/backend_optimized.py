#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - OPTIMIZED
FastAPI backend with fine-tuned model loading and enhanced features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import os
import json
import asyncio
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API", 
    version="2.0.0",
    description="AI-powered Canadian credit card advisor with fine-tuned recommendations"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    user_profile: Optional[Dict] = {}  # For persona-based responses

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    suggested_questions: Optional[List[str]] = []
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    cuda_available: bool
    gpu_memory: Optional[str] = None

class NAVUSModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.model_type = "base"  # "base" or "finetuned"
        self.dataset_df = None
        
        # Load credit card dataset for enhanced responses
        self.load_dataset()
    
    def load_dataset(self):
        """Load credit card dataset for reference"""
        try:
            csv_path = "/Users/joebanerjee/NAVUS/Data/master_card_dataset_cleaned.csv"
            if os.path.exists(csv_path):
                self.dataset_df = pd.read_csv(csv_path)
                logger.info(f"✅ Loaded {len(self.dataset_df)} credit cards for reference")
            else:
                logger.warning("⚠️ Credit card dataset not found")
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
    
    async def load_model(self, model_path: str = "./navus_mistral_finetuned"):
        """Load the fine-tuned model with fallback to base model"""
        try:
            logger.info("🔄 Loading NAVUS model...")
            
            # Try to load fine-tuned model first
            if os.path.exists(model_path) and os.path.exists(f"{model_path}/adapter_config.json"):
                logger.info("📁 Fine-tuned model found, loading...")
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True
                )
                
                # Load tokenizer from fine-tuned model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side="right"
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load LoRA weights
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.model_type = "finetuned"
                
                logger.info("✅ Fine-tuned NAVUS model loaded successfully!")
                
            else:
                raise FileNotFoundError("Fine-tuned model not available")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load fine-tuned model: {e}")
            logger.info("🔄 Loading base model as fallback...")
            
            # Fallback to base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True
            )
            self.model_type = "base"
            
            logger.info("✅ Base model loaded as fallback")
        
        self.loaded = True
        
        # Log model info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    def get_suggested_questions(self, user_message: str) -> List[str]:
        """Generate relevant follow-up questions"""
        suggestions = []
        message_lower = user_message.lower()
        
        # Travel-related suggestions
        if any(word in message_lower for word in ['travel', 'trip', 'vacation', 'airline', 'hotel']):
            suggestions.extend([
                "Which travel card has no foreign transaction fees?",
                "Best card for airport lounge access?",
                "Compare TD Aeroplan vs RBC Avion cards"
            ])
        
        # Cashback-related suggestions
        elif any(word in message_lower for word in ['cashback', 'cash back', 'groceries', 'gas', 'spending']):
            suggestions.extend([
                "Which card gives highest cashback on groceries?",
                "Best no-fee cashback card?",
                "How do I maximize cashback rewards?"
            ])
        
        # Student-related suggestions
        elif any(word in message_lower for word in ['student', 'first card', 'young', 'college']):
            suggestions.extend([
                "Best student cards with no income requirement?",
                "How to build credit as a student?",
                "Student cards that graduate to regular cards?"
            ])
        
        # General suggestions
        else:
            suggestions.extend([
                "What's the best card for my spending habits?",
                "Should I pay an annual fee for better rewards?",
                "How do I choose between points and cashback?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def enhance_response_with_data(self, response: str, user_message: str) -> str:
        """Enhance response with specific data from our dataset"""
        if self.dataset_df is None:
            return response
        
        try:
            message_lower = user_message.lower()
            
            # If asking about specific cards, add current data
            for _, card in self.dataset_df.iterrows():
                card_name = str(card['name']).lower()
                if card_name in message_lower and len(card_name) > 5:  # Avoid short matches
                    annual_fee = f"${card['annual_fee']}" if pd.notna(card['annual_fee']) else "Not specified"
                    rewards = card.get('rewards_type', 'Not specified')
                    
                    enhancement = f"\n\n📊 Current Data: {card['name']} has an annual fee of {annual_fee}"
                    if pd.notna(rewards) and rewards != 'Not specified':
                        enhancement += f" and offers {rewards} rewards."
                    
                    response += enhancement
                    break
            
            return response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response
    
    def generate_response(self, user_message: str, user_profile: Dict = None, max_length: int = 400) -> tuple:
        """Generate response using the model with persona adaptation"""
        if not self.loaded:
            return "Sorry, the model is still loading. Please try again in a moment.", 0.0, []
        
        import time
        start_time = time.time()
        
        try:
            # Adapt prompt based on user profile
            adapted_message = user_message
            if user_profile:
                context_parts = []
                if user_profile.get('persona'):
                    context_parts.append(f"User type: {user_profile['persona']}")
                if user_profile.get('income'):
                    context_parts.append(f"Annual income: ${user_profile['income']}")
                if user_profile.get('location'):
                    context_parts.append(f"Location: {user_profile['location']}")
                
                if context_parts:
                    context = ". ".join(context_parts)
                    adapted_message = f"Context: {context}. Question: {user_message}"
            
            # Format for Mistral
            prompt = f"[INST] {adapted_message} [/INST]"
            
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
                response = "I apologize, but I couldn't process that request properly. Could you try rephrasing your question about Canadian credit cards?"
            
            # Enhance with dataset information
            response = self.enhance_response_with_data(response, user_message)
            
            # Get suggested questions
            suggestions = self.get_suggested_questions(user_message)
            
            processing_time = time.time() - start_time
            
            return response, processing_time, suggestions
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time
            return "I apologize, but I encountered an error. Please try asking your credit card question again.", processing_time, []

# Initialize model
navus_model = NAVUSModel()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await navus_model.load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NAVUS Credit Card Advisor API", 
        "status": "active",
        "version": "2.0.0",
        "model_type": navus_model.model_type
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return HealthResponse(
        status="healthy" if navus_model.loaded else "loading",
        model_loaded=navus_model.loaded,
        model_type=navus_model.model_type,
        cuda_available=torch.cuda.is_available(),
        gpu_memory=gpu_memory
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with persona support"""
    try:
        # Generate response
        response, processing_time, suggestions = navus_model.generate_response(
            request.message, 
            request.user_profile
        )
        
        return ChatResponse(
            response=response,
            suggested_questions=suggestions,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cards")
async def get_featured_cards():
    """Get featured credit cards by category"""
    try:
        if navus_model.dataset_df is None:
            return {"featured_cards": [], "error": "Dataset not available"}
        
        # Get featured cards from different categories
        featured_cards = []
        categories = ['travel', 'cashback', 'student', 'premium', 'no_fee']
        
        for category in categories:
            category_cards = navus_model.dataset_df[
                navus_model.dataset_df['category'] == category
            ].head(1)
            
            for _, card in category_cards.iterrows():
                featured_cards.append({
                    "name": card['name'],
                    "issuer": card['issuer'],
                    "category": card['category'],
                    "annual_fee": float(card['annual_fee']) if pd.notna(card['annual_fee']) else 0,
                    "rewards_type": card.get('rewards_type', 'Not specified'),
                    "features": card.get('features', '').split(' | ')[:3] if pd.notna(card.get('features')) else []
                })
        
        return {
            "featured_cards": featured_cards,
            "total_cards_in_database": len(navus_model.dataset_df)
        }
        
    except Exception as e:
        logger.error(f"Error getting featured cards: {e}")
        return {"featured_cards": [], "error": str(e)}

@app.get("/preset-questions")
async def get_preset_questions():
    """Get preset questions by category"""
    preset_questions = {
        "general": [
            "What's the best credit card for my spending habits?",
            "Should I pay an annual fee for better rewards?",
            "How do I choose between points and cashback?"
        ],
        "travel": [
            "Best travel rewards card in Canada?",
            "Which card has no foreign transaction fees?",
            "Cards with airport lounge access?"
        ],
        "cashback": [
            "Highest cashback rate on groceries?",
            "Best no-fee cashback card?",
            "Cards with rotating cashback categories?"
        ],
        "student": [
            "Best first credit card for students?",
            "Student cards with no income requirement?",
            "How to build credit as a student?"
        ],
        "premium": [
            "Best premium travel card benefits?",
            "Cards with concierge services?",
            "Premium cards worth the annual fee?"
        ]
    }
    
    return {"preset_questions": preset_questions}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting NAVUS Credit Card Advisor API v2.0...")
    print("📱 API will be available at: http://localhost:8000")
    print("📋 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)