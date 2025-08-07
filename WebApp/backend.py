#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API
FastAPI backend for the credit card chat advisor
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NAVUS Credit Card Advisor API", version="1.0.0")

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

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None

class NAVUSModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    async def load_model(self, model_path: str = "./navus_mistral_finetuned"):
        """Load the fine-tuned model"""
        try:
            logger.info("Loading NAVUS model...")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.loaded = True
            
            logger.info("✅ NAVUS model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Fallback to base model
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            logger.info("⚠️  Using base model as fallback")
    
    def generate_response(self, user_message: str, max_length: int = 400) -> str:
        """Generate response using the model"""
        if not self.loaded:
            return "Sorry, the model is still loading. Please try again in a moment."
        
        try:
            # Format prompt for Mistral
            prompt = f"[INST] {user_message} [/INST]"
            
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
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "[/INST]" in full_response:
                response = full_response.split("[/INST]")[-1].strip()
            else:
                response = "I'm sorry, I couldn't process that request. Could you try rephrasing your question about Canadian credit cards?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try asking your credit card question again."

# Initialize model
navus_model = NAVUSModel()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await navus_model.load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "NAVUS Credit Card Advisor API", "status": "active"}

@app.get("/health")
async def health():
    """Health check with model status"""
    return {
        "status": "healthy",
        "model_loaded": navus_model.loaded,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Generate response
        response = navus_model.generate_response(request.message)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/cards")
async def get_sample_cards():
    """Get sample credit card recommendations"""
    import pandas as pd
    
    try:
        # Load the dataset to provide sample cards
        df = pd.read_csv("/Users/joebanerjee/NAVUS/Data/master_card_dataset_cleaned.csv")
        
        # Get top 5 cards from different categories
        sample_cards = []
        for category in ['travel', 'cashback', 'student', 'premium', 'no_fee']:
            category_cards = df[df['category'] == category].head(1)
            for _, card in category_cards.iterrows():
                sample_cards.append({
                    "name": card['name'],
                    "issuer": card['issuer'],
                    "category": card['category'],
                    "annual_fee": float(card['annual_fee']) if pd.notna(card['annual_fee']) else 0,
                    "rewards_type": card.get('rewards_type', 'Not specified')
                })
        
        return {"sample_cards": sample_cards[:5]}
        
    except Exception as e:
        return {"sample_cards": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting NAVUS Credit Card Advisor API...")
    print("📱 API will be available at: http://localhost:8000")
    print("📋 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
