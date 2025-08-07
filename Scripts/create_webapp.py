#!/usr/bin/env python3
"""
NAVUS Web App Creator
Creates a FastAPI + React web app for the credit card advisor
"""

import os

def create_fastapi_backend():
    """Create FastAPI backend server"""
    
    backend_code = '''#!/usr/bin/env python3
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
'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/backend.py', 'w') as f:
        f.write(backend_code)
    
    print("✅ Created backend.py")

def create_frontend():
    """Create React frontend"""
    
    # Create package.json
    package_json = '''{
  "name": "navus-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.0",
    "lucide-react": "^0.263.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/package.json', 'w') as f:
        f.write(package_json)
    
    # Create main App.js
    app_js = '''import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, CreditCard, MapPin, User, DollarSign } from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: "Hi! I'm NAVUS, your Canadian credit card advisor. I can help you find the perfect credit card based on your needs, income, and spending habits. What would you like to know?"
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sampleCards, setSampleCards] = useState([]);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Load sample cards on component mount
  useEffect(() => {
    const loadSampleCards = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/cards`);
        setSampleCards(response.data.sample_cards || []);
      } catch (error) {
        console.error('Error loading sample cards:', error);
      }
    };
    
    loadSampleCards();
  }, []);

  const sendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: userMessage,
        conversation_history: []
      });

      // Add assistant response
      setMessages(prev => [
        ...prev, 
        { type: 'assistant', content: response.data.response }
      ]);

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev, 
        { 
          type: 'assistant', 
          content: "I'm sorry, I encountered an error. Please make sure the backend server is running and try again." 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleQuestion = (question) => {
    setInputMessage(question);
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <CreditCard className="logo-icon" />
          <h1>NAVUS</h1>
          <p>Your Canadian Credit Card Advisor</p>
        </div>
      </header>

      <div className="main-container">
        {/* Sample Questions Sidebar */}
        <div className="sidebar">
          <h3>💡 Try asking:</h3>
          <div className="sample-questions">
            {[
              "What's the best no-fee travel card?",
              "I'm a student looking for my first credit card",
              "Best cashback card for groceries?", 
              "Compare premium travel cards",
              "Credit cards for building credit history"
            ].map((question, index) => (
              <button
                key={index}
                className="sample-question"
                onClick={() => handleSampleQuestion(question)}
                disabled={isLoading}
              >
                {question}
              </button>
            ))}
          </div>

          {/* Sample Cards */}
          {sampleCards.length > 0 && (
            <div className="sample-cards">
              <h3>🏆 Featured Cards:</h3>
              {sampleCards.map((card, index) => (
                <div key={index} className="sample-card">
                  <div className="card-name">{card.name}</div>
                  <div className="card-details">
                    <span className="card-issuer">{card.issuer}</span>
                    <span className="card-fee">${card.annual_fee}/year</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Interface */}
        <div className="chat-container">
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? <User size={20} /> : <CreditCard size={20} />}
                </div>
                <div className="message-content">
                  {message.content}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message assistant">
                <div className="message-avatar">
                  <CreditCard size={20} />
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={sendMessage} className="input-form">
            <div className="input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask me about Canadian credit cards..."
                disabled={isLoading}
                className="message-input"
              />
              <button
                type="submit"
                disabled={isLoading || !inputMessage.trim()}
                className="send-button"
              >
                <Send size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/src/App.js', 'w') as f:
        f.write(app_js)
    
    # Create CSS
    app_css = '''* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Header */
.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  padding: 1rem;
  text-align: center;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}

.logo-icon {
  color: #667eea;
}

.header-content h1 {
  font-size: 2rem;
  font-weight: 700;
  color: #2d3748;
}

.header-content p {
  color: #4a5568;
  font-size: 1rem;
}

/* Main Container */
.main-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Sidebar */
.sidebar {
  width: 300px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  padding: 1.5rem;
  overflow-y: auto;
  border-right: 1px solid rgba(255, 255, 255, 0.2);
}

.sidebar h3 {
  margin-bottom: 1rem;
  color: #2d3748;
  font-size: 1.1rem;
}

.sample-questions {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 2rem;
}

.sample-question {
  background: rgba(102, 126, 234, 0.1);
  border: 1px solid rgba(102, 126, 234, 0.2);
  padding: 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  font-size: 0.9rem;
  color: #4a5568;
}

.sample-question:hover:not(:disabled) {
  background: rgba(102, 126, 234, 0.2);
  transform: translateY(-1px);
}

.sample-question:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Sample Cards */
.sample-cards {
  margin-top: 1rem;
}

.sample-card {
  background: rgba(255, 255, 255, 0.7);
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.card-name {
  font-weight: 600;
  font-size: 0.85rem;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.card-details {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #4a5568;
}

/* Chat Container */
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
}

/* Messages */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  gap: 1rem;
  max-width: 80%;
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message.user .message-avatar {
  background: #667eea;
  color: white;
}

.message.assistant .message-avatar {
  background: #e2e8f0;
  color: #4a5568;
}

.message-content {
  background: #f7fafc;
  padding: 1rem;
  border-radius: 1rem;
  border: 1px solid #e2e8f0;
  line-height: 1.5;
}

.message.user .message-content {
  background: #667eea;
  color: white;
  border-color: #667eea;
}

/* Typing Indicator */
.typing-indicator {
  display: flex;
  gap: 0.25rem;
  align-items: center;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #cbd5e0;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: scale(0.75);
    opacity: 0.5;
  }
  30% {
    transform: scale(1);
    opacity: 1;
  }
}

/* Input Form */
.input-form {
  padding: 1.5rem;
  border-top: 1px solid #e2e8f0;
  background: rgba(255, 255, 255, 0.9);
}

.input-container {
  display: flex;
  gap: 0.75rem;
  align-items: flex-end;
}

.message-input {
  flex: 1;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 1rem;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
  background: rgba(255, 255, 255, 0.9);
}

.message-input:focus {
  border-color: #667eea;
}

.send-button {
  width: 48px;
  height: 48px;
  border: none;
  border-radius: 50%;
  background: #667eea;
  color: white;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.send-button:hover:not(:disabled) {
  background: #5a67d8;
  transform: scale(1.05);
}

.send-button:disabled {
  background: #cbd5e0;
  cursor: not-allowed;
  transform: none;
}

/* Responsive */
@media (max-width: 768px) {
  .main-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    height: 200px;
    border-right: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .message {
    max-width: 90%;
  }
}'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/src/App.css', 'w') as f:
        f.write(app_css)
    
    # Create index.js
    index_js = '''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/src/index.js', 'w') as f:
        f.write(index_js)
    
    # Create index.css
    index_css = '''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/src/index.css', 'w') as f:
        f.write(index_css)
    
    # Create public/index.html
    index_html = '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="NAVUS - Your Canadian Credit Card Advisor"
    />
    <title>NAVUS - Credit Card Advisor</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/frontend/public/index.html', 'w') as f:
        f.write(index_html)
    
    print("✅ Created React frontend")

def create_deployment_scripts():
    """Create deployment and running scripts"""
    
    # Backend requirements
    backend_requirements = '''fastapi==0.104.1
uvicorn==0.24.0
torch>=2.1.0
transformers>=4.36.0
peft>=0.6.0
pandas>=2.0.0
python-multipart==0.0.6
'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/requirements.txt', 'w') as f:
        f.write(backend_requirements)
    
    # Run script
    run_script = '''#!/bin/bash
# NAVUS Web App Launcher

echo "🚀 Starting NAVUS Credit Card Advisor Web App"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "backend.py" ]; then
    echo "❌ Error: backend.py not found. Please run from the WebApp directory."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start backend in background
echo "🔥 Starting backend API server..."
python backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found"
    kill $BACKEND_PID
    exit 1
fi

# Install and start frontend
echo "🎨 Starting frontend development server..."
cd frontend

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "✅ Starting React development server..."
npm start &
FRONTEND_PID=$!

# Wait for user to stop
echo ""
echo "🎉 NAVUS is now running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo "\\n🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT

# Wait indefinitely
wait'''
    
    with open('/Users/joebanerjee/NAVUS/WebApp/run_app.sh', 'w') as f:
        f.write(run_script)
    
    # Make executable
    os.chmod('/Users/joebanerjee/NAVUS/WebApp/run_app.sh', 0o755)
    
    print("✅ Created deployment scripts")

def main():
    """Create complete web application"""
    
    # Create directories
    os.makedirs('/Users/joebanerjee/NAVUS/WebApp', exist_ok=True)
    os.makedirs('/Users/joebanerjee/NAVUS/WebApp/frontend/src', exist_ok=True)
    os.makedirs('/Users/joebanerjee/NAVUS/WebApp/frontend/public', exist_ok=True)
    
    print("🌐 Creating NAVUS Web Application...")
    
    create_fastapi_backend()
    create_frontend()
    create_deployment_scripts()
    
    print("\\n🎯 Web Application Created!")
    print("Files created in /Users/joebanerjee/NAVUS/WebApp/:")
    print("  • backend.py - FastAPI server")
    print("  • frontend/ - React application")
    print("  • requirements.txt - Python dependencies")  
    print("  • run_app.sh - Launch script")

if __name__ == "__main__":
    main()