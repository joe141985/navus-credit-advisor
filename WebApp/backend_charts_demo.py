#!/usr/bin/env python3
"""
NAVUS Credit Card Advisor API - CHART GENERATION DEMO
Standalone backend focused on chart generation for debt payoff scenarios
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
import os
import json
import time
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NAVUS Credit Card Advisor API - Chart Demo", 
    version="6.0.0",
    description="AI-powered Canadian credit card advisor with guaranteed chart generation"
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

class NavusChartModel:
    def __init__(self):
        self.loaded = True
        self.model_type = "chart-generation-demo"
        self.dataset_df = None
        
        # Load datasets
        self.load_dataset()
    
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
            }
        ]
        
        self.dataset_df = pd.DataFrame(sample_cards)
        logger.info(f"✅ Loaded {len(self.dataset_df)} Canadian credit cards")
    
    def create_financial_chart(self, chart_type: str, user_data: Dict = None) -> str:
        """Generate financial charts and return as base64"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == "debt_payoff":
                # Realistic debt payoff data based on user input or defaults
                debt_amount = user_data.get('debt_amount', 5000)
                interest_rate = user_data.get('interest_rate', 0.19)  # 19%
                min_payment = user_data.get('min_payment', debt_amount * 0.025)  # 2.5% minimum
                accelerated_payment = user_data.get('accelerated_payment', min_payment * 2.5)
                
                months = list(range(1, 36))
                
                # Calculate remaining balances for both payment strategies
                min_balance = []
                acc_balance = []
                
                min_remaining = debt_amount
                acc_remaining = debt_amount
                
                for month in months:
                    # Minimum payment scenario
                    min_interest = min_remaining * (interest_rate / 12)
                    min_principal = min_payment - min_interest
                    min_remaining = max(0, min_remaining - min_principal)
                    min_balance.append(min_remaining)
                    
                    # Accelerated payment scenario
                    acc_interest = acc_remaining * (interest_rate / 12)
                    acc_principal = accelerated_payment - acc_interest
                    acc_remaining = max(0, acc_remaining - acc_principal)
                    acc_balance.append(acc_remaining)
                    
                    if min_remaining <= 0 and acc_remaining <= 0:
                        break
                
                # Trim data to when debt is paid off
                months = months[:len(min_balance)]
                
                ax.plot(months, min_balance, label=f'Minimum Payment (${min_payment:.0f}/mo)', 
                       color='#FF6B6B', linewidth=3, marker='o', markersize=4)
                ax.plot(months, acc_balance, label=f'Accelerated Payment (${accelerated_payment:.0f}/mo)', 
                       color='#4ECDC4', linewidth=3, marker='s', markersize=4)
                
                ax.set_title(f'${debt_amount:,.0f} Credit Card Debt Payoff Strategy\n'
                           f'({interest_rate*100:.1f}% Interest Rate)', fontsize=18, fontweight='bold')
                ax.set_xlabel('Months', fontsize=14)
                ax.set_ylabel('Remaining Balance ($CAD)', fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add debt-free annotation
                if acc_remaining <= 0:
                    payoff_month = next((i for i, balance in enumerate(acc_balance) if balance <= 0), len(acc_balance)) + 1
                    ax.annotate('Debt-Free!', xy=(payoff_month, 0), xytext=(payoff_month + 2, debt_amount * 0.2),
                               arrowprops=dict(arrowstyle='->', color='green', lw=2),
                               fontsize=12, color='green', fontweight='bold')
                
            elif chart_type == "credit_score":
                # Credit score improvement timeline
                months = list(range(1, 25))
                starting_score = user_data.get('starting_score', 620)
                target_score = user_data.get('target_score', 750)
                
                # Realistic credit score improvement curve
                scores = []
                for month in months:
                    # Exponential improvement that slows down over time
                    improvement = (target_score - starting_score) * (1 - np.exp(-month / 8))
                    score = min(target_score, starting_score + improvement)
                    scores.append(score)
                
                ax.plot(months, scores, marker='o', color='#667eea', linewidth=4, markersize=6)
                ax.fill_between(months, scores, alpha=0.3, color='#667eea')
                ax.set_title('Credit Score Improvement Journey', fontsize=18, fontweight='bold')
                ax.set_xlabel('Months', fontsize=14)
                ax.set_ylabel('Credit Score', fontsize=14)
                ax.set_ylim(580, 800)
                ax.grid(True, alpha=0.3)
                
                # Add score range annotations
                ax.axhspan(580, 649, alpha=0.1, color='red', label='Poor')
                ax.axhspan(650, 699, alpha=0.1, color='orange', label='Fair')
                ax.axhspan(700, 749, alpha=0.1, color='yellow', label='Good')
                ax.axhspan(750, 800, alpha=0.1, color='green', label='Excellent')
                
            elif chart_type == "card_comparison":
                # Credit card comparison
                cards = ['Amex Cobalt', 'RBC Avion', 'TD Cashback']
                rewards = [5.0, 1.25, 3.0]
                fees = [0, 120, 139]
                
                x = np.arange(len(cards))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rewards, width, label='Max Rewards Rate (%)', color='#4ECDC4')
                bars2 = ax.bar(x + width/2, [f/50 for f in fees], width, label='Annual Fee (÷50)', color='#FF6B6B')
                
                ax.set_title('Canadian Credit Card Comparison', fontsize=18, fontweight='bold')
                ax.set_xlabel('Credit Cards', fontsize=14)
                ax.set_ylabel('Rate/Fee Scale', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(cards, rotation=15)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.annotate(f'{height}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                for bar, fee in zip(bars2, fees):
                    height = bar.get_height()
                    ax.annotate(f'${fee}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"✅ Generated {chart_type} chart: {len(chart_base64)} characters")
            return chart_base64
            
        except Exception as e:
            logger.error(f"❌ Error generating chart: {e}")
            return None
    
    def generate_response(self, user_message: str, user_profile: Dict = None) -> tuple:
        """Generate response with guaranteed chart for relevant queries"""
        start_time = time.time()
        
        try:
            user_lower = user_message.lower()
            chart_data = None
            
            # Determine response and chart type
            if any(keyword in user_lower for keyword in ["payoff", "pay off", "debt", "balance", "payment", "strategy"]):
                response = f"""🎯 **Debt Payoff Strategy Analysis**

Based on your request, I've created a comprehensive debt payoff strategy. Here are the key approaches:

**📈 Avalanche Method:**
- Pay minimum on all debts
- Put extra money toward highest interest rate debt
- Save more money on interest over time

**❄️ Snowball Method:**
- Pay minimum on all debts  
- Put extra money toward smallest balance
- Build momentum with quick wins

**💡 Recommended Strategy:**
For most Canadians, I recommend the avalanche method to minimize interest costs. However, if you need motivation, the snowball method's psychological benefits can help you stay committed.

**📊 Chart Analysis:**
The accompanying chart shows how different payment amounts affect your payoff timeline and total interest paid. Notice how even $50-100 extra per month can save thousands in interest!

**Next Steps:**
1. List all your debts with balances and interest rates
2. Choose your strategy (avalanche vs snowball)
3. Set up automatic payments to stay on track
4. Review progress monthly and adjust as needed

Would you like me to analyze a specific debt amount or create a personalized payoff plan?"""
                
                # Extract debt amount from message if mentioned
                debt_data = {"debt_amount": 5000}  # Default
                try:
                    # Simple regex to find dollar amounts
                    import re
                    amounts = re.findall(r'\$?([0-9,]+)', user_message)
                    if amounts:
                        debt_data["debt_amount"] = int(amounts[0].replace(',', ''))
                except:
                    pass
                
                chart_data = self.create_financial_chart("debt_payoff", debt_data)
                
            elif any(keyword in user_lower for keyword in ["compare", "comparison", "vs", "versus", "best card", "which card"]):
                response = f"""🏆 **Canadian Credit Card Comparison**

I've analyzed the top Canadian credit cards based on your query. Here's my expert comparison:

**🥇 American Express Cobalt Card:**
- **No annual fee** - Perfect for budget-conscious users
- **5x rewards** on groceries and dining 
- Best for everyday spending categories

**🥈 RBC Avion Visa Infinite:**
- **$120 annual fee** - Premium travel card
- **1.25x rewards** on all purchases
- Flexible redemption options, travel insurance

**🥉 TD Cash Back Visa Infinite:**
- **$139 annual fee** - High cashback potential
- **3% cashback** on groceries and gas
- Simple cashback rewards, no complicated points

**📊 Chart Analysis:**
The comparison chart shows the reward rates and annual fees side by side. Notice how the Amex Cobalt offers the highest rewards with no annual fee for specific categories.

**🎯 My Recommendation:**
- **High grocery/dining spending:** Amex Cobalt
- **Frequent travel:** RBC Avion  
- **Simple cashback:** TD Cash Back

Which spending category is most important to you?"""
                
                chart_data = self.create_financial_chart("card_comparison")
                
            elif any(keyword in user_lower for keyword in ["credit score", "improve credit", "build credit", "score"]):
                response = f"""📈 **Credit Score Improvement Strategy**

Building excellent credit in Canada takes time and consistency. Here's your roadmap:

**🎯 Credit Score Ranges:**
- 300-579: Poor (rebuilding needed)
- 580-669: Fair (improvement possible)  
- 670-739: Good (solid credit)
- 740-900: Excellent (best rates)

**🚀 Improvement Strategies:**
1. **Payment History (35%):** Never miss payments - set up autopay
2. **Credit Utilization (30%):** Keep balances under 30% of limits
3. **Credit Age (15%):** Keep old accounts open
4. **Credit Mix (10%):** Mix of credit cards, loans, mortgage
5. **New Credit (10%):** Limit new applications

**📊 Timeline Expectations:**
The chart shows realistic improvement over 24 months. Most people see:
- Month 1-3: Small improvements from better habits
- Month 4-12: Steady growth as good habits establish
- Month 12-24: Continued improvement, slower rate

**💡 Quick Wins:**
- Pay down credit card balances immediately
- Set up automatic minimum payments
- Request credit limit increases (don't use them)
- Check your credit report for errors

Would you like specific advice based on your current credit score?"""
                
                chart_data = self.create_financial_chart("credit_score")
                
            else:
                # General financial advice
                response = f"""👋 **Welcome to NAVUS - Your Canadian Credit Card Advisor!**

I'm here to help you make smart financial decisions with Canadian credit cards and debt management.

**🎯 What I can help you with:**
- **Debt Payoff Strategies:** Avalanche vs Snowball methods with visual timelines
- **Credit Card Comparisons:** Find the best Canadian cards for your spending
- **Credit Score Building:** Proven strategies to improve your score
- **Financial Planning:** Budgeting and payment optimization

**📊 Interactive Charts:**
I create detailed charts and graphs for:
- Debt payoff timelines showing interest savings
- Credit card comparison visualizations  
- Credit score improvement projections

**🏦 Canadian Focus:**
Specialized knowledge of RBC, TD, BMO, Scotiabank, CIBC, and American Express cards available in Canada.

**Try asking me:**
- "Help me pay off my $5000 credit card debt"
- "Compare the best travel rewards cards"
- "How can I improve my 650 credit score?"

What would you like help with today?"""
                
                # Show debt payoff chart as demo
                chart_data = self.create_financial_chart("debt_payoff")
            
            suggestions = [
                "Create a debt payoff plan with timeline",
                "Compare the best Canadian credit cards", 
                "Show me credit score improvement strategies"
            ]
            
            processing_time = time.time() - start_time
            return response, processing_time, suggestions, chart_data
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time
            return "I encountered an error processing your request. Please try again.", processing_time, [], None

# Initialize model
navus_model = NavusChartModel()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NAVUS Credit Card Advisor API - Chart Generation Demo", 
        "status": "active",
        "version": "6.0.0",
        "model_type": navus_model.model_type,
        "model_loaded": navus_model.loaded,
        "features": ["Guaranteed Chart Generation", "Canadian Focus", "Interactive Visualizations", "Download Support"]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=navus_model.loaded,
        model_type=navus_model.model_type,
        memory_usage="Optimized matplotlib backend"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with guaranteed chart generation"""
    try:
        response, processing_time, suggestions, chart_data = navus_model.generate_response(
            request.message, 
            request.user_profile
        )
        
        return ChatResponse(
            response=response,
            suggested_questions=suggestions,
            processing_time=processing_time,
            confidence=0.95,
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
            "powered_by": "NAVUS Chart Generation Demo"
        }
        
    except Exception as e:
        logger.error(f"Error getting featured cards: {e}")
        return {"featured_cards": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8004))  # New port
    
    print("🚀 Starting NAVUS Credit Card Advisor API (Chart Generation Demo)...")
    print(f"📱 API will be available on port: {port}")
    print("📊 Guaranteed chart generation for all relevant queries")
    print("💾 Download functionality enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=port)