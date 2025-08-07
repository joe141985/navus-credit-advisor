#!/usr/bin/env python3
"""
NAVUS Interactive Testing Session
Test the LLM with various scenarios to understand its capabilities
"""

import pandas as pd
import time
import json

class NAVUSInteractiveTester:
    def __init__(self):
        self.dataset_df = None
        self.load_dataset()
        self.session_history = []
        
    def load_dataset(self):
        """Load the credit card dataset"""
        try:
            self.dataset_df = pd.read_csv("/Users/joebanerjee/NAVUS/Data/master_card_dataset_cleaned.csv")
            print(f"✅ Loaded {len(self.dataset_df)} credit cards for intelligent responses")
        except Exception as e:
            print(f"⚠️ Using sample data: {e}")
            sample_cards = [
                {"name": "American Express Cobalt Card", "issuer": "American Express", "category": "basic", "annual_fee": 0, "rewards_type": "Membership Rewards Points", "features": "5x points on dining and transit"},
                {"name": "RBC Avion Visa Infinite", "issuer": "RBC", "category": "travel", "annual_fee": 120, "rewards_type": "Avion Points", "features": "Travel insurance, flexible redemptions"},
                {"name": "TD Cash Back Visa Infinite", "issuer": "TD", "category": "cashback", "annual_fee": 139, "rewards_type": "Cash Back", "features": "3% back on groceries and gas"},
                {"name": "RBC Student Visa", "issuer": "RBC", "category": "student", "annual_fee": 0, "rewards_type": "RBC Rewards", "features": "No income requirement, builds credit"},
                {"name": "Capital One Secured Mastercard", "issuer": "Capital One", "category": "secured", "annual_fee": 59, "rewards_type": "None", "features": "Guaranteed approval, credit building"},
                {"name": "American Express Platinum Card", "issuer": "American Express", "category": "premium", "annual_fee": 699, "rewards_type": "Membership Rewards", "features": "Airport lounge access, hotel status"},
                {"name": "Tangerine Money-Back Credit Card", "issuer": "Tangerine", "category": "cashback", "annual_fee": 0, "rewards_type": "Cash Back", "features": "2% back on chosen categories"},
                {"name": "BMO CashBack Mastercard", "issuer": "BMO", "category": "cashback", "annual_fee": 0, "rewards_type": "Cash Back", "features": "3% cashback on groceries"}
            ]
            self.dataset_df = pd.DataFrame(sample_cards)
    
    def generate_response(self, user_message, user_profile=None):
        """Generate intelligent response using dataset"""
        start_time = time.time()
        message_lower = user_message.lower()
        
        # Analyze user intent and find relevant cards
        relevant_cards = []
        response = ""
        
        # Travel-related queries
        if any(word in message_lower for word in ['travel', 'trip', 'vacation', 'airline', 'airport', 'lounge']):
            travel_cards = self.dataset_df[self.dataset_df['category'] == 'travel']
            premium_travel = self.dataset_df[
                (self.dataset_df['category'] == 'premium') | 
                (self.dataset_df['features'].str.contains('travel|lounge|airport', case=False, na=False))
            ]
            relevant_cards = pd.concat([travel_cards, premium_travel]).drop_duplicates().head(3)
            
            if user_profile and user_profile.get('income'):
                income = int(user_profile.get('income', 0))
                if income > 100000:
                    response = "With your income level, you have access to premium travel cards with exclusive benefits:\n\n"
                elif income > 60000:
                    response = "For your income range, here are excellent travel reward options:\n\n"
                else:
                    response = "Here are travel cards that fit various income levels:\n\n"
            else:
                response = "For travel rewards in Canada, here are the top options:\n\n"
        
        # Cashback queries
        elif any(word in message_lower for word in ['cashback', 'cash back', 'groceries', 'gas', 'spending']):
            cashback_cards = self.dataset_df[self.dataset_df['category'] == 'cashback']
            relevant_cards = cashback_cards.head(3)
            response = "For maximizing cashback in Canada:\n\n"
        
        # Student queries
        elif any(word in message_lower for word in ['student', 'first card', 'college', 'university', 'young']):
            student_cards = self.dataset_df[self.dataset_df['category'] == 'student']
            no_fee_cards = self.dataset_df[self.dataset_df['annual_fee'] == 0]
            relevant_cards = pd.concat([student_cards, no_fee_cards]).drop_duplicates().head(3)
            response = "Perfect credit cards for students and first-time cardholders:\n\n"
        
        # Premium/luxury queries
        elif any(word in message_lower for word in ['premium', 'luxury', 'high-end', 'exclusive', 'platinum']):
            premium_cards = self.dataset_df[
                (self.dataset_df['category'] == 'premium') | 
                (self.dataset_df['annual_fee'] > 300)
            ]
            relevant_cards = premium_cards.head(3)
            response = "For premium benefits and exclusive perks:\n\n"
        
        # Secured/credit building
        elif any(word in message_lower for word in ['secured', 'build credit', 'bad credit', 'rebuild']):
            secured_cards = self.dataset_df[self.dataset_df['category'] == 'secured']
            relevant_cards = secured_cards.head(3)
            response = "For building or rebuilding your credit:\n\n"
        
        # No fee queries
        elif any(word in message_lower for word in ['no fee', 'free', 'annual fee']):
            no_fee_cards = self.dataset_df[self.dataset_df['annual_fee'] == 0]
            relevant_cards = no_fee_cards.head(4)
            response = "Excellent no-fee credit cards in Canada:\n\n"
        
        # Specific card queries
        elif any(card in message_lower for card in ['amex', 'american express', 'rbc', 'td', 'bmo', 'cibc', 'scotia']):
            for issuer in ['American Express', 'RBC', 'TD', 'BMO', 'CIBC', 'Scotia']:
                if issuer.lower() in message_lower or issuer.replace(' ', '').lower() in message_lower:
                    issuer_cards = self.dataset_df[self.dataset_df['issuer'].str.contains(issuer, case=False, na=False)]
                    relevant_cards = issuer_cards.head(3)
                    response = f"Here are the top {issuer} credit cards:\n\n"
                    break
        
        # General advice
        else:
            # Provide personalized general advice
            if user_profile:
                persona = user_profile.get('persona', '')
                income = user_profile.get('income', '')
                location = user_profile.get('location', '')
                
                response = f"Based on your profile"
                if persona:
                    response += f" as a {persona.replace('_', ' ')}"
                if income:
                    response += f" with ${income} annual income"
                if location:
                    response += f" in {location}"
                response += ", here's my recommendation:\n\n"
                
                # Give targeted advice based on persona
                if 'student' in persona:
                    relevant_cards = self.dataset_df[
                        (self.dataset_df['category'] == 'student') | 
                        (self.dataset_df['annual_fee'] == 0)
                    ].head(3)
                elif 'travel' in persona:
                    relevant_cards = self.dataset_df[
                        (self.dataset_df['category'] == 'travel') | 
                        (self.dataset_df['features'].str.contains('travel', case=False, na=False))
                    ].head(3)
                elif 'cashback' in persona:
                    relevant_cards = self.dataset_df[self.dataset_df['category'] == 'cashback'].head(3)
                elif 'premium' in persona:
                    relevant_cards = self.dataset_df[
                        (self.dataset_df['category'] == 'premium') | 
                        (self.dataset_df['annual_fee'] > 200)
                    ].head(3)
                else:
                    relevant_cards = self.dataset_df.head(3)
            else:
                response = "I'd be happy to help you find the perfect Canadian credit card! To give you the best recommendation, could you tell me:\n\n"
                response += "• **Your main goal**: Travel rewards, cashback, or building credit?\n"
                response += "• **Spending habits**: Where do you spend most? (dining, groceries, gas, general)\n"
                response += "• **Annual fee preference**: Open to paying for premium benefits?\n"
                response += "• **Income level**: This helps determine which cards you qualify for\n\n"
                response += "Or try asking something like:\n"
                response += "• 'Best travel card for someone making $70K'\n"
                response += "• 'No-fee cashback card for groceries'\n"
                response += "• 'Student credit card with no income requirement'"
        
        # Add specific card recommendations if we found relevant ones
        if len(relevant_cards) > 0:
            for _, card in relevant_cards.iterrows():
                annual_fee = f"${card['annual_fee']}" if card['annual_fee'] > 0 else "No annual fee"
                rewards = card.get('rewards_type', 'N/A')
                features = card.get('features', '')
                
                response += f"• **{card['name']}** ({card['issuer']})\n"
                response += f"  - {annual_fee}\n"
                if rewards and rewards != 'N/A':
                    response += f"  - Rewards: {rewards}\n"
                if features:
                    response += f"  - Key features: {features}\n"
                response += "\n"
        
        # Add personalized advice based on profile
        if user_profile:
            income_val = user_profile.get('income', '')
            if income_val:
                try:
                    income_num = int(income_val)
                    if income_num < 30000:
                        response += "\n💡 **Tip**: With your income, focus on no-fee cards to build credit without additional costs."
                    elif income_num > 80000:
                        response += "\n💡 **Tip**: Your income qualifies you for premium cards - consider if the benefits justify annual fees."
                except:
                    pass
            
            location = user_profile.get('location', '')
            if location:
                response += f"\n🇨🇦 All recommended cards are available in {location} and across Canada."
        
        processing_time = time.time() - start_time
        
        # Generate follow-up suggestions
        suggestions = self.get_follow_up_suggestions(message_lower)
        
        return response, processing_time, suggestions
    
    def get_follow_up_suggestions(self, message_lower):
        """Generate contextual follow-up questions"""
        if any(word in message_lower for word in ['travel', 'trip']):
            return [
                "Which travel card has no foreign transaction fees?",
                "Compare RBC Avion vs TD Aeroplan cards",
                "Best card for airport lounge access?"
            ]
        elif any(word in message_lower for word in ['cashback', 'groceries']):
            return [
                "How do I maximize cashback rewards?",
                "Best rotating category cards?",
                "Compare cashback vs travel rewards"
            ]
        elif any(word in message_lower for word in ['student', 'first']):
            return [
                "How to build credit as a student?",
                "When should I upgrade from a student card?",
                "Best practices for first-time cardholders?"
            ]
        else:
            return [
                "Should I pay an annual fee for better rewards?",
                "How do I choose between points and cashback?",
                "What credit score do I need for premium cards?"
            ]
    
    def interactive_session(self):
        """Run interactive testing session"""
        print("🎯 NAVUS Interactive Testing Session")
        print("=" * 60)
        print("Test different scenarios to see how NAVUS responds!")
        print()
        print("💡 Commands:")
        print("  'profile' - Set user profile")
        print("  'scenarios' - See suggested test scenarios")
        print("  'history' - Show conversation history")
        print("  'quit' - Exit")
        print()
        
        user_profile = {}
        
        while True:
            try:
                print("-" * 60)
                user_input = input("\n🤔 Ask NAVUS: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for testing NAVUS!")
                    self.save_session_history()
                    break
                
                elif user_input.lower() == 'profile':
                    user_profile = self.set_profile()
                    continue
                
                elif user_input.lower() == 'scenarios':
                    self.show_test_scenarios()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif not user_input:
                    continue
                
                # Generate response
                print("\n🤖 NAVUS is thinking...")
                response, processing_time, suggestions = self.generate_response(user_input, user_profile)
                
                # Display response
                print(f"\n💬 NAVUS Response:")
                print(response)
                print(f"\n⏱️ Processing time: {processing_time:.3f}s")
                
                # Show suggestions
                if suggestions:
                    print(f"\n💡 You might also ask:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                
                # Save to history
                self.session_history.append({
                    'question': user_input,
                    'response': response,
                    'processing_time': processing_time,
                    'profile': user_profile.copy()
                })
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for testing NAVUS!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def set_profile(self):
        """Set user profile for personalized responses"""
        print("\n👤 Set User Profile for Personalized Testing")
        print("-" * 40)
        
        persona_options = [
            "student", "frequent_traveler", "cashback_focused", 
            "premium_seeker", "credit_builder", "business_owner"
        ]
        
        print("Available personas:")
        for i, persona in enumerate(persona_options, 1):
            print(f"  {i}. {persona.replace('_', ' ').title()}")
        
        persona = input("\nPersona (or type custom): ").strip()
        if persona.isdigit() and 1 <= int(persona) <= len(persona_options):
            persona = persona_options[int(persona) - 1]
        
        income = input("Annual income (optional, e.g., 75000): ").strip()
        location = input("Province (optional, e.g., BC, ON, QC): ").strip()
        
        profile = {
            "persona": persona if persona else "",
            "income": income if income else "",
            "location": location if location else ""
        }
        
        print(f"\n✅ Profile set: {profile}")
        return profile
    
    def show_test_scenarios(self):
        """Show suggested test scenarios"""
        print("\n🎭 Suggested Test Scenarios")
        print("-" * 30)
        
        scenarios = [
            {
                "title": "High-Income Travel Enthusiast",
                "profile": "frequent_traveler, $120000, BC",
                "question": "What's the best premium travel card with lounge access?"
            },
            {
                "title": "Budget-Conscious Student", 
                "profile": "student, no income, ON",
                "question": "I need my first credit card to build credit history"
            },
            {
                "title": "Grocery Shopping Family",
                "profile": "cashback_focused, $65000, AB", 
                "question": "Best cashback card for groceries and gas?"
            },
            {
                "title": "Credit Rebuilding",
                "profile": "credit_builder, $40000, QC",
                "question": "I have bad credit and need to rebuild. What secured card should I get?"
            },
            {
                "title": "No Annual Fee Seeker",
                "profile": "cashback_focused, $55000, BC",
                "question": "Show me the best no-fee cards with good rewards"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. **{scenario['title']}**")
            print(f"   Profile: {scenario['profile']}")
            print(f"   Question: \"{scenario['question']}\"")
    
    def show_history(self):
        """Show conversation history"""
        if not self.session_history:
            print("\n📝 No conversation history yet.")
            return
        
        print(f"\n📝 Conversation History ({len(self.session_history)} interactions)")
        print("-" * 50)
        
        for i, item in enumerate(self.session_history, 1):
            print(f"\n{i}. Q: {item['question']}")
            print(f"   A: {item['response'][:100]}...")
            print(f"   Time: {item['processing_time']:.3f}s")
    
    def save_session_history(self):
        """Save session history to file"""
        if self.session_history:
            timestamp = int(time.time())
            filename = f"/Users/joebanerjee/NAVUS/Reports/interactive_session_{timestamp}.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'total_interactions': len(self.session_history),
                        'history': self.session_history
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Session history saved to: {filename}")
            except Exception as e:
                print(f"⚠️ Could not save history: {e}")

def main():
    """Main function"""
    tester = NAVUSInteractiveTester()
    tester.interactive_session()

if __name__ == "__main__":
    main()