"""
Enhanced NAVUS Backend with Chart Generation and Advanced Financial Analysis
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime
import base64
from io import BytesIO

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

try:
    from chart_generator import NAVUSChartGenerator
except ImportError:
    st.error("Chart generator not found. Please ensure chart_generator.py is in the Scripts directory.")
    NAVUSChartGenerator = None

# Streamlit page config
st.set_page_config(
    page_title="NAVUS - Enhanced Financial Advisor", 
    page_icon="💳",
    layout="wide"
)

class EnhancedNAVUSApp:
    """Enhanced NAVUS application with chart generation capabilities"""
    
    def __init__(self):
        self.chart_generator = NAVUSChartGenerator() if NAVUSChartGenerator else None
        self.load_card_data()
        
    def load_card_data(self):
        """Load credit card database"""
        try:
            self.cards_df = pd.read_csv('../Data/master_card_dataset_cleaned.csv')
            st.session_state['cards_loaded'] = True
        except FileNotFoundError:
            st.error("Credit card database not found. Please ensure master_card_dataset_cleaned.csv exists.")
            self.cards_df = pd.DataFrame()
            st.session_state['cards_loaded'] = False
    
    def run(self):
        """Main application interface"""
        
        # Header
        st.title("🏦 NAVUS - Enhanced Financial Advisor")
        st.markdown("*Your AI-powered guide to Canadian credit cards and debt management*")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose Analysis Type", [
            "💳 Credit Card Recommendations",
            "📊 Debt Payoff Planner", 
            "📈 Multi-Card Strategy",
            "🎯 Credit Score Planner",
            "⚖️ Strategy Comparison",
            "🔍 Card Database Explorer"
        ])
        
        # Route to appropriate page
        if page == "💳 Credit Card Recommendations":
            self.credit_card_recommendations()
        elif page == "📊 Debt Payoff Planner":
            self.debt_payoff_planner()
        elif page == "📈 Multi-Card Strategy":
            self.multi_card_strategy()
        elif page == "🎯 Credit Score Planner":
            self.credit_score_planner()
        elif page == "⚖️ Strategy Comparison":
            self.strategy_comparison()
        elif page == "🔍 Card Database Explorer":
            self.database_explorer()
    
    def credit_card_recommendations(self):
        """Credit card recommendation interface"""
        st.header("💳 Credit Card Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Preferences")
            annual_fee = st.selectbox("Annual Fee Preference", ["No Fee ($0)", "Low Fee ($1-99)", "Medium Fee ($100-199)", "High Fee ($200+)"])
            card_type = st.selectbox("Card Type", ["Any", "Cashback", "Travel", "Premium", "Student", "Secured", "Business"])
            min_income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000, step=5000)
        
        with col2:
            st.subheader("Usage Patterns")
            monthly_spend = st.number_input("Monthly Spending ($)", min_value=0, max_value=10000, value=1500, step=100)
            spending_categories = st.multiselect("Main Spending Categories", 
                ["Groceries", "Gas", "Restaurants", "Travel", "Online Shopping", "Bills"])
        
        if st.button("Get Recommendations", type="primary"):
            recommendations = self.get_card_recommendations(annual_fee, card_type, min_income, monthly_spend, spending_categories)
            
            st.subheader("🎯 Recommended Cards")
            for i, card in enumerate(recommendations[:3], 1):
                with st.expander(f"#{i} {card['name']} - {card['issuer']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Annual Fee", f"${card.get('annual_fee', 'N/A')}")
                        st.metric("Network", card.get('network', 'N/A'))
                    
                    with col2:
                        st.metric("Rewards", card.get('rewards_type', 'N/A'))
                        st.metric("Category", card.get('category', 'N/A').title())
                    
                    with col3:
                        welcome_bonus = card.get('welcome_bonus_amount', 0)
                        if welcome_bonus and welcome_bonus > 0:
                            st.metric("Welcome Bonus", f"${welcome_bonus:,.0f}")
                        else:
                            st.metric("Welcome Bonus", "None")
                    
                    st.markdown(f"**Features:** {card.get('features', 'N/A')}")
                    
                    if card.get('apply_url'):
                        st.markdown(f"[Apply Now]({card['apply_url']})")
    
    def debt_payoff_planner(self):
        """Debt payoff planning interface with charts"""
        st.header("📊 Debt Payoff Planner")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Debt Information")
            debt_amount = st.number_input("Total Debt Amount ($)", min_value=100, max_value=100000, value=5000, step=100)
            interest_rate = st.slider("Annual Interest Rate (%)", min_value=5.0, max_value=30.0, value=19.99, step=0.1)
            min_payment = st.number_input("Current Monthly Payment ($)", min_value=25, max_value=5000, value=125, step=25)
        
        with col2:
            st.subheader("Payoff Strategy")
            accelerated_payment = st.number_input("Proposed Monthly Payment ($)", min_value=min_payment, max_value=10000, value=300, step=25)
            extra_payment_source = st.selectbox("Extra Payment Source", 
                ["Increased Income", "Reduced Expenses", "Tax Refund", "Bonus", "Side Hustle"])
        
        if st.button("Analyze Payoff Strategy", type="primary"):
            # Generate analysis
            analysis = self.calculate_debt_payoff(debt_amount, interest_rate, min_payment, accelerated_payment)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Time Saved", f"{analysis['months_saved']} months", 
                         delta=f"-{analysis['months_saved']} months")
            
            with col2:
                st.metric("Interest Saved", f"${analysis['interest_saved']:,.0f}",
                         delta=f"-${analysis['interest_saved']:,.0f}")
            
            with col3:
                st.metric("Strategy Score", f"{analysis['score']:.1f}/10",
                         delta=f"{analysis['score']:.1f}")
            
            with col4:
                st.metric("Total Savings", f"${analysis['total_savings']:,.0f}",
                         delta=f"-${analysis['total_savings']:,.0f}")
            
            # Generate and display chart
            if self.chart_generator:
                chart_b64 = self.chart_generator.generate_debt_payoff_comparison(
                    debt_amount, interest_rate, min_payment, accelerated_payment)
                
                st.subheader("📈 Payoff Timeline Visualization")
                st.image(f"data:image/png;base64,{chart_b64}")
            
            # Detailed breakdown
            st.subheader("💡 Personalized Recommendations")
            
            if analysis['score'] >= 9.0:
                st.success("🌟 Excellent Strategy! This plan will save you significant money and time.")
            elif analysis['score'] >= 7.0:
                st.info("👍 Good Strategy! Consider these optimizations...")
            else:
                st.warning("⚠️ This strategy needs improvement. Consider these alternatives...")
            
            # Balance transfer recommendation
            if interest_rate > 20:
                st.subheader("🔄 Balance Transfer Opportunity")
                bt_cards = self.get_balance_transfer_cards()
                if not bt_cards.empty:
                    st.markdown("Consider these balance transfer cards:")
                    for _, card in bt_cards.head(2).iterrows():
                        st.markdown(f"- **{card['name']}** ({card['issuer']}) - ${card['annual_fee']} annual fee")
    
    def multi_card_strategy(self):
        """Multi-card debt strategy interface"""
        st.header("📈 Multi-Card Debt Strategy")
        
        st.markdown("Enter information for each of your credit cards:")
        
        # Dynamic card input
        if 'num_cards' not in st.session_state:
            st.session_state.num_cards = 2
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add Card"):
                st.session_state.num_cards += 1
            if st.button("Remove Card") and st.session_state.num_cards > 1:
                st.session_state.num_cards -= 1
        
        # Collect card data
        cards_data = []
        for i in range(st.session_state.num_cards):
            st.subheader(f"Card {i+1}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                name = st.text_input(f"Card Name", value=f"Card {i+1}", key=f"name_{i}")
            with col2:
                balance = st.number_input(f"Balance ($)", min_value=0, max_value=50000, value=2000, key=f"balance_{i}")
            with col3:
                rate = st.slider(f"Interest Rate (%)", min_value=5.0, max_value=35.0, value=20.0, key=f"rate_{i}")
            
            cards_data.append({'name': name, 'balance': balance, 'rate': rate})
        
        # Strategy selection
        strategy = st.selectbox("Payoff Strategy", ["Avalanche (Highest Rate First)", "Snowball (Lowest Balance First)"])
        total_budget = st.number_input("Total Monthly Budget for Debt ($)", min_value=100, max_value=5000, value=500)
        
        if st.button("Optimize Strategy", type="primary"):
            # Generate strategy analysis
            optimized_plan = self.optimize_multi_card_strategy(cards_data, strategy, total_budget)
            
            # Display results
            st.subheader("🎯 Optimized Payment Strategy")
            
            # Strategy overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Debt", f"${optimized_plan['total_debt']:,.0f}")
            with col2:
                st.metric("Payoff Time", f"{optimized_plan['payoff_months']} months")
            with col3:
                st.metric("Total Interest", f"${optimized_plan['total_interest']:,.0f}")
            
            # Payment plan table
            st.subheader("📋 Monthly Payment Plan")
            plan_df = pd.DataFrame(optimized_plan['payment_plan'])
            st.dataframe(plan_df, use_container_width=True)
            
            # Generate visualization
            if self.chart_generator:
                chart_b64 = self.chart_generator.generate_multi_card_strategy_chart(cards_data)
                st.subheader("📊 Strategy Visualization")
                st.image(f"data:image/png;base64,{chart_b64}")
    
    def credit_score_planner(self):
        """Credit score improvement planning"""
        st.header("🎯 Credit Score Improvement Planner")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Situation")
            current_score = st.slider("Current Credit Score", min_value=300, max_value=850, value=650)
            payment_history = st.selectbox("Payment History", ["Excellent", "Good", "Fair", "Poor"])
            credit_utilization = st.slider("Credit Utilization (%)", min_value=0, max_value=100, value=30)
            
        with col2:
            st.subheader("Goals & Timeline")
            target_score = st.slider("Target Credit Score", min_value=current_score, max_value=850, value=750)
            timeline_months = st.selectbox("Timeline", [6, 12, 18, 24, 36])
            
        if st.button("Create Improvement Plan", type="primary"):
            # Generate improvement plan
            plan = self.create_credit_improvement_plan(current_score, target_score, timeline_months, credit_utilization)
            
            # Display plan
            st.subheader("📈 Credit Score Improvement Plan")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Projected Score", f"{plan['projected_score']}", 
                         delta=f"+{plan['projected_score'] - current_score}")
            with col2:
                st.metric("Success Probability", f"{plan['success_rate']}%")
            with col3:
                st.metric("Timeline", f"{timeline_months} months")
            
            # Action plan
            st.subheader("🚀 Action Steps")
            for i, step in enumerate(plan['action_steps'], 1):
                st.markdown(f"{i}. {step}")
            
            # Generate timeline chart
            if self.chart_generator:
                chart_b64 = self.chart_generator.generate_credit_score_timeline(current_score, timeline_months)
                st.subheader("📊 Score Improvement Timeline")
                st.image(f"data:image/png;base64,{chart_b64}")
    
    def strategy_comparison(self):
        """Compare different debt strategies"""
        st.header("⚖️ Debt Strategy Comparison")
        
        # Input scenario
        st.subheader("Your Debt Situation")
        col1, col2 = st.columns(2)
        
        with col1:
            total_debt = st.number_input("Total Debt ($)", min_value=1000, max_value=100000, value=10000)
            avg_rate = st.slider("Average Interest Rate (%)", min_value=5.0, max_value=30.0, value=21.0)
        
        with col2:
            monthly_budget = st.number_input("Monthly Budget ($)", min_value=200, max_value=5000, value=500)
            current_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
        
        if st.button("Compare Strategies", type="primary"):
            # Analyze different strategies
            strategies = self.compare_debt_strategies(total_debt, avg_rate, monthly_budget, current_score)
            
            # Display comparison table
            st.subheader("📊 Strategy Comparison")
            comparison_df = pd.DataFrame(strategies).T
            st.dataframe(comparison_df, use_container_width=True)
            
            # Generate radar chart comparison
            if self.chart_generator:
                # Prepare data for radar chart
                radar_data = {}
                for strategy_name, data in strategies.items():
                    radar_data[strategy_name] = [
                        data['Interest Savings'] / 10,  # Normalize to 0-10 scale
                        data['Time Efficiency'],
                        data['Feasibility'],
                        data['Credit Impact'],
                        data['Risk Level']
                    ]
                
                chart_b64 = self.chart_generator.generate_strategy_scoring_chart(radar_data)
                st.subheader("🕷️ Strategy Comparison Radar")
                st.image(f"data:image/png;base64,{chart_b64}")
            
            # Recommendation
            best_strategy = max(strategies.items(), key=lambda x: x[1]['Overall Score'])
            st.success(f"🏆 Recommended Strategy: **{best_strategy[0]}** (Score: {best_strategy[1]['Overall Score']:.1f}/10)")
    
    def database_explorer(self):
        """Credit card database explorer"""
        st.header("🔍 Credit Card Database Explorer")
        
        if not st.session_state.get('cards_loaded', False):
            st.error("Card database not loaded.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            issuer_filter = st.multiselect("Issuer", options=self.cards_df['issuer'].unique())
            
        with col2:
            fee_filter = st.selectbox("Annual Fee", ["All", "No Fee", "Low Fee", "Medium Fee", "High Fee"])
            
        with col3:
            category_filter = st.multiselect("Category", options=self.cards_df['category'].unique())
        
        # Apply filters
        filtered_df = self.cards_df.copy()
        
        if issuer_filter:
            filtered_df = filtered_df[filtered_df['issuer'].isin(issuer_filter)]
        
        if fee_filter != "All":
            fee_map = {
                "No Fee": "no_fee",
                "Low Fee": "low_fee", 
                "Medium Fee": "medium_fee",
                "High Fee": "high_fee"
            }
            filtered_df = filtered_df[filtered_df['fee_category'] == fee_map[fee_filter]]
        
        if category_filter:
            filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
        
        # Display results
        st.subheader(f"📋 Results ({len(filtered_df)} cards)")
        
        # Key metrics
        if not filtered_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_fee = filtered_df['annual_fee'].mean()
                st.metric("Average Fee", f"${avg_fee:.0f}")
            
            with col2:
                no_fee_count = len(filtered_df[filtered_df['annual_fee'] == 0])
                st.metric("No-Fee Cards", no_fee_count)
            
            with col3:
                rewards_count = len(filtered_df[filtered_df['rewards_type'].notna()])
                st.metric("Rewards Cards", rewards_count)
            
            with col4:
                welcome_bonus_count = len(filtered_df[filtered_df['has_welcome_bonus'] == True])
                st.metric("Welcome Bonuses", welcome_bonus_count)
            
            # Detailed table
            display_columns = ['name', 'issuer', 'network', 'category', 'annual_fee', 'rewards_type', 'features']
            st.dataframe(filtered_df[display_columns], use_container_width=True)
    
    # Helper methods
    def get_card_recommendations(self, annual_fee, card_type, min_income, monthly_spend, spending_categories):
        """Get personalized card recommendations"""
        if self.cards_df.empty:
            return []
        
        # Apply filters
        filtered_cards = self.cards_df.copy()
        
        # Fee filter
        fee_map = {
            "No Fee ($0)": "no_fee",
            "Low Fee ($1-99)": "low_fee",
            "Medium Fee ($100-199)": "medium_fee", 
            "High Fee ($200+)": "high_fee"
        }
        if annual_fee in fee_map:
            filtered_cards = filtered_cards[filtered_cards['fee_category'] == fee_map[annual_fee]]
        
        # Type filter
        if card_type != "Any":
            filtered_cards = filtered_cards[filtered_cards['category'] == card_type.lower()]
        
        # Convert to list of dicts
        return filtered_cards.head(10).to_dict('records')
    
    def calculate_debt_payoff(self, debt_amount, interest_rate, min_payment, accelerated_payment):
        """Calculate debt payoff scenarios"""
        monthly_rate = interest_rate / 100 / 12
        
        # Minimum payment scenario
        balance = debt_amount
        min_months = 0
        min_interest = 0
        
        while balance > 0 and min_months < 600:
            interest = balance * monthly_rate
            principal = min(min_payment - interest, balance)
            if principal <= 0:
                min_months = 600  # Can't pay off
                break
            balance -= principal
            min_interest += interest
            min_months += 1
        
        # Accelerated payment scenario
        balance = debt_amount
        acc_months = 0
        acc_interest = 0
        
        while balance > 0 and acc_months < 600:
            interest = balance * monthly_rate
            principal = min(accelerated_payment - interest, balance)
            balance -= principal
            acc_interest += interest
            acc_months += 1
        
        # Calculate metrics
        months_saved = min_months - acc_months
        interest_saved = min_interest - acc_interest
        total_savings = interest_saved + (months_saved * min_payment)
        
        # Calculate score
        score = min(10, 5 + (interest_saved / 1000) + (months_saved / 12))
        
        return {
            'months_saved': months_saved,
            'interest_saved': interest_saved,
            'total_savings': total_savings,
            'score': score
        }
    
    def get_balance_transfer_cards(self):
        """Get cards suitable for balance transfers"""
        if self.cards_df.empty:
            return pd.DataFrame()
        
        # Filter for low-fee cards suitable for balance transfers
        bt_cards = self.cards_df[
            (self.cards_df['annual_fee'] <= 100) & 
            (self.cards_df['category'].isin(['cashback', 'basic', 'no_fee']))
        ]
        
        return bt_cards
    
    def optimize_multi_card_strategy(self, cards_data, strategy, total_budget):
        """Optimize multi-card debt payoff strategy"""
        total_debt = sum(card['balance'] for card in cards_data)
        
        # Sort based on strategy
        if strategy == "Avalanche (Highest Rate First)":
            sorted_cards = sorted(cards_data, key=lambda x: x['rate'], reverse=True)
        else:  # Snowball
            sorted_cards = sorted(cards_data, key=lambda x: x['balance'])
        
        # Calculate minimum payments (assume 2% of balance)
        min_payments = []
        total_min_payment = 0
        
        for card in sorted_cards:
            min_pay = max(25, card['balance'] * 0.02)  # 2% or $25 minimum
            min_payments.append(min_pay)
            total_min_payment += min_pay
        
        # Allocate extra payment to priority card
        extra_payment = max(0, total_budget - total_min_payment)
        
        # Create payment plan
        payment_plan = []
        for i, (card, min_pay) in enumerate(zip(sorted_cards, min_payments)):
            if i == 0:  # Priority card gets extra payment
                total_payment = min_pay + extra_payment
            else:
                total_payment = min_pay
            
            payment_plan.append({
                'Card': card['name'],
                'Balance': f"${card['balance']:,.0f}",
                'Rate': f"{card['rate']:.2f}%",
                'Payment': f"${total_payment:.0f}",
                'Priority': i + 1
            })
        
        # Estimate payoff time (simplified)
        payoff_months = max(12, total_debt / total_budget)
        total_interest = total_debt * 0.15  # Rough estimate
        
        return {
            'total_debt': total_debt,
            'payoff_months': int(payoff_months),
            'total_interest': total_interest,
            'payment_plan': payment_plan
        }
    
    def create_credit_improvement_plan(self, current_score, target_score, timeline_months, utilization):
        """Create credit score improvement plan"""
        score_gap = target_score - current_score
        monthly_improvement = score_gap / timeline_months
        projected_score = min(850, current_score + (monthly_improvement * timeline_months))
        
        # Success rate based on timeline and gap
        if score_gap <= 50 and timeline_months >= 12:
            success_rate = 90
        elif score_gap <= 100 and timeline_months >= 18:
            success_rate = 75
        else:
            success_rate = 60
        
        # Action steps
        action_steps = [
            "Pay all bills on time (35% of score)",
            f"Reduce credit utilization to under 10% (currently {utilization}%)",
            "Keep old accounts open to maintain credit history",
            "Monitor credit report monthly for errors",
            "Consider becoming an authorized user on family member's account"
        ]
        
        if utilization > 30:
            action_steps.insert(1, "URGENT: Pay down balances to under 30% utilization")
        
        if current_score < 650:
            action_steps.append("Consider a secured credit card to build payment history")
        
        return {
            'projected_score': int(projected_score),
            'success_rate': success_rate,
            'action_steps': action_steps
        }
    
    def compare_debt_strategies(self, total_debt, avg_rate, monthly_budget, current_score):
        """Compare different debt payoff strategies"""
        strategies = {
            'Balance Transfer': {
                'Interest Savings': 8,
                'Time Efficiency': 7,
                'Feasibility': 6,
                'Credit Impact': 8,
                'Risk Level': 7,
                'Overall Score': 7.2
            },
            'Debt Avalanche': {
                'Interest Savings': 9,
                'Time Efficiency': 8,
                'Feasibility': 8,
                'Credit Impact': 7,
                'Risk Level': 9,
                'Overall Score': 8.2
            },
            'Debt Snowball': {
                'Interest Savings': 6,
                'Time Efficiency': 7,
                'Feasibility': 9,
                'Credit Impact': 8,
                'Risk Level': 9,
                'Overall Score': 7.8
            },
            'Consolidation Loan': {
                'Interest Savings': 7,
                'Time Efficiency': 6,
                'Feasibility': 7,
                'Credit Impact': 6,
                'Risk Level': 8,
                'Overall Score': 6.8
            }
        }
        
        return strategies

# Main app
if __name__ == "__main__":
    app = EnhancedNAVUSApp()
    app.run()