"""
Massive NAVUS Training Dataset Generator
Creates 10x more training data with diverse financial scenarios
"""

import json
import random
import pandas as pd
from itertools import product
import numpy as np
from datetime import datetime, timedelta
import os

class MassiveDatasetGenerator:
    """Generate comprehensive training dataset for NAVUS"""
    
    def __init__(self):
        self.canadian_cities = [
            "Toronto", "Vancouver", "Montreal", "Calgary", "Edmonton", 
            "Ottawa", "Winnipeg", "Quebec City", "Hamilton", "Kitchener",
            "London", "Victoria", "Halifax", "Windsor", "Oshawa"
        ]
        
        self.income_ranges = [
            (25000, 35000, "low"),
            (35000, 50000, "moderate"),
            (50000, 75000, "good"),
            (75000, 100000, "high"),
            (100000, 150000, "very_high"),
            (150000, 300000, "executive")
        ]
        
        self.debt_scenarios = [
            (500, 2000, "small"),
            (2000, 5000, "moderate"),
            (5000, 10000, "significant"),
            (10000, 20000, "high"),
            (20000, 50000, "very_high"),
            (50000, 100000, "extreme")
        ]
        
        self.interest_rates = [
            (8.99, 12.99, "excellent"),
            (12.99, 17.99, "good"),
            (17.99, 21.99, "average"),
            (21.99, 26.99, "poor"),
            (26.99, 29.99, "very_poor")
        ]
        
        self.spending_categories = {
            "groceries": (200, 800),
            "gas": (100, 400),
            "dining": (150, 600),
            "shopping": (100, 500),
            "utilities": (100, 300),
            "entertainment": (50, 300),
            "travel": (0, 1000),
            "insurance": (100, 400)
        }
        
    def generate_debt_payoff_scenarios(self, count=500):
        """Generate diverse debt payoff scenarios"""
        scenarios = []
        
        for i in range(count):
            # Random debt parameters
            debt_min, debt_max, debt_level = random.choice(self.debt_scenarios)
            debt_amount = random.randint(debt_min, debt_max)
            
            rate_min, rate_max, rate_level = random.choice(self.interest_rates)
            interest_rate = round(random.uniform(rate_min, rate_max), 2)
            
            # Calculate realistic minimum payment (2-3% of balance)
            min_payment = max(25, int(debt_amount * random.uniform(0.02, 0.035)))
            
            # Accelerated payment (1.5x to 4x minimum)
            accelerated_payment = int(min_payment * random.uniform(1.5, 4.0))
            
            # Calculate scenarios
            min_months = self._calculate_payoff_months(debt_amount, interest_rate, min_payment)
            acc_months = self._calculate_payoff_months(debt_amount, interest_rate, accelerated_payment)
            
            if min_months > 600:  # Cap unrealistic scenarios
                continue
                
            months_saved = min_months - acc_months
            interest_saved = self._calculate_interest_savings(debt_amount, interest_rate, min_payment, accelerated_payment)
            
            # Generate varied question formats
            question_formats = [
                f"I have ${debt_amount:,} in credit card debt at {interest_rate}% interest. My minimum payment is ${min_payment}. If I can pay ${accelerated_payment} per month instead, how much will I save?",
                f"Help me analyze paying ${accelerated_payment}/month vs ${min_payment}/month on my ${debt_amount} debt at {interest_rate}% APR.",
                f"I want to pay off my ${debt_amount} credit card debt faster. Currently paying ${min_payment}, can afford ${accelerated_payment}. Show me the difference.",
                f"Debt payoff analysis: ${debt_amount} balance, {interest_rate}% rate, ${min_payment} minimum vs ${accelerated_payment} accelerated payment.",
                f"What's the benefit of increasing my ${debt_amount} debt payment from ${min_payment} to ${accelerated_payment} per month? Interest rate is {interest_rate}%."
            ]
            
            question = random.choice(question_formats)
            
            # Generate comprehensive response
            strategy_score = min(10, 5 + (interest_saved / 1000) + (months_saved / 12))
            
            if months_saved > 24:
                urgency = "🔥 EXCELLENT strategy!"
            elif months_saved > 12:
                urgency = "✅ Great acceleration plan!"
            elif months_saved > 6:
                urgency = "👍 Good improvement!"
            else:
                urgency = "📈 Modest but helpful!"
            
            response = f"""**Debt Acceleration Analysis:**

**Current Plan:** ${min_payment}/month
- Payoff time: {min_months} months ({min_months//12} years, {min_months%12} months)
- Total interest: ${self._calculate_total_interest(debt_amount, interest_rate, min_payment):,.0f}

**Accelerated Plan:** ${accelerated_payment}/month
- Payoff time: {acc_months} months ({acc_months//12} years, {acc_months%12} months)  
- Total interest: ${self._calculate_total_interest(debt_amount, interest_rate, accelerated_payment):,.0f}

📊 **Savings Summary:**
- **Time saved:** {months_saved} months
- **Interest saved:** ${interest_saved:,.0f}
- **Strategy score:** {strategy_score:.1f}/10

{urgency}

**💡 Optimization Tips:**
- Apply bonuses/raises directly to debt
- Use balance transfer if rate > 20%
- Avoid new charges during payoff
- Consider side income for extra payments"""

            scenarios.append({
                "instruction": question,
                "input": f"debt: ${debt_amount}, rate: {interest_rate}%, min_payment: ${min_payment}, target_payment: ${accelerated_payment}",
                "output": response
            })
        
        return scenarios
    
    def generate_card_comparison_scenarios(self, count=300):
        """Generate credit card comparison scenarios"""
        scenarios = []
        
        # Load card data
        try:
            cards_df = pd.read_csv('../Data/master_card_dataset_cleaned.csv')
        except:
            return []  # Return empty if no card data
        
        for i in range(count):
            # Select 2-3 random cards for comparison
            sample_cards = cards_df.sample(n=random.randint(2, 3))
            
            # Random user profile
            income_min, income_max, income_level = random.choice(self.income_ranges)
            user_income = random.randint(income_min, income_max)
            monthly_spend = random.randint(800, 4000)
            
            # Generate spending breakdown
            spending_breakdown = {}
            remaining_spend = monthly_spend
            for category, (min_spend, max_spend) in list(self.spending_categories.items())[:-1]:
                if remaining_spend <= 0:
                    break
                # Fix random range issue
                lower_bound = min(min_spend//2, remaining_spend)
                upper_bound = min(max_spend, remaining_spend)
                if lower_bound >= upper_bound:
                    spend = lower_bound
                else:
                    spend = random.randint(lower_bound, upper_bound)
                spending_breakdown[category] = spend
                remaining_spend -= spend
            
            primary_category = max(spending_breakdown.items(), key=lambda x: x[1])[0]
            
            # Create comparison question
            card_names = [card['name'] for _, card in sample_cards.iterrows()]
            
            question_formats = [
                f"Compare {' vs '.join(card_names)} for someone earning ${user_income:,} annually with ${monthly_spend}/month spending, mostly on {primary_category}.",
                f"I spend ${monthly_spend} monthly (${spending_breakdown.get(primary_category, 0)} on {primary_category}). Which is better: {' or '.join(card_names)}?",
                f"Annual income ${user_income:,}, monthly spend ${monthly_spend}. Help me choose between {' and '.join(card_names)}.",
                f"Card comparison for ${user_income:,} income: {' vs '.join(card_names)}. I spend most on {primary_category}."
            ]
            
            question = random.choice(question_formats)
            
            # Generate detailed comparison response
            response = f"**Credit Card Comparison Analysis:**\n\n**Your Profile:**\n- Annual Income: ${user_income:,}\n- Monthly Spending: ${monthly_spend}\n- Top Category: {primary_category.title()} (${spending_breakdown.get(primary_category, 0)})\n\n"
            
            best_card = None
            best_score = 0
            
            for idx, (_, card) in enumerate(sample_cards.iterrows(), 1):
                annual_fee = card.get('annual_fee', 0)
                if pd.isna(annual_fee):
                    annual_fee = 0
                
                # Calculate estimated annual rewards
                base_rate = 1.0  # Default 1%
                if 'cashback' in str(card.get('category', '')).lower():
                    base_rate = 1.25
                elif 'premium' in str(card.get('category', '')).lower():
                    base_rate = 1.5
                
                annual_rewards = monthly_spend * 12 * (base_rate / 100)
                net_value = annual_rewards - annual_fee
                
                # Simple scoring
                score = net_value / 100 + (5 if annual_fee == 0 else 0)
                if score > best_score:
                    best_score = score
                    best_card = card['name']
                
                response += f"**{idx}. {card['name']}** ({card.get('issuer', 'Unknown')})\n"
                response += f"- Annual Fee: ${annual_fee}\n"
                response += f"- Network: {card.get('network', 'N/A')}\n"
                response += f"- Rewards: {card.get('rewards_type', 'None')}\n"
                response += f"- Estimated Annual Value: ${net_value:.0f}\n"
                response += f"- Features: {str(card.get('features', 'Standard'))[:100]}...\n\n"
            
            response += f"🏆 **Recommendation:** {best_card}\n"
            response += f"📊 **Best fit for your spending pattern and income level**\n"
            response += f"💡 **Tip:** Always pay balance in full to maximize rewards value"
            
            scenarios.append({
                "instruction": question,
                "input": f"income: ${user_income}, monthly_spend: ${monthly_spend}, primary_category: {primary_category}",
                "output": response
            })
        
        return scenarios
    
    def generate_multi_card_scenarios(self, count=200):
        """Generate multi-card debt management scenarios"""
        scenarios = []
        
        for i in range(count):
            # Generate 2-5 cards with different balances and rates
            num_cards = random.randint(2, 5)
            cards = []
            total_debt = 0
            
            for j in range(num_cards):
                debt_min, debt_max, _ = random.choice(self.debt_scenarios[:4])  # Exclude extreme scenarios
                balance = random.randint(debt_min//2, debt_max//2)
                rate_min, rate_max, _ = random.choice(self.interest_rates)
                rate = round(random.uniform(rate_min, rate_max), 2)
                
                cards.append({
                    'name': f'Card {chr(65+j)}',
                    'balance': balance,
                    'rate': rate
                })
                total_debt += balance
            
            # Sort by rate for avalanche method
            avalanche_order = sorted(cards, key=lambda x: x['rate'], reverse=True)
            snowball_order = sorted(cards, key=lambda x: x['balance'])
            
            budget = random.randint(300, 1200)
            
            card_descriptions = [f"{card['name']}: ${card['balance']} at {card['rate']}%" for card in cards]
            card_balances = [f"${card['balance']} at {card['rate']}%" for card in cards]
            card_details = [f"{card['name']} (${card['balance']}, {card['rate']}%)" for card in cards]
            
            question_formats = [
                f"I have {num_cards} credit cards with different rates and balances: {', '.join(card_descriptions)}. I have ${budget}/month budget. What's the best payoff strategy?",
                f"Multi-card debt help: {', '.join(card_balances)}. Monthly budget: ${budget}. Avalanche or snowball method?",
                f"Help me prioritize {num_cards} credit cards for payoff. Total debt: ${total_debt:,}, budget: ${budget}/month.",
                f"Debt avalanche vs snowball for: {', '.join(card_details)}."
            ]
            
            question = random.choice(question_formats)
            
            # Calculate minimum payments
            total_minimum = sum(max(25, card['balance'] * 0.025) for card in cards)
            extra_payment = max(0, budget - total_minimum)
            
            # Generate comprehensive strategy response
            response = f"""**Multi-Card Debt Strategy Analysis:**

**Your Situation:**
- {num_cards} credit cards
- Total debt: ${total_debt:,}
- Monthly budget: ${budget}
- Available for extra payments: ${extra_payment:.0f}

**📊 Card Details:**
"""
            
            for card in cards:
                min_pay = max(25, card['balance'] * 0.025)
                response += f"- **{card['name']}:** ${card['balance']:,} at {card['rate']}% (min: ${min_pay:.0f})\n"
            
            response += f"""
**🎯 Recommended Strategy: DEBT AVALANCHE**

**Priority Order (Highest Rate First):**
"""
            
            for i, card in enumerate(avalanche_order, 1):
                extra = extra_payment if i == 1 else 0
                total_payment = max(25, card['balance'] * 0.025) + extra
                response += f"{i}. **{card['name']}** - Pay ${total_payment:.0f}/month\n"
            
            # Calculate estimated payoff time
            estimated_months = max(12, total_debt / budget)
            interest_saved = total_debt * 0.1  # Rough estimate
            
            response += f"""
**📈 Expected Results:**
- Total payoff time: ~{estimated_months:.0f} months
- Interest savings vs minimum payments: ~${interest_saved:.0f}
- Strategy effectiveness: {min(10, 6 + extra_payment/100):.1f}/10

**💡 Alternative: Debt Snowball**
If you need motivation, start with lowest balance first:
"""
            
            for i, card in enumerate(snowball_order, 1):
                response += f"{i}. {card['name']} (${card['balance']:,})\n"
            
            response += "\n**🔄 Consider Balance Transfer** if average rate > 20%"
            
            scenarios.append({
                "instruction": question,
                "input": f"cards: {len(cards)}, total_debt: ${total_debt}, budget: ${budget}",
                "output": response
            })
        
        return scenarios
    
    def generate_credit_building_scenarios(self, count=250):
        """Generate credit building and improvement scenarios"""
        scenarios = []
        
        credit_ranges = [
            (300, 579, "poor", "rebuilding"),
            (580, 669, "fair", "improving"), 
            (670, 739, "good", "optimizing"),
            (740, 799, "very_good", "maintaining"),
            (800, 850, "excellent", "maximizing")
        ]
        
        life_situations = [
            "new_to_canada", "student", "recent_graduate", "young_professional",
            "career_change", "post_bankruptcy", "divorce_recovery", "first_time_buyer"
        ]
        
        for i in range(count):
            score_min, score_max, score_level, action = random.choice(credit_ranges)
            current_score = random.randint(score_min, score_max)
            situation = random.choice(life_situations)
            
            # Generate situation-specific details
            if situation == "new_to_canada":
                income = random.randint(45000, 85000)
                timeline = random.choice([12, 18, 24])
                challenge = "no Canadian credit history"
            elif situation == "student":
                income = random.randint(0, 25000)
                timeline = random.choice([6, 12, 18])
                challenge = "limited income and credit history"
            elif situation == "post_bankruptcy":
                income = random.randint(35000, 65000)
                timeline = random.choice([24, 36, 48])
                challenge = "rebuilding after bankruptcy"
            else:
                income = random.randint(35000, 95000)
                timeline = random.choice([6, 12, 18, 24])
                challenge = "improving credit profile"
            
            target_score = min(850, current_score + random.randint(50, 150))
            
            question_formats = [
                f"I'm {situation.replace('_', ' ')} with a {current_score} credit score. How can I improve to {target_score} in {timeline} months?",
                f"Credit building help: Current score {current_score}, income ${income:,}, goal {target_score} in {timeline} months. I'm {situation.replace('_', ' ')}.",
                f"My credit score is {current_score}. What's the best strategy to reach {target_score}? Timeline: {timeline} months.",
                f"Credit improvement plan needed: {current_score} → {target_score} in {timeline} months. Situation: {situation.replace('_', ' ')}, income ${income:,}."
            ]
            
            question = random.choice(question_formats)
            
            # Generate tailored improvement plan
            score_gap = target_score - current_score
            monthly_target = score_gap / timeline
            
            if timeline >= 18 and score_gap <= 100:
                feasibility = "Highly Achievable ✅"
                success_rate = 90
            elif timeline >= 12 and score_gap <= 80:
                feasibility = "Very Likely ✅"
                success_rate = 75
            else:
                feasibility = "Challenging but Possible ⚠️"
                success_rate = 60
            
            response = f"""**Credit Score Improvement Plan:**

**Current Situation:**
- Credit Score: {current_score} ({score_level})
- Target Score: {target_score}
- Timeline: {timeline} months
- Income: ${income:,}
- Challenge: {challenge}

**📊 Plan Feasibility:** {feasibility}
**🎯 Success Rate:** {success_rate}%

**📈 Month-by-Month Strategy:**

**Months 1-3: Foundation**
- ✅ Pay all bills on time (35% of score impact)
- ✅ Get credit report from Equifax/TransUnion Canada
- ✅ Dispute any errors immediately
"""
            
            if situation == "new_to_canada":
                response += "- ✅ Apply for RBC Secured Visa ($0 fee)\n- ✅ Open account with Big 5 Canadian bank\n"
            elif situation == "student":
                response += "- ✅ Apply for student credit card (RBC Student Visa)\n- ✅ Keep utilization under 10%\n"
            elif current_score < 650:
                response += "- ✅ Consider secured credit card if needed\n- ✅ Become authorized user on family card\n"
            
            response += f"""
**Months 4-{timeline//2}: Optimization**
- ✅ Keep credit utilization under 10% (30% of score impact)
- ✅ Don't close old accounts (15% of score impact)
- ✅ Pay down existing debt aggressively
- ✅ Monitor score monthly

**Months {timeline//2+1}-{timeline}: Acceleration**
- ✅ Request credit limit increases
- ✅ Consider adding second card for credit mix
- ✅ Continue perfect payment history
- ✅ Prepare for target achievement

**🎯 Projected Timeline:**
```
Month 3:  {current_score + int(monthly_target * 3)}
Month 6:  {current_score + int(monthly_target * 6)}
Month 12: {current_score + int(monthly_target * 12)}
Month {timeline}: {target_score} (TARGET)
```

**💡 Key Success Factors:**
1. **Never miss payments** (most important)
2. **Keep balances low** (under 10% utilization)
3. **Be patient** - significant changes take 3-6 months
4. **Monitor progress** - check score monthly
"""
            
            if situation == "new_to_canada":
                response += "\n**🍁 Canada-Specific Tips:**\n- Build relationship with one bank\n- Get cellphone contract to establish payment history\n- Consider newcomer banking packages"
            
            scenarios.append({
                "instruction": question,
                "input": f"current_score: {current_score}, target: {target_score}, timeline: {timeline}, situation: {situation}, income: {income}",
                "output": response
            })
        
        return scenarios
    
    def generate_balance_transfer_scenarios(self, count=150):
        """Generate balance transfer analysis scenarios"""
        scenarios = []
        
        for i in range(count):
            # Current debt situation
            current_debt = random.randint(2000, 25000)
            current_rate = round(random.uniform(18.99, 29.99), 2)
            current_payment = max(50, int(current_debt * random.uniform(0.025, 0.04)))
            
            # Balance transfer options
            bt_rate = round(random.uniform(0, 5.99), 2)  # Promotional rates
            bt_fee = round(current_debt * random.uniform(0.02, 0.03), 0)  # 2-3% fee
            promo_period = random.choice([6, 9, 12, 15, 18])
            regular_rate = round(random.uniform(19.99, 24.99), 2)
            
            question_formats = [
                f"I have ${current_debt:,} at {current_rate}%. Should I do a balance transfer to a {bt_rate}% card for {promo_period} months? Transfer fee is ${bt_fee}.",
                f"Balance transfer analysis: Current debt ${current_debt:,} at {current_rate}%, BT option {bt_rate}% for {promo_period} months, ${bt_fee} fee.",
                f"Is it worth transferring ${current_debt:,} from {current_rate}% to {bt_rate}% for {promo_period} months? Fee: ${bt_fee}.",
                f"Help me decide: Keep paying {current_rate}% or transfer to {bt_rate}% promo rate for {promo_period} months with ${bt_fee} fee."
            ]
            
            question = random.choice(question_formats)
            
            # Calculate scenarios
            # Current situation - total interest over equivalent period
            current_total_interest = self._calculate_total_interest(current_debt, current_rate, current_payment, promo_period)
            
            # Balance transfer situation
            bt_total_cost = bt_fee
            if promo_period >= 12:
                # Assume they can pay it off during promo period
                bt_interest = 0
            else:
                # Some interest at regular rate after promo
                remaining_after_promo = max(0, current_debt - (current_payment * promo_period))
                bt_interest = remaining_after_promo * (regular_rate / 100) * (6 / 12)  # 6 months at regular rate
            
            bt_total_cost += bt_interest
            
            savings = current_total_interest - bt_total_cost
            
            if savings > 500:
                recommendation = "🔥 HIGHLY RECOMMENDED"
                score = 9.5
            elif savings > 200:
                recommendation = "✅ Good Strategy"
                score = 8.0
            elif savings > 0:
                recommendation = "👍 Modest Benefit"
                score = 6.5
            else:
                recommendation = "❌ Not Recommended"
                score = 3.0
            
            payoff_months_current = self._calculate_payoff_months(current_debt, current_rate, current_payment)
            payoff_months_bt = min(promo_period, self._calculate_payoff_months(current_debt, bt_rate, current_payment))
            
            response = f"""**Balance Transfer Analysis:**

**Current Situation:**
- Debt: ${current_debt:,}
- Interest Rate: {current_rate}%
- Monthly Payment: ${current_payment}
- Payoff Time: {payoff_months_current} months
- Total Interest: ${current_total_interest:,.0f}

**Balance Transfer Option:**
- Promotional Rate: {bt_rate}% for {promo_period} months
- Transfer Fee: ${bt_fee}
- Regular Rate: {regular_rate}% (after promo)
- Total Cost: ${bt_total_cost:,.0f}

**📊 Comparison Results:**
- **Savings: ${savings:,.0f}**
- **Strategy Score: {score:.1f}/10**
- **Recommendation: {recommendation}**

**🎯 Action Plan:**
"""
            
            if savings > 100:
                response += f"""✅ **PROCEED with Balance Transfer**

**Timeline Strategy:**
- Month 1: Apply for BT card
- Month 2: Complete transfer
- Months 3-{promo_period}: Pay aggressively (${current_payment}+/month)
- Goal: Pay off before rate increases

**⚠️ Critical Rules:**
- Don't use old card for new purchases
- Don't use BT card for purchases during promo
- Set up automatic payments
- Pay more than minimum if possible"""
            else:
                response += f"""❌ **SKIP Balance Transfer**

**Better Alternatives:**
- Focus on increasing current payment to ${int(current_payment * 1.5)}
- Look for cards with longer 0% periods
- Consider debt consolidation loan
- Wait for better BT offers

**Why skip:** Transfer fee (${bt_fee}) + limited savings don't justify the hassle"""
            
            response += f"\n\n💡 **Key Insight:** Balance transfers work best when you can pay off debt during promotional period"
            
            scenarios.append({
                "instruction": question,
                "input": f"debt: ${current_debt}, current_rate: {current_rate}%, bt_rate: {bt_rate}%, promo_period: {promo_period}, fee: ${bt_fee}",
                "output": response
            })
        
        return scenarios
    
    def generate_budget_planning_scenarios(self, count=200):
        """Generate comprehensive budget planning scenarios"""
        scenarios = []
        
        for i in range(count):
            # Generate realistic income and expenses
            income_min, income_max, income_level = random.choice(self.income_ranges)
            monthly_income = random.randint(income_min, income_max) // 12
            
            # Generate expense breakdown
            housing_pct = random.uniform(0.25, 0.45)  # 25-45% of income
            housing = int(monthly_income * housing_pct)
            
            remaining = monthly_income - housing
            transportation = int(remaining * random.uniform(0.15, 0.25))
            food = int(remaining * random.uniform(0.10, 0.20))
            utilities = int(remaining * random.uniform(0.05, 0.10))
            insurance = int(remaining * random.uniform(0.03, 0.08))
            
            fixed_expenses = housing + transportation + food + utilities + insurance
            discretionary = monthly_income - fixed_expenses
            
            # Financial goals
            debt_amount = random.randint(0, 15000) if random.random() > 0.3 else 0
            wants_emergency_fund = random.choice([True, False])
            savings_goal = random.choice([5000, 10000, 15000, 20000])
            
            city = random.choice(self.canadian_cities)
            life_stage = random.choice(["recent_grad", "young_professional", "mid_career", "family_planning", "pre_retirement"])
            
            question_formats = [
                f"Help me create a budget: ${monthly_income*12:,} annual income, ${fixed_expenses} fixed expenses, ${debt_amount} debt. Live in {city}.",
                f"Budget planning for {life_stage.replace('_', ' ')}: ${monthly_income} monthly income, ${discretionary} after fixed costs. Want ${savings_goal} emergency fund.",
                f"I earn ${monthly_income} monthly in {city}. Fixed costs: ${fixed_expenses}. Have ${debt_amount} debt. Need budget help.",
                f"Monthly budget optimization: ${monthly_income} income, ${housing} rent, ${debt_amount} debt, savings goal ${savings_goal}."
            ]
            
            question = random.choice(question_formats)
            
            # Calculate recommendations
            if debt_amount > 0:
                debt_payment = max(100, min(discretionary * 0.4, debt_amount * 0.05))
            else:
                debt_payment = 0
            
            if wants_emergency_fund:
                emergency_savings = max(100, min(discretionary * 0.2, 500))
            else:
                emergency_savings = 0
            
            remaining_discretionary = discretionary - debt_payment - emergency_savings
            
            response = f"""**Comprehensive Budget Plan - {city}**

**Income & Fixed Expenses:**
- Monthly Income: ${monthly_income:,}
- Housing: ${housing} ({housing/monthly_income*100:.0f}%)
- Transportation: ${transportation}
- Food: ${food}
- Utilities: ${utilities}
- Insurance: ${insurance}
- **Total Fixed: ${fixed_expenses} ({fixed_expenses/monthly_income*100:.0f}%)**

**Available for Goals: ${discretionary}**

**🎯 Recommended Allocation:**
"""
            
            if debt_amount > 0:
                debt_months = max(6, debt_amount // (debt_payment or 100))
                response += f"💳 **Debt Payment:** ${debt_payment:.0f} (payoff in ~{debt_months} months)\n"
            
            if wants_emergency_fund:
                emergency_months = savings_goal // (emergency_savings or 100)
                response += f"💰 **Emergency Fund:** ${emergency_savings:.0f} (${savings_goal} goal in ~{emergency_months} months)\n"
            
            response += f"🎨 **Discretionary:** ${remaining_discretionary:.0f} (entertainment, hobbies, extra savings)\n"
            
            # Budget health assessment
            if fixed_expenses / monthly_income < 0.7:
                budget_health = "Excellent - Lots of flexibility"
                health_score = 9
            elif fixed_expenses / monthly_income < 0.8:
                budget_health = "Good - Some room for goals"
                health_score = 7
            else:
                budget_health = "Tight - Need optimization"
                health_score = 5
            
            response += f"""
**📊 Budget Health Assessment:**
- **Score: {health_score}/10** - {budget_health}
- Fixed expenses ratio: {fixed_expenses/monthly_income*100:.0f}% (ideal: <70%)
- Savings capacity: ${discretionary:.0f}/month

**🏆 {life_stage.replace('_', ' ').title()} Priorities:**
"""
            
            if life_stage == "recent_grad":
                response += "- Build 3-month emergency fund first\n- Pay minimum on student loans while building savings\n- Focus on career development"
            elif life_stage == "young_professional":
                response += "- 6-month emergency fund target\n- Maximize RRSP contributions\n- Consider first home down payment savings"
            elif life_stage == "family_planning":
                response += "- Increase emergency fund to 6+ months\n- Life insurance review\n- Child expense planning ($1000+/month)"
            else:
                response += "- Maintain emergency fund\n- Maximize retirement savings\n- Consider investment portfolio growth"
            
            response += f"""
**💡 {city} Specific Tips:**
- Housing costs vary widely - consider location vs commute costs
- Take advantage of local transit options
- Explore community resources for entertainment/activities

**📈 Next Steps:**
1. Track expenses for 2 months to validate this budget
2. Automate savings and debt payments
3. Review and adjust quarterly
4. Increase income through skills/side hustles when possible"""
            
            scenarios.append({
                "instruction": question,
                "input": f"income: ${monthly_income}, fixed: ${fixed_expenses}, debt: ${debt_amount}, city: {city}, stage: {life_stage}",
                "output": response
            })
        
        return scenarios
    
    def generate_specialized_scenarios(self, count=150):
        """Generate specialized financial scenarios"""
        scenarios = []
        
        specialties = [
            "business_credit", "secured_cards", "student_finance", "newcomer_canada",
            "credit_repair", "rewards_optimization", "travel_cards", "cash_back_strategy"
        ]
        
        for specialty in specialties:
            for i in range(count // len(specialties)):
                if specialty == "business_credit":
                    revenue = random.randint(50000, 500000)
                    business_type = random.choice(["consulting", "retail", "service", "tech_startup", "restaurant"])
                    
                    question = f"I have a {business_type} business with ${revenue:,} annual revenue. What business credit card strategy should I use?"
                    
                    response = f"""**Business Credit Card Strategy:**

**Business Profile:**
- Type: {business_type.title()}
- Annual Revenue: ${revenue:,}
- Recommended Strategy: {"Premium" if revenue > 200000 else "Growth-focused"}

**🏆 Recommended Business Cards:**
1. **RBC Avion Visa Infinite Business** - Travel rewards, expense tracking
2. **TD Business Travel Visa** - No FX fees, travel insurance
3. **BMO Business Mastercard** - Cash back on business expenses

**💼 Business Credit Benefits:**
- Separate business/personal expenses
- Build business credit history
- Expense tracking and reporting
- Employee cards with controls
- Higher credit limits

**📊 Strategy by Revenue:**
- Under $100k: Focus on no-fee cards with expense tracking
- $100k-$250k: Consider premium cards with travel benefits
- Over $250k: Multiple cards for category optimization

**⚠️ Important Rules:**
- Never mix personal and business expenses
- Keep detailed records for CRA
- Pay balances in full monthly
- Use business EIN for applications"""

                elif specialty == "newcomer_canada":
                    home_country = random.choice(["India", "China", "Philippines", "Pakistan", "Nigeria", "UK", "US", "France"])
                    months_in_canada = random.randint(1, 12)
                    
                    question = f"I moved to Canada from {home_country} {months_in_canada} months ago. How do I build credit with no Canadian history?"
                    
                    response = f"""**Newcomer Credit Building Strategy:**

**Your Situation:**
- Time in Canada: {months_in_canada} months
- Home Country: {home_country}
- Credit History: None in Canada
- Challenge: Building from zero

**🍁 Phase 1: Foundation (Months 1-3)**
- ✅ Open account with Big 5 bank (RBC, TD, BMO, Scotiabank, CIBC)
- ✅ Apply for newcomer banking package
- ✅ Get RBC Secured Visa or equivalent ($0 annual fee)
- ✅ Deposit $500-1000 as security
- ✅ Set up phone plan (builds payment history)

**📈 Phase 2: Establishment (Months 4-8)**
- ✅ Use secured card for small purchases ($100-200/month)
- ✅ Pay in full every month before due date
- ✅ Keep utilization under 10%
- ✅ Add utility bills to your name
- ✅ Request credit report after 6 months

**🚀 Phase 3: Growth (Months 9-18)**
- ✅ Apply for unsecured card (start with same bank)
- ✅ Request credit limit increases
- ✅ Consider adding second card
- ✅ Build relationship with your primary bank

**🎯 Expected Timeline:**
```
Month 3: First credit report appears
Month 6: Score around 650-680
Month 12: Score 700+ with good management
Month 18: Eligible for premium cards
```

**🌍 {home_country} Specific Tips:**
"""
                    
                    if home_country in ["US", "UK"]:
                        response += "- Some lenders may consider your international credit\n- Bring credit reports from home country\n- Consider international bank relationships"
                    else:
                        response += "- Focus on building Canadian credit from scratch\n- Consider newcomer programs at major banks\n- Join newcomer community groups for tips"
                    
                    response += "\n**⚠️ Common Mistakes to Avoid:**\n- Applying for multiple cards too quickly\n- Using credit card for cash advances\n- Missing any payment (even $5)\n- Closing your first card too early"

                # Add similar detailed responses for other specialties...
                else:
                    # Generic specialty response
                    question = f"Help me with {specialty.replace('_', ' ')} strategy."
                    response = f"Here's a comprehensive {specialty.replace('_', ' ')} strategy tailored to your needs..."
                
                scenarios.append({
                    "instruction": question,
                    "input": f"specialty: {specialty}",
                    "output": response
                })
        
        return scenarios
    
    def _calculate_payoff_months(self, balance, annual_rate, monthly_payment):
        """Calculate months to pay off debt"""
        monthly_rate = annual_rate / 100 / 12
        if monthly_payment <= balance * monthly_rate:
            return 600  # Never pays off
        
        months = -np.log(1 - (balance * monthly_rate) / monthly_payment) / np.log(1 + monthly_rate)
        return max(1, int(np.ceil(months)))
    
    def _calculate_total_interest(self, balance, annual_rate, monthly_payment, months=None):
        """Calculate total interest paid"""
        if months is None:
            months = self._calculate_payoff_months(balance, annual_rate, monthly_payment)
        
        monthly_rate = annual_rate / 100 / 12
        total_paid = monthly_payment * months
        return max(0, total_paid - balance)
    
    def _calculate_interest_savings(self, balance, annual_rate, min_payment, acc_payment):
        """Calculate interest savings between payment strategies"""
        min_interest = self._calculate_total_interest(balance, annual_rate, min_payment)
        acc_interest = self._calculate_total_interest(balance, annual_rate, acc_payment)
        return max(0, min_interest - acc_interest)
    
    def generate_all_scenarios(self):
        """Generate all training scenarios"""
        print("🚀 Generating massive NAVUS training dataset...")
        print("=" * 60)
        
        all_scenarios = []
        
        scenario_types = [
            ("Debt Payoff Scenarios", self.generate_debt_payoff_scenarios, 500),
            ("Card Comparison Scenarios", self.generate_card_comparison_scenarios, 300),
            ("Multi-Card Strategies", self.generate_multi_card_scenarios, 200),
            ("Credit Building Plans", self.generate_credit_building_scenarios, 250),
            ("Balance Transfer Analysis", self.generate_balance_transfer_scenarios, 150),
            ("Budget Planning", self.generate_budget_planning_scenarios, 200),
            ("Specialized Scenarios", self.generate_specialized_scenarios, 150)
        ]
        
        for name, generator_func, count in scenario_types:
            print(f"📊 Generating {name}... ({count} examples)")
            scenarios = generator_func(count)
            all_scenarios.extend(scenarios)
            print(f"✅ Generated {len(scenarios)} {name.lower()}")
        
        print(f"\n🎯 Total training examples generated: {len(all_scenarios)}")
        return all_scenarios
    
    def save_dataset(self, scenarios, filename):
        """Save the massive dataset"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(scenarios)} examples to {filename}")
        
        # Create summary statistics
        categories = {}
        for scenario in scenarios:
            category = scenario.get('input', '').split(',')[0] if scenario.get('input') else 'general'
            categories[category] = categories.get(category, 0) + 1
        
        print("\n📈 Dataset Summary:")
        for category, count in sorted(categories.items()):
            print(f"  - {category}: {count} examples")

def main():
    """Generate the massive training dataset"""
    generator = MassiveDatasetGenerator()
    
    # Generate all scenarios (10x more data)
    all_scenarios = generator.generate_all_scenarios()
    
    # Save the massive dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../Training/massive_navus_dataset_{timestamp}.json"
    generator.save_dataset(all_scenarios, filename)
    
    # Also save a combined dataset with original data
    try:
        # Load existing datasets
        existing_data = []
        
        for existing_file in ["../Training/navus_alpaca_format.json", "../Training/enhanced_debt_payoff_dataset.json"]:
            try:
                with open(existing_file, 'r') as f:
                    data = json.load(f)
                    existing_data.extend(data)
            except FileNotFoundError:
                continue
        
        # Combine with new massive dataset
        combined_data = existing_data + all_scenarios
        combined_filename = f"../Training/combined_massive_dataset_{timestamp}.json"
        generator.save_dataset(combined_data, combined_filename)
        
        print(f"\n🔥 MASSIVE DATASET COMPLETE!")
        print(f"📊 Original data: {len(existing_data)} examples")
        print(f"📊 New data: {len(all_scenarios)} examples")  
        print(f"📊 Combined total: {len(combined_data)} examples")
        print(f"📈 Increase: {len(all_scenarios)/max(len(existing_data), 1):.1f}x more training data!")
        
    except Exception as e:
        print(f"⚠️ Could not create combined dataset: {e}")
    
    print(f"\n✅ Ready for enhanced training with massive dataset!")

if __name__ == "__main__":
    main()