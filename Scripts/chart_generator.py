"""
NAVUS Chart Generator
Generates financial charts and visualizations to support LLM responses
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO

class NAVUSChartGenerator:
    """Generate charts for financial analysis and debt payoff scenarios"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_debt_payoff_comparison(self, debt_amount, interest_rate, min_payment, accelerated_payment):
        """Generate comparison chart of minimum vs accelerated debt payoff"""
        
        # Calculate payoff schedules
        min_schedule = self._calculate_payoff_schedule(debt_amount, interest_rate, min_payment)
        acc_schedule = self._calculate_payoff_schedule(debt_amount, interest_rate, accelerated_payment)
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Debt balance over time
        ax1.plot(range(len(min_schedule)), min_schedule, 'r-', linewidth=2, label=f'Minimum (${min_payment}/mo)')
        ax1.plot(range(len(acc_schedule)), acc_schedule, 'g-', linewidth=2, label=f'Accelerated (${accelerated_payment}/mo)')
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Remaining Debt ($)')
        ax1.set_title('Debt Payoff Timeline Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total cost comparison (bar chart)
        min_total = debt_amount + (len(min_schedule) * min_payment - debt_amount)
        acc_total = debt_amount + (len(acc_schedule) * accelerated_payment - debt_amount)
        
        categories = ['Minimum Payment', 'Accelerated Payment']
        totals = [min_total, acc_total]
        colors = ['#ff6b6b', '#51cf66']
        
        bars = ax2.bar(categories, totals, color=colors, alpha=0.8)
        ax2.set_ylabel('Total Cost ($)')
        ax2.set_title('Total Cost Comparison')
        
        # Add value labels on bars
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${total:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_multi_card_strategy_chart(self, cards_data):
        """Generate chart showing optimal multi-card payoff strategy"""
        
        # cards_data = [{'name': 'Card A', 'balance': 3000, 'rate': 24.99}, ...]
        df = pd.DataFrame(cards_data)
        df = df.sort_values('rate', ascending=False)  # Sort by interest rate (avalanche method)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Interest rate vs balance scatter plot
        scatter = ax1.scatter(df['balance'], df['rate'], s=200, alpha=0.7, c=range(len(df)), cmap='RdYlGn_r')
        
        for i, card in df.iterrows():
            ax1.annotate(card['name'], (card['balance'], card['rate']), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax1.set_xlabel('Balance ($)')
        ax1.set_ylabel('Interest Rate (%)')
        ax1.set_title('Credit Cards: Balance vs Interest Rate')
        ax1.grid(True, alpha=0.3)
        
        # Priority order (horizontal bar chart)
        y_pos = np.arange(len(df))
        colors = ['#d73527', '#ff9800', '#4caf50'][:len(df)]
        
        bars = ax2.barh(y_pos, df['balance'], color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{row['name']} ({row['rate']}%)" for _, row in df.iterrows()])
        ax2.set_xlabel('Balance ($)')
        ax2.set_title('Payoff Priority Order (Highest Rate First)')
        
        # Add priority labels
        for i, (bar, balance) in enumerate(zip(bars, df['balance'])):
            width = bar.get_width()
            ax2.text(width/2, bar.get_y() + bar.get_height()/2,
                    f'#{i+1}', ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_credit_score_timeline(self, current_score, timeline_months=24):
        """Generate projected credit score improvement timeline"""
        
        # Simulate credit score improvement over time
        months = np.arange(0, timeline_months + 1)
        
        # Base improvement curve (diminishing returns)
        base_improvement = 120 * (1 - np.exp(-months / 8))  # Exponential curve
        projected_scores = current_score + base_improvement
        
        # Cap at 850 (max FICO score)
        projected_scores = np.minimum(projected_scores, 850)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Main score line
        ax.plot(months, projected_scores, 'b-', linewidth=3, label='Projected Score')
        ax.fill_between(months, current_score, projected_scores, alpha=0.3, color='blue')
        
        # Add milestone markers
        milestones = [(6, 'Payment History Impact'), (12, 'Lower Utilization'), (18, 'Credit Mix Benefit'), (24, 'Time Factor')]
        for month, label in milestones:
            if month <= timeline_months:
                score_at_month = projected_scores[month]
                ax.annotate(f'{score_at_month:.0f}\n{label}', 
                           xy=(month, score_at_month), xytext=(10, 10),
                           textcoords='offset points', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Months')
        ax.set_ylabel('Credit Score')
        ax.set_title('Projected Credit Score Improvement Timeline')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add score ranges background
        ax.axhspan(300, 579, alpha=0.1, color='red', label='Poor')
        ax.axhspan(580, 669, alpha=0.1, color='orange', label='Fair')
        ax.axhspan(670, 739, alpha=0.1, color='yellow', label='Good')
        ax.axhspan(740, 799, alpha=0.1, color='lightgreen', label='Very Good')
        ax.axhspan(800, 850, alpha=0.1, color='green', label='Excellent')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_strategy_scoring_chart(self, strategies_data):
        """Generate radar chart comparing different debt strategies"""
        
        categories = ['Interest Savings', 'Time Efficiency', 'Feasibility', 'Credit Impact', 'Risk Level']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the circle
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        
        for i, (strategy_name, scores) in enumerate(strategies_data.items()):
            scores = scores + [scores[0]]  # Close the circle
            ax.plot(angles, scores, 'o-', linewidth=2, label=strategy_name, color=colors[i % len(colors)])
            ax.fill(angles, scores, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Debt Strategy Comparison (NAVUS Scoring System)', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _calculate_payoff_schedule(self, balance, annual_rate, monthly_payment):
        """Calculate month-by-month debt payoff schedule"""
        monthly_rate = annual_rate / 100 / 12
        schedule = []
        current_balance = balance
        
        while current_balance > 0:
            interest_charge = current_balance * monthly_rate
            principal_payment = min(monthly_payment - interest_charge, current_balance)
            current_balance -= principal_payment
            schedule.append(max(0, current_balance))
            
            # Prevent infinite loop
            if len(schedule) > 600:  # 50 years max
                break
                
        return schedule
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64

# Example usage and testing
if __name__ == "__main__":
    generator = NAVUSChartGenerator()
    
    # Test debt payoff comparison
    print("Generating debt payoff comparison chart...")
    chart1 = generator.generate_debt_payoff_comparison(5000, 19.99, 125, 300)
    print(f"Chart 1 generated: {len(chart1)} characters")
    
    # Test multi-card strategy
    print("Generating multi-card strategy chart...")
    cards = [
        {'name': 'Card A', 'balance': 3000, 'rate': 24.99},
        {'name': 'Card B', 'balance': 2000, 'rate': 18.99},
        {'name': 'Card C', 'balance': 1500, 'rate': 21.99}
    ]
    chart2 = generator.generate_multi_card_strategy_chart(cards)
    print(f"Chart 2 generated: {len(chart2)} characters")
    
    # Test credit score timeline
    print("Generating credit score timeline...")
    chart3 = generator.generate_credit_score_timeline(580, 24)
    print(f"Chart 3 generated: {len(chart3)} characters")
    
    # Test strategy scoring
    print("Generating strategy scoring chart...")
    strategies = {
        'Balance Transfer': [9, 8, 7, 8, 9],
        'Debt Avalanche': [8, 9, 9, 7, 8],
        'Debt Snowball': [6, 7, 9, 8, 9],
        'Consolidation Loan': [7, 6, 8, 6, 7]
    }
    chart4 = generator.generate_strategy_scoring_chart(strategies)
    print(f"Chart 4 generated: {len(chart4)} characters")
    
    print("All charts generated successfully!")