#!/usr/bin/env python3
"""
Test chart generation functionality for NAVUS
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import numpy as np
import base64
import io

def create_debt_payoff_chart():
    """Test the debt payoff chart generation"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample debt payoff data for $5000 at 19% interest
    months = list(range(1, 25))
    # With minimum payment ($125/month)
    minimum_balance = [5000 * (1.19/12)**(i) - 125 * sum((1.19/12)**(j) for j in range(i)) for i in months]
    minimum_balance = [max(0, b) for b in minimum_balance]  # Don't go negative
    
    # With accelerated payment ($300/month)
    accelerated_balance = [5000 * (1.19/12)**(i) - 300 * sum((1.19/12)**(j) for j in range(i)) for i in months]
    accelerated_balance = [max(0, b) for b in accelerated_balance]  # Don't go negative
    
    # Interest saved calculation
    interest_saved = [(5000 - min_bal) - (5000 - acc_bal) for min_bal, acc_bal in zip(minimum_balance, accelerated_balance)]
    interest_saved = [max(0, i) for i in interest_saved]
    
    ax.plot(months, minimum_balance, label='Minimum Payment ($125/mo)', color='#FF6B6B', linewidth=3, marker='o')
    ax.plot(months, accelerated_balance, label='Accelerated Payment ($300/mo)', color='#4ECDC4', linewidth=3, marker='s')
    
    ax.set_title('$5,000 Credit Card Debt Payoff Strategy\n(19% Interest Rate)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Months', fontsize=14)
    ax.set_ylabel('Remaining Balance ($CAD)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Debt-Free!', xy=(18, 0), xytext=(20, 1000),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

def create_card_comparison_chart():
    """Test the card comparison chart generation"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cards = ['Amex Cobalt', 'RBC Avion', 'TD Cashback', 'Scotia Gold']
    rewards = [5.0, 1.25, 3.0, 5.0]
    fees = [0, 120, 139, 139]
    
    x = np.arange(len(cards))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rewards, width, label='Max Rewards Rate (%)', color='#4ECDC4')
    bars2 = ax.bar(x + width/2, [f/25 for f in fees], width, label='Annual Fee (÷25)', color='#FF6B6B')
    
    ax.set_title('Canadian Credit Card Comparison', fontsize=18, fontweight='bold')
    ax.set_xlabel('Credit Cards', fontsize=14)
    ax.set_ylabel('Rate/Fee Scale', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cards, rotation=45)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar, fee in zip(bars2, fees):
        height = bar.get_height()
        ax.annotate(f'${fee}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return chart_base64

if __name__ == "__main__":
    print("Testing chart generation...")
    
    # Test debt payoff chart
    print("1. Generating debt payoff chart...")
    debt_chart = create_debt_payoff_chart()
    print(f"   Debt chart generated: {len(debt_chart)} characters")
    
    # Test card comparison chart
    print("2. Generating card comparison chart...")
    card_chart = create_card_comparison_chart()
    print(f"   Card comparison chart generated: {len(card_chart)} characters")
    
    print("\nChart generation test completed successfully!")
    print(f"Sample debt chart base64 (first 100 chars): {debt_chart[:100]}")
    print(f"Sample card chart base64 (first 100 chars): {card_chart[:100]}")