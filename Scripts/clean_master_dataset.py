#!/usr/bin/env python3
"""
Master Dataset Cleaning Script
Fix data quality issues identified in validation for optimal LLM training.
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

class MasterDatasetCleaner:
    def __init__(self):
        self.input_file = "NAVUS/master_card_dataset.csv"
        self.output_file = "NAVUS/master_card_dataset_cleaned.csv"
        
    def load_data(self):
        """Load the master dataset."""
        print("📂 Loading master dataset...")
        try:
            self.df = pd.read_csv(self.input_file, encoding='utf-8')
            print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def clean_missing_values(self):
        """Clean missing values in critical columns."""
        print("\n🧹 CLEANING MISSING VALUES...")
        
        # For cards missing 'name', try to get from other name columns
        if 'name' in self.df.columns:
            missing_names = self.df['name'].isnull() | (self.df['name'] == '')
            
            # Try to fill from card_name column
            if 'card_name' in self.df.columns:
                self.df.loc[missing_names, 'name'] = self.df.loc[missing_names, 'card_name']
                filled = missing_names.sum() - (self.df['name'].isnull() | (self.df['name'] == '')).sum()
                if filled > 0:
                    print(f"   ✅ Filled {filled} missing names from card_name column")
            
            # For remaining missing names, use a placeholder that indicates data source
            still_missing = self.df['name'].isnull() | (self.df['name'] == '')
            if still_missing.sum() > 0:
                self.df.loc[still_missing, 'name'] = self.df.loc[still_missing, 'source_file'].apply(
                    lambda x: f"Card from {x.split('.')[0]}" if pd.notna(x) else "Unknown Card"
                )
                print(f"   ✅ Filled {still_missing.sum()} remaining names with placeholders")
        
        # For issuers, try to get from issuer_name column
        if 'issuer' in self.df.columns:
            missing_issuers = self.df['issuer'].isnull() | (self.df['issuer'] == '')
            
            if 'issuer_name' in self.df.columns:
                self.df.loc[missing_issuers, 'issuer'] = self.df.loc[missing_issuers, 'issuer_name']
                filled = missing_issuers.sum() - (self.df['issuer'].isnull() | (self.df['issuer'] == '')).sum()
                if filled > 0:
                    print(f"   ✅ Filled {filled} missing issuers from issuer_name column")
            
            # Fill remaining with 'Unknown'
            still_missing = self.df['issuer'].isnull() | (self.df['issuer'] == '')
            if still_missing.sum() > 0:
                self.df.loc[still_missing, 'issuer'] = 'Unknown Issuer'
                print(f"   ✅ Filled {still_missing.sum()} remaining issuers with 'Unknown Issuer'")
    
    def clean_numeric_columns(self):
        """Clean and standardize numeric columns."""
        print("\n🔢 CLEANING NUMERIC COLUMNS...")
        
        numeric_columns = {
            'annual_fee': 'Annual Fee',
            'purchase_rate': 'Purchase Interest Rate',
            'balance_transfer_rate': 'Balance Transfer Rate',
            'cash_advance_rate': 'Cash Advance Rate',
            'min_income': 'Minimum Income',
            'welcome_bonus': 'Welcome Bonus',
            'cashback_rate': 'Cashback Rate'
        }
        
        for col, description in numeric_columns.items():
            if col in self.df.columns:
                print(f"   🔧 Cleaning {description} ({col})...")
                
                original_non_null = self.df[col].notna().sum()
                
                # Handle JSON-like structures in min_income and welcome_bonus
                if col in ['min_income', 'welcome_bonus']:
                    self.df[col] = self.df[col].apply(self.extract_numeric_from_json)
                
                # Convert to numeric, keeping NaN for non-convertible values
                numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                
                # Replace the column with cleaned numeric values
                self.df[col] = numeric_series
                
                final_non_null = self.df[col].notna().sum()
                print(f"      ✅ Converted {final_non_null}/{original_non_null} values to numeric")
                
                # Apply business logic bounds
                if col == 'annual_fee':
                    # Cap at reasonable maximum
                    high_fees = (self.df[col] > 1000).sum()
                    if high_fees > 0:
                        print(f"      ℹ️  Found {high_fees} cards with fees >$1000 (keeping as-is)")
                
                elif col in ['purchase_rate', 'balance_transfer_rate', 'cash_advance_rate']:
                    # Ensure rates are reasonable (0-50%)
                    invalid_rates = ((self.df[col] < 0) | (self.df[col] > 50)).sum()
                    if invalid_rates > 0:
                        self.df.loc[(self.df[col] < 0) | (self.df[col] > 50), col] = np.nan
                        print(f"      🧹 Removed {invalid_rates} invalid rate values")
                
                elif col == 'min_income':
                    # Cap minimum income at reasonable range
                    if self.df[col].max() > 500000:
                        high_income = (self.df[col] > 500000).sum()
                        self.df.loc[self.df[col] > 500000, col] = 500000
                        print(f"      🧹 Capped {high_income} very high income requirements to $500,000")
    
    def extract_numeric_from_json(self, value):
        """Extract numeric value from JSON-like strings."""
        if pd.isna(value) or value == '':
            return np.nan
        
        # If already numeric, return as-is
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        
        # Try to parse as JSON
        try:
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    # Try common keys
                    for key in ['public', 'amount', 'minimum', 'value']:
                        if key in parsed and parsed[key] is not None:
                            return float(parsed[key])
                elif isinstance(parsed, (int, float)):
                    return float(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Try to extract first number from string
        if isinstance(value, str):
            numbers = re.findall(r'[\d,]+\.?\d*', str(value))
            if numbers:
                try:
                    # Remove commas and convert
                    return float(numbers[0].replace(',', ''))
                except ValueError:
                    pass
        
        return np.nan
    
    def standardize_issuers(self):
        """Standardize issuer names."""
        print("\n🏦 STANDARDIZING ISSUER NAMES...")
        
        if 'issuer' not in self.df.columns:
            return
        
        # Create standardization mapping
        issuer_mapping = {
            'td bank': 'TD',
            'td': 'TD',
            'toronto dominion bank': 'TD',
            'toronto-dominion bank': 'TD',
            'bank of montreal': 'BMO',
            'bmo': 'BMO',
            'royal bank of canada': 'RBC',
            'rbc': 'RBC',
            'canadian imperial bank of commerce': 'CIBC',
            'cibc': 'CIBC',
            'bank of nova scotia': 'Scotiabank',
            'scotiabank': 'Scotiabank',
            'scotia': 'Scotiabank',
            'american express': 'American Express',
            'amex': 'American Express',
            'capital one': 'Capital One',
            'national bank of canada': 'National Bank',
            'national bank': 'National Bank',
            'canadian tire financial services': 'Canadian Tire Financial',
            'neo financial': 'Neo Financial',
            'tangerine': 'Tangerine'
        }
        
        # Apply standardization
        original_issuers = self.df['issuer'].value_counts()
        
        self.df['issuer'] = self.df['issuer'].astype(str).str.lower().str.strip()
        self.df['issuer'] = self.df['issuer'].replace(issuer_mapping)
        
        # Capitalize properly
        self.df['issuer'] = self.df['issuer'].str.title()
        
        new_issuers = self.df['issuer'].value_counts()
        
        print(f"   ✅ Standardized issuer names: {len(original_issuers)} → {len(new_issuers)} unique issuers")
        print(f"   🏦 Top issuers after standardization:")
        for issuer, count in new_issuers.head(8).items():
            print(f"      {issuer}: {count} cards")
    
    def clean_text_fields(self):
        """Clean and standardize text fields."""
        print("\n📝 CLEANING TEXT FIELDS...")
        
        text_columns = ['name', 'category', 'rewards_type', 'network', 'features']
        
        for col in text_columns:
            if col in self.df.columns:
                print(f"   🧹 Cleaning {col}...")
                
                # Remove extra whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Replace empty strings with NaN
                self.df[col] = self.df[col].replace('', np.nan)
                self.df[col] = self.df[col].replace('nan', np.nan)
                
                # Standardize common values
                if col == 'network':
                    network_mapping = {
                        'visa': 'Visa',
                        'mastercard': 'Mastercard', 
                        'american express': 'American Express',
                        'american-express': 'American Express',
                        'amex': 'American Express'
                    }
                    self.df[col] = self.df[col].str.lower().replace(network_mapping)
                
                elif col == 'category':
                    # Standardize category names
                    category_mapping = {
                        'no-fee': 'no_fee',
                        'cash-back': 'cashback',
                        'cash back': 'cashback'
                    }
                    self.df[col] = self.df[col].str.lower().replace(category_mapping)
                
                non_null_after = self.df[col].notna().sum()
                print(f"      ✅ {non_null_after} non-null values after cleaning")
    
    def add_computed_features(self):
        """Add computed features useful for LLM training."""
        print("\n🔧 ADDING COMPUTED FEATURES...")
        
        # Fee category
        if 'annual_fee' in self.df.columns:
            self.df['fee_category'] = pd.cut(
                self.df['annual_fee'].fillna(0), 
                bins=[-1, 0, 99, 199, 999, float('inf')],
                labels=['no_fee', 'low_fee', 'medium_fee', 'high_fee', 'premium_fee']
            )
            print(f"   ✅ Added fee_category feature")
        
        # Rate category (using purchase rate as primary)
        if 'purchase_rate' in self.df.columns:
            self.df['rate_category'] = pd.cut(
                self.df['purchase_rate'],
                bins=[0, 18, 20, 22, 25, float('inf')],
                labels=['excellent_rate', 'good_rate', 'average_rate', 'high_rate', 'very_high_rate']
            )
            print(f"   ✅ Added rate_category feature")
        
        # Has welcome bonus
        if 'welcome_bonus' in self.df.columns:
            self.df['has_welcome_bonus'] = (self.df['welcome_bonus'] > 0).astype(bool)
            bonus_count = self.df['has_welcome_bonus'].sum()
            print(f"   ✅ Added has_welcome_bonus feature ({bonus_count} cards with bonuses)")
        
        # Feature count (from features text)
        if 'features' in self.df.columns:
            self.df['feature_count_computed'] = self.df['features'].fillna('').str.count('\\|') + 1
            self.df.loc[self.df['features'].isna(), 'feature_count_computed'] = 0
            avg_features = self.df['feature_count_computed'].mean()
            print(f"   ✅ Added feature_count_computed (avg: {avg_features:.1f} features per card)")
        
        # Premium card indicator
        premium_keywords = ['platinum', 'world elite', 'infinite', 'signature', 'premium']
        if 'name' in self.df.columns:
            self.df['is_premium'] = self.df['name'].str.lower().str.contains('|'.join(premium_keywords), na=False)
            premium_count = self.df['is_premium'].sum()
            print(f"   ✅ Added is_premium feature ({premium_count} premium cards)")
    
    def optimize_for_llm(self):
        """Optimize dataset structure for LLM training."""
        print("\n🤖 OPTIMIZING FOR LLM TRAINING...")
        
        # Reorder columns by importance for LLM training
        priority_columns = [
            'name', 'issuer', 'network', 'category', 'annual_fee', 'fee_category',
            'purchase_rate', 'rate_category', 'rewards_type', 'welcome_bonus', 'has_welcome_bonus',
            'features', 'feature_count_computed', 'is_premium', 'min_income'
        ]
        
        # Add remaining columns
        remaining_columns = [col for col in self.df.columns if col not in priority_columns]
        final_column_order = priority_columns + remaining_columns
        
        # Filter to only existing columns
        final_column_order = [col for col in final_column_order if col in self.df.columns]
        
        # Reorder
        self.df = self.df[final_column_order]
        print(f"   ✅ Reordered columns for optimal LLM training")
        
        # Create a summary text column for each card (useful for embeddings)
        summary_parts = []
        
        for _, row in self.df.iterrows():
            parts = []
            
            if pd.notna(row.get('name')):
                parts.append(f"Card: {row['name']}")
            
            if pd.notna(row.get('issuer')):
                parts.append(f"Issuer: {row['issuer']}")
            
            if pd.notna(row.get('category')):
                parts.append(f"Type: {row['category']}")
            
            if pd.notna(row.get('annual_fee')):
                fee = int(row['annual_fee']) if row['annual_fee'] == int(row['annual_fee']) else row['annual_fee']
                parts.append(f"Annual Fee: ${fee}")
            
            if pd.notna(row.get('rewards_type')):
                parts.append(f"Rewards: {row['rewards_type']}")
            
            if pd.notna(row.get('features')) and row['features'] != '':
                features = row['features'].replace(' | ', ', ')
                parts.append(f"Features: {features}")
            
            summary_parts.append('. '.join(parts))
        
        self.df['card_summary'] = summary_parts
        print(f"   ✅ Added card_summary column for embeddings")
    
    def save_cleaned_dataset(self):
        """Save the cleaned dataset."""
        print(f"\n💾 SAVING CLEANED DATASET...")
        
        try:
            # Save with proper formatting
            self.df.to_csv(self.output_file,
                          index=False,
                          encoding='utf-8',
                          na_rep='',
                          quoting=1,  # Quote all non-numeric fields
                          escapechar='\\')
            
            print(f"✅ Cleaned dataset saved: {self.output_file}")
            print(f"   📊 Final dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
            
            # Show data quality improvement
            print(f"\n📈 DATA QUALITY IMPROVEMENTS:")
            
            # Missing values in critical columns
            critical_cols = ['name', 'issuer', 'annual_fee']
            for col in critical_cols:
                if col in self.df.columns:
                    missing = self.df[col].isnull().sum()
                    print(f"   {col}: {missing}/{len(self.df)} missing ({missing/len(self.df)*100:.1f}%)")
            
            # Show feature summary
            if 'card_summary' in self.df.columns:
                avg_summary_length = self.df['card_summary'].str.len().mean()
                print(f"   📝 Average card summary length: {avg_summary_length:.0f} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving cleaned dataset: {e}")
            return False
    
    def generate_cleaning_report(self):
        """Generate a cleaning report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"NAVUS/Reports/dataset_cleaning_report_{timestamp}.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("DATASET CLEANING REPORT\\n")
                f.write("=" * 40 + "\\n\\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Input: {self.input_file}\\n")
                f.write(f"Output: {self.output_file}\\n\\n")
                
                f.write("CLEANING ACTIONS PERFORMED:\\n")
                f.write("- Filled missing name and issuer values\\n")
                f.write("- Cleaned and standardized numeric columns\\n")
                f.write("- Standardized issuer names\\n")
                f.write("- Cleaned text fields\\n")
                f.write("- Added computed features for LLM training\\n")
                f.write("- Optimized column order and structure\\n")
                f.write("- Created card summary for embeddings\\n\\n")
                
                f.write(f"FINAL DATASET STATISTICS:\\n")
                f.write(f"- Total cards: {len(self.df)}\\n")
                f.write(f"- Total columns: {len(self.df.columns)}\\n")
                f.write(f"- Unique issuers: {self.df['issuer'].nunique() if 'issuer' in self.df.columns else 'N/A'}\\n")
                f.write(f"- Unique categories: {self.df['category'].nunique() if 'category' in self.df.columns else 'N/A'}\\n")
            
            print(f"📄 Cleaning report saved: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"⚠️  Could not save cleaning report: {e}")
            return None
    
    def run_full_cleaning(self):
        """Run complete cleaning process."""
        print("🧹 MASTER DATASET CLEANING STARTING...")
        print("=" * 50)
        
        if not self.load_data():
            return False
        
        # Run all cleaning steps
        self.clean_missing_values()
        self.clean_numeric_columns()
        self.standardize_issuers()
        self.clean_text_fields()
        self.add_computed_features()
        self.optimize_for_llm()
        
        # Save results
        success = self.save_cleaned_dataset()
        if success:
            self.generate_cleaning_report()
        
        return success

def main():
    """Main cleaning function."""
    cleaner = MasterDatasetCleaner()
    success = cleaner.run_full_cleaning()
    
    if success:
        print(f"\\n🎉 CLEANING COMPLETE!")
        print(f"✅ Your dataset is now optimized for LLM training!")
        print(f"📁 Use: NAVUS/master_card_dataset_cleaned.csv")
    else:
        print(f"\\n❌ Cleaning failed - check error messages above")
    
    return success

if __name__ == "__main__":
    main()