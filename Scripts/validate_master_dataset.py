#!/usr/bin/env python3
"""
Master Dataset Validation Script
Cross-check master dataset against individual CSV files for LLM training readiness.
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import re

class MasterDatasetValidator:
    def __init__(self):
        self.master_file = "NAVUS/master_card_dataset_cleaned.csv"
        self.individual_files = glob.glob("NAVUS/CSV/*.csv")
        self.validation_report = []
        self.errors = []
        self.warnings = []
        
    def log(self, message, level="INFO"):
        """Log validation messages."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {level}: {message}"
        print(full_message)
        self.validation_report.append(full_message)
        
        if level == "ERROR":
            self.errors.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
    
    def load_datasets(self):
        """Load master dataset and individual datasets."""
        self.log("Loading datasets for validation...")
        
        try:
            # Load master dataset
            self.master_df = pd.read_csv(self.master_file, encoding='utf-8')
            self.log(f"✅ Master dataset loaded: {len(self.master_df)} rows, {len(self.master_df.columns)} columns")
            
            # Load individual datasets
            self.individual_dfs = {}
            total_individual_rows = 0
            
            for file_path in self.individual_files:
                filename = os.path.basename(file_path)
                if filename == "master_card_dataset.csv":
                    continue  # Skip the master file itself
                    
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    self.individual_dfs[filename] = df
                    total_individual_rows += len(df)
                    self.log(f"✅ Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    self.log(f"❌ Failed to load {filename}: {e}", "ERROR")
            
            self.log(f"📊 Total individual files: {len(self.individual_dfs)}")
            self.log(f"📊 Total individual rows: {total_individual_rows}")
            self.log(f"📊 Master dataset rows: {len(self.master_df)}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to load datasets: {e}", "ERROR")
            return False
    
    def validate_row_counts(self):
        """Validate that master dataset contains expected number of rows."""
        self.log("\n🔢 VALIDATING ROW COUNTS...")
        
        # Calculate expected total (may have duplicates between files)
        total_individual_rows = sum(len(df) for df in self.individual_dfs.values())
        master_rows = len(self.master_df)
        
        self.log(f"Individual files total: {total_individual_rows} rows")
        self.log(f"Master dataset: {master_rows} rows")
        
        if master_rows <= total_individual_rows:
            self.log("✅ Row count validation passed (duplicates were properly handled)")
        else:
            self.log(f"⚠️  Master has more rows than sum of individuals - possible data duplication", "WARNING")
        
        # Check source file tracking
        if 'source_file' in self.master_df.columns:
            source_counts = self.master_df['source_file'].value_counts()
            self.log(f"📋 Rows by source file:")
            for source, count in source_counts.items():
                expected = len(self.individual_dfs.get(source, pd.DataFrame()))
                status = "✅" if count <= expected else "⚠️"
                self.log(f"   {status} {source}: {count} rows (expected ≤{expected})")
        else:
            self.log("⚠️  No source_file column found for tracking", "WARNING")
    
    def validate_data_integrity(self):
        """Validate data integrity and consistency."""
        self.log("\n🔍 VALIDATING DATA INTEGRITY...")
        
        # Check for completely empty rows
        empty_rows = self.master_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            self.log(f"❌ Found {empty_rows} completely empty rows", "ERROR")
        else:
            self.log("✅ No completely empty rows found")
        
        # Check for duplicate rows
        duplicate_rows = self.master_df.duplicated().sum()
        if duplicate_rows > 0:
            self.log(f"⚠️  Found {duplicate_rows} duplicate rows", "WARNING")
            # Show sample duplicates
            duplicates = self.master_df[self.master_df.duplicated(keep=False)]
            if 'name' in duplicates.columns:
                dup_names = duplicates['name'].value_counts().head(3)
                self.log(f"   Sample duplicates: {list(dup_names.index)}")
        else:
            self.log("✅ No duplicate rows found")
        
        # Validate critical columns
        critical_columns = ['name', 'issuer', 'annual_fee']
        for col in critical_columns:
            if col in self.master_df.columns:
                null_count = self.master_df[col].isnull().sum()
                empty_count = (self.master_df[col] == '').sum() if self.master_df[col].dtype == 'object' else 0
                total_missing = null_count + empty_count
                
                if total_missing > len(self.master_df) * 0.5:  # More than 50% missing
                    self.log(f"❌ Critical column '{col}' has {total_missing}/{len(self.master_df)} missing values", "ERROR")
                elif total_missing > 0:
                    self.log(f"⚠️  Column '{col}' has {total_missing}/{len(self.master_df)} missing values", "WARNING")
                else:
                    self.log(f"✅ Column '{col}' has no missing values")
            else:
                self.log(f"⚠️  Critical column '{col}' not found", "WARNING")
    
    def validate_data_types(self):
        """Validate data types for LLM training compatibility."""
        self.log("\n📝 VALIDATING DATA TYPES FOR LLM TRAINING...")
        
        # Check numeric columns
        numeric_columns = ['annual_fee', 'purchase_rate', 'balance_transfer_rate', 'cash_advance_rate', 
                          'min_income', 'welcome_bonus', 'cashback_rate']
        
        for col in numeric_columns:
            if col in self.master_df.columns:
                non_numeric = pd.to_numeric(self.master_df[col], errors='coerce').isnull().sum()
                total_non_null = self.master_df[col].notna().sum()
                
                if non_numeric > 0 and total_non_null > 0:
                    self.log(f"⚠️  Column '{col}': {non_numeric} non-numeric values out of {total_non_null} non-null", "WARNING")
                    # Show sample non-numeric values
                    non_num_values = self.master_df[col][pd.to_numeric(self.master_df[col], errors='coerce').isnull() & self.master_df[col].notna()]
                    if len(non_num_values) > 0:
                        sample_values = list(non_num_values.unique())[:3]
                        self.log(f"   Sample non-numeric values: {sample_values}")
                else:
                    self.log(f"✅ Column '{col}': All values are numeric or null")
        
        # Check text columns for LLM training
        text_columns = ['name', 'issuer', 'features', 'rewards_type', 'category']
        
        for col in text_columns:
            if col in self.master_df.columns:
                # Check for very long text that might cause issues
                if self.master_df[col].dtype == 'object':
                    max_length = self.master_df[col].astype(str).str.len().max()
                    avg_length = self.master_df[col].astype(str).str.len().mean()
                    
                    if max_length > 1000:
                        self.log(f"⚠️  Column '{col}': Very long text (max {max_length} chars)", "WARNING")
                    else:
                        self.log(f"✅ Column '{col}': Text length OK (max {max_length}, avg {avg_length:.1f} chars)")
                    
                    # Check for special characters that might cause issues
                    special_chars = self.master_df[col].astype(str).str.contains(r'[^\w\s\-\.,\(\)&]', regex=True, na=False).sum()
                    if special_chars > 0:
                        self.log(f"ℹ️  Column '{col}': {special_chars} entries contain special characters")
    
    def validate_business_logic(self):
        """Validate business logic and data consistency."""
        self.log("\n💼 VALIDATING BUSINESS LOGIC...")
        
        # Check annual fee ranges
        if 'annual_fee' in self.master_df.columns:
            fee_col = pd.to_numeric(self.master_df['annual_fee'], errors='coerce')
            
            negative_fees = (fee_col < 0).sum()
            if negative_fees > 0:
                self.log(f"❌ Found {negative_fees} cards with negative annual fees", "ERROR")
            
            very_high_fees = (fee_col > 2000).sum()
            if very_high_fees > 0:
                self.log(f"⚠️  Found {very_high_fees} cards with very high annual fees (>$2000)", "WARNING")
                high_fee_cards = self.master_df[fee_col > 2000]
                if 'name' in high_fee_cards.columns:
                    sample_names = high_fee_cards['name'].head(3).tolist()
                    self.log(f"   Sample high-fee cards: {sample_names}")
            
            fee_stats = fee_col.describe()
            self.log(f"📊 Annual fee statistics: Min=${fee_stats['min']:.0f}, Max=${fee_stats['max']:.0f}, Avg=${fee_stats['mean']:.0f}")
        
        # Check interest rate ranges
        rate_columns = ['purchase_rate', 'balance_transfer_rate', 'cash_advance_rate']
        for col in rate_columns:
            if col in self.master_df.columns:
                rate_col = pd.to_numeric(self.master_df[col], errors='coerce')
                
                invalid_rates = ((rate_col < 0) | (rate_col > 50)).sum()
                if invalid_rates > 0:
                    self.log(f"⚠️  Column '{col}': {invalid_rates} rates outside normal range (0-50%)", "WARNING")
                
                if rate_col.notna().sum() > 0:
                    rate_stats = rate_col.describe()
                    self.log(f"📊 {col} statistics: Min={rate_stats['min']:.1f}%, Max={rate_stats['max']:.1f}%, Avg={rate_stats['mean']:.1f}%")
        
        # Check issuer consistency
        if 'issuer' in self.master_df.columns:
            issuers = self.master_df['issuer'].value_counts()
            self.log(f"🏦 Found {len(issuers)} unique issuers")
            
            # Check for potential issuer name inconsistencies
            issuer_names = [str(name).lower().strip() for name in issuers.index if pd.notna(name)]
            similar_names = []
            for i, name1 in enumerate(issuer_names):
                for name2 in issuer_names[i+1:]:
                    if name1 != name2 and (name1 in name2 or name2 in name1):
                        similar_names.append((name1, name2))
            
            if similar_names:
                self.log(f"⚠️  Found potentially similar issuer names:", "WARNING")
                for name1, name2 in similar_names[:3]:  # Show first 3
                    self.log(f"   '{name1}' vs '{name2}'")
    
    def validate_llm_readiness(self):
        """Validate dataset readiness for LLM training."""
        self.log("\n🤖 VALIDATING LLM TRAINING READINESS...")
        
        # Check data diversity
        total_cards = len(self.master_df)
        
        # Category diversity
        if 'category' in self.master_df.columns:
            categories = self.master_df['category'].nunique()
            self.log(f"📊 Category diversity: {categories} unique categories across {total_cards} cards")
            if categories < 5:
                self.log(f"⚠️  Limited category diversity may affect recommendation quality", "WARNING")
        
        # Issuer diversity
        if 'issuer' in self.master_df.columns:
            issuers = self.master_df['issuer'].nunique()
            self.log(f"🏦 Issuer diversity: {issuers} unique issuers")
            if issuers < 5:
                self.log(f"⚠️  Limited issuer diversity may affect recommendation quality", "WARNING")
        
        # Feature richness
        feature_columns = ['features', 'rewards_type', 'network', 'category', 'annual_fee', 'purchase_rate']
        available_features = sum(1 for col in feature_columns if col in self.master_df.columns)
        self.log(f"📋 Feature richness: {available_features}/{len(feature_columns)} key feature columns present")
        
        # Check for training/validation split readiness
        self.log(f"📏 Dataset size assessment:")
        if total_cards < 20:
            self.log(f"⚠️  Small dataset ({total_cards} cards) - consider gathering more data", "WARNING")
        elif total_cards < 50:
            self.log(f"ℹ️  Medium dataset ({total_cards} cards) - suitable for basic recommendations")
        else:
            self.log(f"✅ Large dataset ({total_cards} cards) - excellent for comprehensive recommendations")
        
        # Check text quality for embeddings
        text_cols_for_embeddings = ['name', 'features', 'category', 'rewards_type']
        for col in text_cols_for_embeddings:
            if col in self.master_df.columns:
                non_empty = self.master_df[col].notna().sum()
                coverage = (non_empty / total_cards) * 100
                
                if coverage < 50:
                    self.log(f"⚠️  Column '{col}': Only {coverage:.1f}% coverage - may limit embedding quality", "WARNING")
                else:
                    self.log(f"✅ Column '{col}': {coverage:.1f}% coverage - good for embeddings")
    
    def compare_with_individuals(self):
        """Compare master dataset against individual files."""
        self.log("\n🔄 COMPARING WITH INDIVIDUAL FILES...")
        
        for filename, individual_df in self.individual_dfs.items():
            self.log(f"\n📊 Comparing with {filename}:")
            
            # Find rows from this source in master dataset
            if 'source_file' in self.master_df.columns:
                master_subset = self.master_df[self.master_df['source_file'] == filename]
                
                self.log(f"   Individual file: {len(individual_df)} rows")
                self.log(f"   In master dataset: {len(master_subset)} rows")
                
                if len(master_subset) == 0:
                    self.log(f"   ❌ No data from {filename} found in master dataset", "ERROR")
                    continue
                
                # Compare key columns that should exist in both
                common_columns = set(individual_df.columns) & set(master_subset.columns)
                common_columns.discard('source_file')  # Added by master dataset
                
                if common_columns:
                    self.log(f"   📋 Common columns: {len(common_columns)}")
                    
                    # Sample comparison for key fields
                    key_fields = ['name', 'issuer', 'annual_fee', 'category'] 
                    for field in key_fields:
                        if field in common_columns:
                            # Check if values are preserved
                            individual_values = set(individual_df[field].dropna().astype(str))
                            master_values = set(master_subset[field].dropna().astype(str))
                            
                            if individual_values.issubset(master_values):
                                self.log(f"   ✅ Field '{field}': All values preserved")
                            else:
                                missing = individual_values - master_values
                                self.log(f"   ⚠️  Field '{field}': {len(missing)} values not found in master", "WARNING")
                                if missing:
                                    sample_missing = list(missing)[:3]
                                    self.log(f"      Sample missing: {sample_missing}")
                else:
                    self.log(f"   ⚠️  No common columns found", "WARNING")
            else:
                self.log(f"   ⚠️  Cannot track source - no source_file column", "WARNING")
    
    def generate_summary_report(self):
        """Generate final validation summary."""
        self.log("\n📋 VALIDATION SUMMARY REPORT")
        self.log("=" * 50)
        
        self.log(f"✅ Total validation checks completed")
        self.log(f"❌ Errors found: {len(self.errors)}")
        self.log(f"⚠️  Warnings found: {len(self.warnings)}")
        
        if self.errors:
            self.log(f"\n❌ CRITICAL ERRORS:")
            for error in self.errors:
                self.log(f"   - {error}")
        
        if self.warnings:
            self.log(f"\n⚠️  WARNINGS:")
            for warning in self.warnings[:10]:  # Show first 10 warnings
                self.log(f"   - {warning}")
            if len(self.warnings) > 10:
                self.log(f"   ... and {len(self.warnings) - 10} more warnings")
        
        # Overall assessment
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                self.log(f"\n🎉 EXCELLENT: Dataset is ready for LLM training!")
                quality_score = "EXCELLENT"
            elif len(self.warnings) <= 5:
                self.log(f"\n✅ GOOD: Dataset is suitable for LLM training with minor issues")
                quality_score = "GOOD"
            else:
                self.log(f"\n⚠️  FAIR: Dataset can be used but has several issues to consider")
                quality_score = "FAIR"
        else:
            self.log(f"\n❌ POOR: Dataset has critical errors that should be fixed before training")
            quality_score = "POOR"
        
        return quality_score
    
    def save_validation_report(self, quality_score):
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"NAVUS/Reports/master_dataset_validation_{timestamp}.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("MASTER DATASET VALIDATION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.master_file}\n")
                f.write(f"Quality Score: {quality_score}\n\n")
                
                f.write(f"SUMMARY:\n")
                f.write(f"- Total Errors: {len(self.errors)}\n")
                f.write(f"- Total Warnings: {len(self.warnings)}\n")
                f.write(f"- Master Dataset: {len(self.master_df)} cards, {len(self.master_df.columns)} columns\n\n")
                
                f.write("DETAILED LOG:\n")
                f.write("-" * 30 + "\n")
                for line in self.validation_report:
                    f.write(line + "\n")
            
            self.log(f"\n💾 Validation report saved: {report_file}")
            return report_file
            
        except Exception as e:
            self.log(f"❌ Failed to save validation report: {e}", "ERROR")
            return None
    
    def run_full_validation(self):
        """Run complete validation suite."""
        self.log("🔍 MASTER DATASET VALIDATION STARTING...")
        self.log("=" * 60)
        
        # Load data
        if not self.load_datasets():
            return False
        
        # Run all validations
        self.validate_row_counts()
        self.validate_data_integrity()
        self.validate_data_types()
        self.validate_business_logic()
        self.validate_llm_readiness()
        self.compare_with_individuals()
        
        # Generate summary
        quality_score = self.generate_summary_report()
        
        # Save report
        self.save_validation_report(quality_score)
        
        return quality_score

def main():
    """Main validation function."""
    validator = MasterDatasetValidator()
    quality_score = validator.run_full_validation()
    
    if quality_score:
        print(f"\n🎯 FINAL ASSESSMENT: {quality_score}")
        
        if quality_score in ["EXCELLENT", "GOOD"]:
            print("✅ Your dataset is ready for LLM training! 🚀")
        elif quality_score == "FAIR":
            print("⚠️  Dataset can be used but consider addressing warnings for better quality")
        else:
            print("❌ Please fix critical errors before using for LLM training")
    
    return quality_score

if __name__ == "__main__":
    main()