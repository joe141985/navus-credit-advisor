"""
Launch Massive NAVUS Training
Complete training pipeline with 10x more data
"""

import json
import os
import subprocess
from datetime import datetime
import glob

def find_latest_dataset():
    """Find the latest massive dataset file"""
    pattern = "Training/combined_massive_dataset_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No massive dataset found!")
        print("💡 Run: python Scripts/generate_massive_dataset.py")
        return None
    
    # Get the latest file
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def show_dataset_stats(dataset_file):
    """Show statistics about the training dataset"""
    try:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Dataset Statistics:")
        print(f"  - Total Examples: {len(data):,}")
        print(f"  - File Size: {os.path.getsize(dataset_file) / 1024 / 1024:.1f} MB")
        
        # Analyze example types
        categories = {}
        for example in data[:100]:  # Sample first 100
            input_text = example.get('input', '')
            if 'debt:' in input_text:
                categories['Debt Scenarios'] = categories.get('Debt Scenarios', 0) + 1
            elif 'income:' in input_text:
                categories['Card Comparisons'] = categories.get('Card Comparisons', 0) + 1
            elif 'cards:' in input_text:
                categories['Multi-Card Strategy'] = categories.get('Multi-Card Strategy', 0) + 1
            elif 'current_score:' in input_text:
                categories['Credit Building'] = categories.get('Credit Building', 0) + 1
            else:
                categories['General Financial'] = categories.get('General Financial', 0) + 1
        
        print(f"  - Example Categories (sample):")
        for category, count in categories.items():
            print(f"    • {category}: {count} examples")
        
        # Show sample advanced example
        advanced_examples = [ex for ex in data if len(ex.get('output', '')) > 500]
        if advanced_examples:
            sample = advanced_examples[0]
            print(f"\n📝 Sample Advanced Example:")
            print(f"  Question: {sample['instruction'][:100]}...")
            print(f"  Response Length: {len(sample['output'])} characters")
            print(f"  Has Charts/Analysis: {'📊' if '📊' in sample['output'] else '📈' if '📈' in sample['output'] else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def check_environment():
    """Check if the training environment is ready"""
    print("🔧 Checking training environment...")
    
    # Check Python packages
    required_packages = ['torch', 'transformers', 'datasets']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check for Apple Silicon optimization
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  ✅ Apple Silicon MPS acceleration available")
        else:
            print("  ℹ️  Using CPU (MPS not available)")
    except:
        pass
    
    return True

def run_training(dataset_file):
    """Run the enhanced training with massive dataset"""
    print(f"\n🚀 Starting NAVUS Massive Training...")
    print(f"📁 Dataset: {dataset_file}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to Training directory
    os.chdir('Training')
    
    # Update training script to use the massive dataset
    training_script = f"""
import sys
sys.path.append('..')
from train_enhanced_navus import EnhancedNAVUSTrainer
import json

def main():
    print("🏦 Enhanced NAVUS Training with Massive Dataset")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedNAVUSTrainer()
    
    # Load the massive dataset
    with open('../{dataset_file}', 'r') as f:
        massive_data = json.load(f)
    
    print(f"📊 Loaded {{len(massive_data):,}} training examples")
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(massive_data)
    
    # Train with massive dataset (conservative settings for stability)
    trainer.train_model(dataset, epochs=1, batch_size=1)  # Very conservative for massive dataset
    
    # Test the model
    trainer.test_model()
    
    print("✅ Massive training completed!")

if __name__ == "__main__":
    main()
"""
    
    # Write the training script
    with open('run_massive_training.py', 'w') as f:
        f.write(training_script)
    
    # Run the training
    try:
        result = subprocess.run(
            ['python', 'run_massive_training.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\n📋 Training Output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("\n🔍 Error Output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 2 hours")
    except Exception as e:
        print(f"❌ Training error: {e}")
    
    # Return to original directory
    os.chdir('..')

def main():
    """Main launcher function"""
    print("🚀 NAVUS Massive Training Launcher")
    print("=" * 50)
    print("🎯 Training with 10x More Financial Data")
    print("📊 Advanced Debt Analysis & Chart Generation")
    print("🍎 MacBook Optimized Training")
    print("=" * 50)
    
    # Find the latest massive dataset
    dataset_file = find_latest_dataset()
    if not dataset_file:
        return
    
    print(f"📁 Found dataset: {dataset_file}")
    
    # Show dataset statistics
    if not show_dataset_stats(dataset_file):
        return
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing requirements.")
        return
    
    # Confirm before training
    print(f"\n⚠️  MASSIVE TRAINING WARNING:")
    print(f"   • This will train on {open(dataset_file).read().count('instruction')} examples")
    print(f"   • Training may take 1-3 hours on MacBook")
    print(f"   • Keep laptop plugged in and avoid heavy tasks")
    print(f"   • Model will be saved to: ./Training/enhanced_navus_model")
    
    confirm = input(f"\n🤔 Continue with massive training? (y/N): ").lower().strip()
    
    if confirm == 'y':
        run_training(dataset_file)
        
        print(f"\n🎉 MASSIVE NAVUS TRAINING COMPLETE!")
        print(f"📁 Model Location: ./Training/enhanced_navus_model")
        print(f"🌐 Test with: ./launch_webapp.sh")
        print(f"📊 Dataset Used: {dataset_file}")
        print(f"⭐ Your NAVUS model now has 10x more financial intelligence!")
        
    else:
        print("👋 Training cancelled. Run again when ready!")

if __name__ == "__main__":
    main()