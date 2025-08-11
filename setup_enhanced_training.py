"""
Enhanced NAVUS Setup Script
Optimized for MacBook training environment with all dependencies
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def install_requirements():
    """Install all required packages for enhanced NAVUS training"""
    
    requirements = [
        "streamlit>=1.28.0",
        "torch>=2.0.0", 
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "accelerate>=0.21.0"
    ]
    
    print("🚀 Installing Enhanced NAVUS Dependencies...")
    print("="*60)
    
    for package in requirements:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories for training"""
    
    directories = [
        "Training/models",
        "Training/logs", 
        "Training/checkpoints",
        "WebApp/static",
        "WebApp/templates",
        "Scripts/charts",
        "Data/processed"
    ]
    
    print("\n📁 Setting up directory structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True

def create_requirements_file():
    """Create requirements.txt for the enhanced setup"""
    
    requirements_content = """# Enhanced NAVUS Training Requirements
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
accelerate>=0.21.0
jupyter>=1.0.0
ipywidgets>=8.0.0
"""
    
    with open("requirements_enhanced.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Created requirements_enhanced.txt")

def create_training_config():
    """Create training configuration file"""
    
    config = {
        "model_settings": {
            "base_model": "microsoft/DialoGPT-medium",
            "output_dir": "./enhanced_navus_model",
            "max_length": 512,
            "batch_size": 2,  # MacBook optimized
            "epochs": 2,
            "learning_rate": 5e-5,
            "warmup_steps": 100
        },
        "data_settings": {
            "original_data": "navus_alpaca_format.json",
            "enhanced_data": "enhanced_debt_payoff_dataset.json",
            "card_database": "../Data/master_card_dataset_cleaned.csv"
        },
        "chart_settings": {
            "output_format": "png",
            "dpi": 300,
            "style": "seaborn-v0_8-darkgrid"
        },
        "macbook_optimizations": {
            "use_mps": True,
            "gradient_accumulation_steps": 4,
            "dataloader_pin_memory": False,
            "fp16": False
        }
    }
    
    with open("Training/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created training configuration")

def test_environment():
    """Test if the environment is properly set up"""
    
    print("\n🧪 Testing environment setup...")
    
    # Test imports
    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("streamlit", "Streamlit"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas")
    ]
    
    all_good = True
    
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_good = False
    
    # Test PyTorch MPS (Apple Silicon optimization)
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon GPU) available")
        else:
            print("ℹ️  MPS not available, will use CPU")
    except:
        print("⚠️  Could not check MPS availability")
    
    return all_good

def create_launch_scripts():
    """Create convenient launch scripts"""
    
    # Training launcher
    training_script = """#!/bin/bash
echo "🚀 Launching Enhanced NAVUS Training..."
echo "=================================="

# Change to Training directory
cd Training

# Run training with progress monitoring
python train_enhanced_navus.py 2>&1 | tee training_log.txt

echo "✅ Training completed!"
echo "📁 Model saved to: ./enhanced_navus_model"
echo "📄 Log saved to: training_log.txt"
"""
    
    with open("launch_training.sh", "w") as f:
        f.write(training_script)
    
    os.chmod("launch_training.sh", 0o755)
    
    # Web app launcher
    webapp_script = """#!/bin/bash
echo "🌐 Launching Enhanced NAVUS Web App..."
echo "====================================="

# Change to WebApp directory
cd WebApp

# Launch Streamlit app
streamlit run enhanced_backend.py --server.port 8501

echo "🌍 Access the app at: http://localhost:8501"
"""
    
    with open("launch_webapp.sh", "w") as f:
        f.write(webapp_script)
    
    os.chmod("launch_webapp.sh", 0o755)
    
    print("✅ Created launch scripts")

def create_dataset_merger():
    """Create script to merge all training datasets"""
    
    merger_script = """
import json
import pandas as pd
from pathlib import Path

def merge_all_datasets():
    \"\"\"Merge all available training datasets\"\"\"
    
    all_data = []
    
    # Load original NAVUS data
    if Path('navus_alpaca_format.json').exists():
        with open('navus_alpaca_format.json', 'r') as f:
            original_data = json.load(f)
            all_data.extend(original_data)
            print(f"✅ Loaded {len(original_data)} original examples")
    
    # Load enhanced debt payoff data
    if Path('enhanced_debt_payoff_dataset.json').exists():
        with open('enhanced_debt_payoff_dataset.json', 'r') as f:
            enhanced_data = json.load(f)
            all_data.extend(enhanced_data)
            print(f"✅ Loaded {len(enhanced_data)} enhanced examples")
    
    # Save merged dataset
    with open('merged_training_dataset.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"🎯 Total training examples: {len(all_data)}")
    print("✅ Saved merged dataset to: merged_training_dataset.json")
    
    return len(all_data)

if __name__ == "__main__":
    merge_all_datasets()
"""
    
    with open("Training/merge_datasets.py", "w") as f:
        f.write(merger_script)
    
    print("✅ Created dataset merger script")

def main():
    """Main setup function"""
    
    print("🏦 NAVUS Enhanced Training Setup")
    print("="*50)
    print("🍎 MacBook Optimized Training Environment")
    print("📊 With Chart Generation & Advanced Analytics")
    print("="*50)
    
    steps = [
        ("📦 Installing Dependencies", install_requirements),
        ("📁 Setting up Directories", setup_directories), 
        ("📄 Creating Requirements File", create_requirements_file),
        ("⚙️  Creating Training Config", create_training_config),
        ("🚀 Creating Launch Scripts", create_launch_scripts),
        ("🔗 Creating Dataset Merger", create_dataset_merger),
        ("🧪 Testing Environment", test_environment)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if step_func():
                success_count += 1
            else:
                print(f"❌ {step_name} failed")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("\n🎉 ENHANCED NAVUS SETUP COMPLETE!")
        print("\n📋 Next Steps:")
        print("1. Run: ./launch_training.sh    (to start training)")
        print("2. Run: ./launch_webapp.sh      (to start web app)")
        print("3. Check Training/ directory for logs and models")
        print("4. Access charts at http://localhost:8501")
        
        print("\n💡 Tips for MacBook Training:")
        print("- Keep laptop plugged in during training")
        print("- Close unnecessary apps to free up RAM")
        print("- Training will use Apple Silicon GPU if available")
        print("- Expect 1-3 hours for full training depending on data size")
        
    else:
        print("\n⚠️  Some setup steps failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements_enhanced.txt")

if __name__ == "__main__":
    main()