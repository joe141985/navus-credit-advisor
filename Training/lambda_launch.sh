#!/bin/bash
# Lambda Labs Training Launch Script

echo "🚀 NAVUS Mistral 7B Fine-tuning on Lambda Labs"
echo "=============================================="

# Update system and install dependencies
sudo apt update
sudo apt install -y git python3-pip

# Install Python packages
pip install -r requirements.txt

# Login to Weights & Biases (optional - replace with your API key)
# wandb login YOUR_WANDB_API_KEY

# Download training data (make sure to upload your data first)
echo "📥 Training data should be uploaded to the Lambda instance"
ls -la *.jsonl

# Start training
echo "🔥 Starting Mistral 7B fine-tuning..."
python train_mistral.py

echo "✅ Training complete!"
echo "💾 Model saved in ./navus_mistral_finetuned/"

# Optionally compress the model for download
echo "📦 Compressing model for download..."
tar -czf navus_mistral_finetuned.tar.gz navus_mistral_finetuned/

echo "🎉 All done! Download navus_mistral_finetuned.tar.gz to your local machine"
