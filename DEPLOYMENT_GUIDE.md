# 🚀 NAVUS LLM Training & Deployment Guide

## Your Technical Cofounder's Complete Step-by-Step Guide

Congratulations! You now have a complete LLM training and web app pipeline. Here's how to deploy your NAVUS credit card advisor from start to finish.

---

## 📋 **PHASE 1: LAMBDA LABS TRAINING**

### Step 1: Sign up for Lambda Labs
```bash
# Go to https://lambdalabs.com/
# Sign up for an account
# Add payment method (training will cost ~$5-15)
```

### Step 2: Launch Lambda Labs Instance
```bash
# In Lambda Labs dashboard:
# 1. Click "Launch Instance"
# 2. Select "1x RTX 4090" or "1x A100" (recommended)
# 3. Choose "PyTorch" image
# 4. Launch the instance
```

### Step 3: Upload Training Files
```bash
# In your NAVUS directory, copy these files to Lambda Labs:
cd /Users/joebanerjee/NAVUS

# Files you need to upload to Lambda Labs:
Training/navus_chat_format.jsonl
Training/train_mistral.py
Training/requirements.txt
Training/lambda_launch.sh
```

**Upload Method:**
- Use the Lambda Labs web interface file upload
- Or use SCP: `scp -i key.pem file.txt ubuntu@instance-ip:/home/ubuntu/`

### Step 4: Run Training on Lambda Labs
```bash
# SSH into your Lambda Labs instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Make the launch script executable and run it
chmod +x lambda_launch.sh
./lambda_launch.sh
```

**Training will take 2-4 hours and cost approximately $8-15.**

### Step 5: Download Trained Model
```bash
# After training completes, download the model
scp -r -i key.pem ubuntu@instance-ip:/home/ubuntu/navus_mistral_finetuned ./

# Or download the compressed version:
scp -i key.pem ubuntu@instance-ip:/home/ubuntu/navus_mistral_finetuned.tar.gz ./
tar -xzf navus_mistral_finetuned.tar.gz
```

---

## 🖥️ **PHASE 2: LOCAL TESTING**

### Step 6: Test Your Model Locally
```bash
cd /Users/joebanerjee/NAVUS/Training

# Install requirements locally
pip install torch transformers peft accelerate

# Test the model
python navus_inference.py
```

**Expected output:**
```
🔄 Loading NAVUS Credit Advisor...
✅ NAVUS Credit Advisor loaded!
🧪 Testing with sample questions...

Q: What's the best no-fee travel card in Canada?
A: I'd recommend the Capital One Aspire Travel Mastercard...
```

---

## 🌐 **PHASE 3: WEB APP DEPLOYMENT**

### Step 7: Set Up Web App
```bash
cd /Users/joebanerjee/NAVUS/WebApp

# Copy your trained model to the WebApp directory
cp -r ../Training/navus_mistral_finetuned ./

# Install Python backend dependencies
pip install -r requirements.txt

# Install Node.js if you don't have it
# Download from: https://nodejs.org/
```

### Step 8: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 9: Launch the Complete App
```bash
# From the WebApp directory:
./run_app.sh
```

**Expected output:**
```
🚀 Starting NAVUS Credit Card Advisor Web App
📦 Installing Python dependencies...
🔥 Starting backend API server...
🎨 Starting frontend development server...
✅ Starting React development server...

🎉 NAVUS is now running!
📱 Frontend: http://localhost:3000
🔧 Backend API: http://localhost:8000
```

---

## 💡 **PHASE 4: DEMO PREPARATION**

### Step 10: Test Your MVP
Visit `http://localhost:3000` and test these questions:
- "What's the best travel card for someone making $60,000?"
- "I'm a student looking for my first credit card"  
- "Best no-fee cashback card for groceries?"
- "Compare RBC vs TD travel cards"

### Step 11: Create Demo Script
Here's your investor demo flow:

1. **Open the app** - Show clean, professional interface
2. **Ask travel question** - Demonstrate travel card expertise
3. **Ask student question** - Show demographic targeting
4. **Ask comparison** - Show analytical capabilities
5. **Show responsive design** - Test on mobile/tablet

---

## 📊 **COST BREAKDOWN**

| Component | Cost | Duration |
|-----------|------|----------|
| Lambda Labs Training | $8-15 | One-time |
| Local Development | $0 | Ongoing |
| Web Hosting (Optional) | $5-20/month | If deployed |
| **TOTAL MVP COST** | **$8-15** | **One-time** |

---

## 🔧 **TROUBLESHOOTING**

### Model Not Loading
```bash
# If model fails to load, use base model fallback:
# Edit backend.py line 45, change model_path to:
model_path = "mistralai/Mistral-7B-Instruct-v0.3"
```

### CUDA Memory Issues
```bash
# Reduce batch size in train_mistral.py:
per_device_train_batch_size=1  # Already set to 1
gradient_accumulation_steps=8  # Increase this instead
```

### Frontend Won't Start
```bash
cd WebApp/frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

---

## 🚀 **NEXT STEPS FOR SCALING**

1. **Deploy to cloud** (AWS/GCP) for public access
2. **Add user accounts** and conversation history
3. **Integrate real-time rates** from bank APIs
4. **Add file upload** for personal financial data
5. **Implement A/B testing** for different responses

---

## 📞 **SUPPORT**

If you encounter issues:

1. **Check the logs** in terminal output
2. **Verify file paths** match your system
3. **Test components individually** (model → API → frontend)
4. **Check port availability** (3000, 8000)

---

## 🎯 **SUCCESS METRICS FOR INVESTORS**

- **Response Accuracy**: Model provides relevant card recommendations
- **User Experience**: Clean, fast, intuitive chat interface  
- **Technical Robustness**: Handles edge cases and errors gracefully
- **Scalability**: Architecture supports growth and additional features
- **Cost Efficiency**: Low operational costs with high-value output

**Your NAVUS MVP demonstrates enterprise-ready AI capabilities at startup costs!** 🚀

---

*Created by your Technical AI Cofounder*